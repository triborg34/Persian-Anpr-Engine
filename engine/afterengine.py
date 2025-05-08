import gc
import logging
import os
import sys
import time
import numpy as np
import cv2
import warnings
import torch
import asyncio
import base64
import threading
import json
import statistics
from ultralytics import YOLO
from configParams import Parameters
from database.db_entries_utils import db_entries_time
import websockets
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from camera import FreshestFrame

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CCTV-Server")
logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Parameters
params = Parameters()
port = int(params.socketport)
host = '0.0.0.0'

# Device setup
device = torch.device(0 if torch.cuda.is_available() else "cpu")
logger.info(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device.")
logger.info(f"Version : 10.0.2 Up 5/8/2025")

# A dictionary to store FreshestFrame objects for each RTSP source
camera_feeds = {}

# A dictionary to store active client connections
active_connections = {}

# Frame rate limiter (FPS)
TARGET_FPS = 15  # Adjust based on your needs
FRAME_DELAY = 1.0 / TARGET_FPS

def restart_program():
    try:
        logger.warning("Restarting program... Cleaning up first.")
        
        # Stop camera capture if possible
        for cam_key, cam in camera_feeds.items():
            try:
                if hasattr(cam, 'stop') and callable(cam.stop):
                    cam.stop()
                logger.info(f"Stopped camera feed {cam_key}")
            except Exception as e:
                logger.error(f"Error stopping camera {cam_key}: {e}")
        
        # GPU and memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Cleanup complete, restarting application...")
        
        # Restart
        if getattr(sys, 'frozen', False):
            # Running as .exe (frozen by PyInstaller or similar)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            # Running as .py script with Python
            os.execv(sys.executable, ['python'] + sys.argv)

    except Exception as e:
        logger.error(f"Error during restart cleanup: {e}")
        os._exit(1)  # Hard exit if cleanup fails

class ConfigFileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("config.ini"):
            logger.warning("config.ini changed. Restarting script...")
            restart_program()

# YOLO Models
class YOLOModels:
    def __init__(self, plate_model_path, char_model_path, arvand_model_path):
        logger.info("Loading YOLO models...")
        self.model_plate = torch.hub.load(
            'yolov5', 'custom', plate_model_path, source='local', device=device, force_reload=True)
        self.model_char = torch.hub.load(
            'yolov5', 'custom', char_model_path, source='local', force_reload=True)
        self.carmodel = YOLO('model/yolo11n.pt', verbose=False).to(device)
        logger.info("Models loaded successfully")

models = YOLOModels(params.modelPlate_path,
                    params.modelCharX_path, params.modelArvand_path)
logger.info('Server initialization complete')

# Memory management function
def clear_memory():
    logger.warning("Clearing memory to prevent leaks...")
    torch.cuda.empty_cache()
    gc.collect()

# Initialize cameras - call once at startup
def initialize_cameras():
    for i, source in enumerate(params.rtps):
        path_key = f"/rt{i+1}"
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open camera source: {source}")
                continue
                
            camera_feeds[path_key] = FreshestFrame(cap)
            logger.info(f"Camera initialized for {path_key}: {source}")
        except Exception as e:
            logger.error(f"Error initializing camera {source}: {str(e)}")

# Correcting angles
def correct_perspective(image, scale_factor):
    try:
        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(gray, 30, 150)

        # Line detection
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi/180,
                                threshold=30,
                                minLineLength=20,
                                maxLineGap=5)

        if lines is None:
            return image, (0, 0, 0, 0)

        # Calculate angle
        angles = []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            dx = x2_l - x1_l
            dy = y2_l - y1_l
            angle = np.degrees(np.arctan2(dy, dx))
            if -45 <= angle <= 45 or 135 <= abs(angle) <= 180:
                angles.append(angle)

        if not angles:
            return image, (0, 0, 0, 0)

        median_angle = np.median(angles)

        # Vertical angle correction
        if abs(median_angle) > 45:
            median_angle = 90 - median_angle

        if abs(median_angle) < 2:
            return image, (0, 0, 0, 0)

        # Rotate image
        (h, w) = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        # Calculate new size
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w - w)/2
        M[1, 2] += (new_h - h)/2

        deskewed = cv2.warpAffine(image, M, (new_w, new_h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Transform coordinates considering scale
        original_points = np.array([
            [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
        ], dtype=np.float32)

        transformed_points = cv2.transform(
            original_points.reshape(1, -1, 2), M
        ).squeeze().astype(float)

        # Apply scaling
        deskewed = cv2.resize(deskewed, None,
                              fx=scale_factor,
                              fy=scale_factor,
                              interpolation=cv2.INTER_LANCZOS4)

        # Scale coordinates
        transformed_points *= scale_factor

        new_x1 = int(transformed_points[:, 0].min())
        new_y1 = int(transformed_points[:, 1].min())
        new_x2 = int(transformed_points[:, 0].max())
        new_y2 = int(transformed_points[:, 1].max())

        return deskewed, (new_x1, new_y1, new_x2, new_y2)

    except Exception as e:
        logger.error(f"Error in correct_perspective: {e}")
        return image, (0, 0, 0, 0)

# Character Detection
def detect_plate_chars(cropped_plate):
    chars, confidences, char_detected = [], [], []
    results = models.model_char(cropped_plate)
    # Sort by x-coordinate
    detections = sorted(results.pred[0], key=lambda x: x[0])
    for det in detections:
        conf = det[4]

        if conf > 0.5:
            cls = int(det[5].item())
            char = params.char_id_dict.get(str(cls), '')
            chars.append(char)
            confidences.append(conf.item())
            char_detected.append(det.tolist())
    char_conf_avg = round(statistics.mean(confidences)
                          * 100) if confidences else 0
    return ''.join(chars), char_conf_avg

# Process a single frame with vehicle and plate detection
def process_frame(frame, path):
    try:
        # Create a lower-resolution copy for detection
        lowres_for_detection = cv2.resize(
            frame,
            (640, int(frame.shape[0] * 640 / frame.shape[1])),
            interpolation=cv2.INTER_AREA
        )
        scale_x = frame.shape[1] / lowres_for_detection.shape[1]
        scale_y = frame.shape[0] / lowres_for_detection.shape[0]

        # Detect vehicles in low-res
        car_res = models.carmodel(
            lowres_for_detection, device=device, classes=[2, 5, 7])
        
        if len(car_res[0]) > 0:
            for box in car_res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                # Scale back to original resolution
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cropped_car = frame[y1:y2, x1:x2]

                plate_results = models.model_plate(
                    cropped_car).pandas().xyxy[0]
                
                if not plate_results.empty:
                    for _, plate in plate_results.iterrows():
                        plate_conf = int(plate['confidence'] * 100)
                        if plate_conf >= int(params.plateConf):
                            x_min, y_min, x_max, y_max = (
                                int(plate['xmin']), int(plate['ymin']),
                                int(plate['xmax']), int(plate['ymax'])
                            )
                            
                            # Safety check to prevent out-of-bounds issues
                            if (y_min >= y_max or x_min >= x_max or 
                                y_min < 0 or x_min < 0 or 
                                y_max > cropped_car.shape[0] or 
                                x_max > cropped_car.shape[1]):
                                continue
                                
                            cropped_plate = cropped_car[y_min:y_max, x_min:x_max]
                            
                            # Skip if the cropped plate is empty
                            if cropped_plate.size == 0:
                                continue
                                
                            plate_text, char_conf_avg = detect_plate_chars(cropped_plate)

                            cv2.rectangle(
                                cropped_car, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                            plate_text = plate_text.replace('Taxi', 'x')
                            confidence = float(params.charConf) * 100

                            if char_conf_avg >= confidence and len(plate_text) >= 8:
                                cv2.putText(cropped_car, f"Plate: {plate_text}", (x_min, y_min - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
                                db_entries_time(
                                    number=plate_text,
                                    charConfAvg=char_conf_avg,
                                    plateConfAvg=plate_conf,
                                    croppedPlate=cropped_plate,
                                    status="Active",
                                    frame=frame,
                                    isarvand='notarvand',
                                    rtpath=path
                                )
                                break
                            else:
                                deskewed_plate, (newx1, newy1, newx2, newy2) = correct_perspective(
                                    cropped_plate, 1.0)
                                if deskewed_plate.size == 0:
                                    continue
                                    
                                newx1 = max(0, newx1)
                                newy1 = max(0, newy1)
                                newx2 = min(deskewed_plate.shape[1], newx2)
                                newy2 = min(deskewed_plate.shape[0], newy2)
                                
                                if (newx2 <= newx1) or (newy2 <= newy1):
                                    newx1, newy1 = 0, 0
                                    newx2, newy2 = deskewed_plate.shape[1], deskewed_plate.shape[0]
                                    
                                d = newy2 - newy1
                                tempyMax = newy1 + int(d / 2)
                                
                                # Safety check before cropping
                                if (tempyMax > newy1 and newx2 > newx1 and 
                                    newy1 >= 0 and newx1 >= 0 and 
                                    tempyMax <= deskewed_plate.shape[0] and 
                                    newx2 <= deskewed_plate.shape[1]):
                                    
                                    cropped_plate_nesf = deskewed_plate[newy1:tempyMax, newx1:newx2]
                                    
                                    if cropped_plate_nesf.size > 0:
                                        plate_text_arvnad, char_conf_arvnad = detect_plate_chars(
                                            cropped_plate_nesf)

                                        if len(plate_text_arvnad) >= 5 and char_conf_arvnad >= confidence - 3:
                                            cv2.putText(cropped_car, f"Plate: {plate_text_arvnad}", (x_min, y_min - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
                                            db_entries_time(
                                                number=plate_text_arvnad,
                                                charConfAvg=char_conf_arvnad,
                                                plateConfAvg=plate_conf,
                                                croppedPlate=cropped_plate,
                                                status="Active",
                                                frame=frame,
                                                isarvand='arvand',
                                                rtpath=path
                                            )
        
        return frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame

# Heartbeat implementation
async def send_ping(websocket):
    try:
        while True:
            await asyncio.sleep(30)  # Send ping every 30 seconds
            try:
                await websocket.ping()
            except:
                break
    except asyncio.CancelledError:
        pass

# WebSocket Connection Handler
async def ws_handler(websocket):
    path = websocket.request.path
    client_id = id(websocket)
    
    # Add connection to active connections
    active_connections[client_id] = {
        'websocket': websocket,
        'last_frame_time': 0
    }
    
    logger.info(f"Client {client_id} connected on path {path}")
    
    # Setup ping/pong for connection health monitoring
    ping_task = asyncio.create_task(send_ping(websocket))
    
    try:
        # Check if this is a request for a specific camera feed
        if path.startswith('/rt'):
            # Handle single camera request
            await handle_single_camera(websocket, path, client_id)
        elif path == '/all':
            # Handle request for all cameras
            await handle_all_cameras(websocket, client_id)
        else:
            logger.warning(f"Invalid path: {path}")
            await websocket.close(1008, "Invalid camera path")
    except asyncio.CancelledError:
        logger.info(f"Connection to client {client_id} cancelled")
    except Exception as e:
        logger.error(f"Error in connection handler: {str(e)}")
    finally:
        # Clean up
        ping_task.cancel()
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

# Handle single camera feed
async def handle_single_camera(websocket, path, client_id):
    if path not in camera_feeds:
        logger.warning(f"Camera path not found: {path}")
        await websocket.close(1008, "Camera not found")
        return
    
    camera = camera_feeds[path]
    
    try:
        while client_id in active_connections:
            # Rate limiting
            current_time = time.time()
            if current_time - active_connections[client_id]['last_frame_time'] < FRAME_DELAY:
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
                continue
                
            active_connections[client_id]['last_frame_time'] = current_time
            
            # Read frame
            ret, frame = camera.read()
            if not ret:
                logger.warning(f"Failed to read frame from {path}")
                await asyncio.sleep(0.5)  # Wait before trying again
                continue
                
            # Process frame (detect vehicles and plates)
            processed_frame = process_frame(frame, path)
            
            # Encode and send frame
            try:
                _, encoded = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                data = base64.b64encode(encoded).decode('utf-8')
                await websocket.send(data)
                
                # Explicitly manage memory
                if current_time % 10 < 0.1:  # Every ~10 seconds
                    clear_memory()
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection closed for client {client_id}")
                break
            except Exception as e:
                logger.error(f"Error sending frame: {str(e)}")
                await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in handle_single_camera: {str(e)}")

# Handle all camera feeds
async def handle_all_cameras(websocket, client_id):
    try:
        while client_id in active_connections:
            # Rate limiting
            current_time = time.time()
            if current_time - active_connections[client_id]['last_frame_time'] < FRAME_DELAY:
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
                continue
                
            active_connections[client_id]['last_frame_time'] = current_time
            
            # Prepare data for all cameras
            frames_data = {}
            
            for path, camera in camera_feeds.items():
                ret, frame = camera.read()
                if not ret:
                    logger.warning(f"Failed to read frame from {path}")
                    continue
                    
                # Process frame (detect vehicles and plates)
                processed_frame = process_frame(frame, path)
                
                # Encode frame
                _, encoded = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                encoded_data = base64.b64encode(encoded).decode('utf-8')
                
                # Add to frames data
                frames_data[path] = encoded_data
            
            # Send all frames as JSON
            if frames_data:
                try:
                    await websocket.send(json.dumps(frames_data))
                    
                    # Explicitly manage memory
                    if current_time % 10 < 0.1:  # Every ~10 seconds
                        clear_memory()
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Connection closed for client {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Error sending frames: {str(e)}")
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.1)  # No frames available, wait briefly
                
    except Exception as e:
        logger.error(f"Error in handle_all_cameras: {str(e)}")

# Global variables for cleanup and shutdown management
server = None
observer = None
shutdown_event = threading.Event()

# Graceful shutdown function
def graceful_shutdown():
    logger.info("Initiating graceful shutdown...")
    shutdown_event.set()
    
    # Stop cameras
    for cam_key, cam in camera_feeds.items():
        try:
            if hasattr(cam, 'stop') and callable(cam.stop):
                cam.stop()
            logger.info(f"Stopped camera feed {cam_key}")
        except Exception as e:
            logger.error(f"Error stopping camera {cam_key}: {e}")
    
    # Clean up resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Stop observer if running
    global observer
    if observer and observer.is_alive():
        observer.stop()
        observer.join(timeout=1.0)
        logger.info("Config file observer stopped")
    
    logger.info("Cleanup complete. Shutting down.")

# Signal handler for SIGINT (Ctrl+C) and SIGTERM
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    graceful_shutdown()
    sys.exit(0)

# WebSocket Server with shutdown support
async def websocket_server():
    global server
    logger.info(f"Starting WebSocket server at ws://{host}:{port}")
    print(f'WebSocket server started at ws://{host}:{port}')
    print(f'Camera feeds available at:')
    for path_key in camera_feeds.keys():
        print(f'  - ws://{host}:{port}{path_key}')
    print(f'All cameras are available at: ws://{host}:{port}/all')

    server = await websockets.serve(
        ws_handler,
        host,
        port,
    )
    
    # Setup shutdown detection
    while not shutdown_event.is_set():
        await asyncio.sleep(0.1)
    
    # Close server when shutdown is requested
    if server:
        server.close()
        await server.wait_closed()
        logger.info("WebSocket server closed")

# Main
if __name__ == "__main__":
    # Register signal handlers
    import signal
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Termination signal
    
    # Start config watcher
    observer = Observer()
    event_handler = ConfigFileChangeHandler()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    
    # Initialize all cameras
    initialize_cameras()
    
    # Run WebSocket server
    try:
        asyncio.run(websocket_server())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        restart_program()