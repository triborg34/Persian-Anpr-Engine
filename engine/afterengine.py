import gc
import logging
import os
import sys
import time
import numpy as np
import cv2
import warnings
import psutil
import torch
import asyncio
import base64
import threading
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
logger.info(f"Version : 10.0.5 Up 05/10/2025")

# A dictionary to store FreshestFrame objects for each RTSP source
camera_feeds = {}

# A dictionary to store active client connections
active_connections = {}

# Camera health monitoring
camera_health = {}

# Frame rate limiter (FPS)
TARGET_FPS = 15  # Adjust based on your needs
FRAME_DELAY = 1.0 / TARGET_FPS

# Constants for health monitoring
MAX_FAILED_FRAMES = 10  # Number of consecutive failed frame reads before restart
MAX_CLIENT_WAIT_TIME = 5.0  # Maximum time (seconds) a client should wait for a frame
CAMERA_HEALTH_CHECK_INTERVAL = 30  # Check camera health every 30 seconds




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

def restart_camera(camera_path):
    """Attempt to restart a specific camera feed"""
    try:
        logger.warning(f"Attempting to restart camera {camera_path}...")
        
        # Get camera source index
        cam_index = int(camera_path.replace('/rt', '')) - 1
        if cam_index < 0 or cam_index >= len(params.rtps):
            logger.error(f"Invalid camera index for {camera_path}")
            return False
            
        source = params.rtps[cam_index]
        
        # Stop existing camera if it exists
        if camera_path in camera_feeds:
            try:
                cam = camera_feeds[camera_path]
                if hasattr(cam, 'stop') and callable(cam.stop):
                    cam.stop()
                logger.info(f"Stopped existing camera feed {camera_path}")
            except Exception as e:
                logger.error(f"Error stopping camera {camera_path}: {e}")
        
        # Reinitialize camera
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to reopen camera source: {source}")
                return False
                
            camera_feeds[camera_path] = FreshestFrame(cap)
            camera_health[camera_path] = {
                'failed_frames': 0,
                'last_successful_read': time.time(),
                'restart_attempts': camera_health.get(camera_path, {}).get('restart_attempts', 0) + 1
            }
            logger.info(f"Camera reinitialized for {camera_path}: {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error reinitializing camera {source}: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in restart_camera: {str(e)}")
        return False

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
            camera_health[path_key] = {
                'failed_frames': 0, 
                'last_successful_read': time.time(),
                'restart_attempts': 0
            }
            logger.info(f"Camera initialized for {path_key}: {source}")
        except Exception as e:
            logger.error(f"Error initializing camera {source}: {str(e)}")
            camera_health[path_key] = {
                'failed_frames': MAX_FAILED_FRAMES,  # Mark as failed immediately
                'last_successful_read': 0,
                'restart_attempts': 0
            }

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
        'last_frame_time': 0,
        'last_frame_sent': time.time(),  # Track when the last frame was successfully sent
        'frames_sent': 0,  # Count frames sent to this client
        'connection_time': time.time()  # When the client connected
    }
    
    # Log connection without exposing the actual client_id
    logger.info(f"New client connected on path {path}")
    
    # Setup ping/pong for connection health monitoring
    ping_task = asyncio.create_task(send_ping(websocket))
    
    try:
        # Check if this is a request for a specific camera feed
        if path.startswith('/rt'):
            # Handle single camera request
            await handle_single_camera(websocket, path, client_id)
        else:
            logger.warning(f"Invalid path: {path}")
            await websocket.close(1008, "Invalid camera path")
    except asyncio.CancelledError:
        logger.info(f"Connection cancelled on path {path}")
    except Exception as e:
        
        logger.error(f"Error in connection handler: {str(e)}")
        restart_program()
        
        
    finally:
        # Clean up
        ping_task.cancel()
        if client_id in active_connections:
            # Calculate connection duration for logging
            duration = time.time() - active_connections[client_id].get('connection_time', time.time())
            del active_connections[client_id]
            logger.info(f"Client disconnected from {path} after {duration:.1f}s")

# Handle single camera feed
async def handle_single_camera(websocket, path, client_id):
    if path not in camera_feeds:
        logger.warning(f"Camera path not found: {path}")
        await websocket.close(1008, "Camera not found")
        return
    
    camera = camera_feeds[path]
    consecutive_failures = 0
    
    try:
        while client_id in active_connections:
            # Rate limiting
            current_time = time.time()
            if current_time - active_connections[client_id]['last_frame_time'] < FRAME_DELAY:
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
                continue
                
            active_connections[client_id]['last_frame_time'] = current_time
            
            # Check if this camera needs reset based on recent failed frames
            if (path in camera_health and 
                camera_health[path]['failed_frames'] >= MAX_FAILED_FRAMES):
                # Too many consecutive failures for this camera
                logger.warning(f"Camera {path} has failed too many times, attempting restart")
                if restart_camera(path):
                    # Camera was restarted successfully, update references
                    camera = camera_feeds[path]
                    consecutive_failures = 0
                else:
                    # Camera restart failed
                    logger.error(f"Failed to restart camera {path}")
                    if camera_health[path]['restart_attempts'] >= 3:
                        logger.critical(f"Multiple camera restart attempts failed for {path}. Restarting program...")
                        restart_program()
                    
                    await asyncio.sleep(5)  # Wait before trying again
                    continue
            
            # Read frame
            try:
                ret, frame = camera.read(timeout=2)
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    camera_health[path]['failed_frames'] += 1
                    
                    logger.warning(f"Failed to read frame from {path} (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= 5:
                        logger.error(f"Multiple consecutive frame read failures for {path}")
                        
                        # Check if this client has been waiting too long
                        client_wait_time = current_time - active_connections[client_id]['last_frame_sent'] 
                        if client_wait_time > MAX_CLIENT_WAIT_TIME:
                            logger.warning(f"Client {client_id} has been waiting for frames for {client_wait_time:.1f}s, attempting camera restart")
                            restart_camera(path)
                            camera = camera_feeds[path]
                            
                    await asyncio.sleep(0.5)  # Wait before trying again
                    continue
                
                # Reset failure counters on successful frame read
                consecutive_failures = 0
                camera_health[path]['failed_frames'] = 0
                camera_health[path]['last_successful_read'] = current_time
                
                # Process frame (detect vehicles and plates)
                processed_frame = process_frame(frame, path)
                
                # Encode and send frame
                try:
                    _, encoded = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    data = base64.b64encode(encoded).decode('utf-8')
                    await websocket.send(data)
                    
                    # Update client stats
                    active_connections[client_id]['last_frame_sent'] = current_time
                    active_connections[client_id]['frames_sent'] += 1
                    
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
                logger.error(f"Error reading frame: {str(e)}")
                consecutive_failures += 1
                camera_health[path]['failed_frames'] += 1
                await asyncio.sleep(0.5)
                
    except Exception as e:
        logger.error(f"Error in handle_single_camera: {str(e)}")

# Check for cameras that haven't sent frames in a while
async def monitor_camera_health():
    """Periodically check the health of all cameras and clients"""
    while not shutdown_event.is_set():
        try:
            current_time = time.time()
            
            # Check each camera
            for cam_path, health in camera_health.items():
                time_since_last_frame = current_time - health['last_successful_read']
                
                # If camera hasn't delivered a good frame in too long
                if time_since_last_frame > 60:  # No good frames for 60 seconds
                    logger.warning(f"Camera {cam_path} hasn't delivered frames for {time_since_last_frame:.1f}s")
                    
                    # Check if any clients are connected to this camera
                    has_clients = any(client['websocket'].request.path == cam_path 
                                     for client in active_connections.values())
                    
                    if has_clients:
                        logger.warning(f"Clients are waiting for camera {cam_path}, attempting restart")
                        restart_camera(cam_path)
            
            # Check for clients that haven't received frames
            for client_id, client_data in list(active_connections.items()):
                time_since_last_frame = current_time - client_data['last_frame_sent']
                
                # If client hasn't received frames in too long (and has been connected for more than a few seconds)
                if time_since_last_frame > MAX_CLIENT_WAIT_TIME * 2 and client_data['frames_sent'] > 0:
                    logger.warning(f"Client {client_id} hasn't received frames for {time_since_last_frame:.1f}s")
                    cam_path = client_data['websocket'].request.path
                    
                    if cam_path in camera_feeds:
                        logger.warning(f"Restarting camera {cam_path} for client {client_id}")
                        restart_camera(cam_path)
            
            # Check system resources
            mem_info = psutil.virtual_memory()
            if mem_info.percent > 95:  # More than 95% memory usage
                logger.warning(f"System memory usage critical: {mem_info.percent}%")
                clear_memory()
                
            await asyncio.sleep(CAMERA_HEALTH_CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in camera health monitor: {str(e)}")
            await asyncio.sleep(5)  # Short delay before retrying

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
    os._exit(0)  # Use os._exit instead of sys.exit for more forceful termination

# Enhanced signal handler for SIGINT (Ctrl+C) and SIGTERM
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, shutting down...")
    graceful_shutdown()

# WebSocket Server with shutdown support
async def websocket_server():
    global server
    logger.info(f"Starting WebSocket server at ws://{host}:{port}")
    print(f'WebSocket server started at ws://{host}:{port}')

    server = await websockets.serve(
        ws_handler,
        host,
        port,
    )
    
    # Start camera health monitoring task
    camera_monitor_task = asyncio.create_task(monitor_camera_health())
    
    # Setup shutdown detection
    while not shutdown_event.is_set():
        await asyncio.sleep(0.1)
    
    # Cancel monitoring task when shutdown is requested
    camera_monitor_task.cancel()
    
    # Close server when shutdown is requested
    if server:
        server.close()
        await server.wait_closed()
        logger.info("WebSocket server closed")

# Auto-restart watchdog
def watchdog_monitor():
    """Monitor system for conditions that require restart"""
    while not shutdown_event.is_set():
        try:
            # Check RAM usage
            mem = psutil.virtual_memory()
            if mem.percent > 95:  # Over 95% memory usage
                logger.warning(f"Memory usage critical: {mem.percent}%. Triggering restart.")
                restart_program()
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 95:  # Over 95% CPU usage
                logger.warning(f"CPU usage critical: {cpu_usage}%. Triggering restart.")
                restart_program()
            
            # Check if any cameras need reset
            camera_failures = sum(1 for health in camera_health.values() 
                                 if health['failed_frames'] >= MAX_FAILED_FRAMES)
            
            if camera_failures >= len(camera_health) * 0.5 and len(camera_health) > 0:
                # If 50% or more cameras have failed
                logger.warning(f"{camera_failures} cameras have failed. Triggering restart.")
                restart_program()
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in watchdog monitor: {str(e)}")
            time.sleep(5)

# Main
if __name__ == "__main__":
    # Register signal handlers with more forceful approach
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
    
    # Start watchdog monitor in a separate thread
    watchdog_thread = threading.Thread(target=watchdog_monitor, daemon=True)
    watchdog_thread.start()
    
    # Record start time for uptime tracking
    start_time = time.time()
    
    # Run WebSocket server
    try:
        # Create a Future that will be set if Ctrl+C is pressed
        loop = asyncio.get_event_loop()
        main_task = loop.create_task(websocket_server())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        restart_program()