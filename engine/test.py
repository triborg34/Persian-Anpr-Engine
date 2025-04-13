from collections import deque
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
from queue import Queue
import statistics
from ultralytics import YOLO
import websockets.http
import websockets.uri
from configParams import Parameters
from database.db_entries_utils import db_entries_time
import websockets
from deep_sort_realtime.deepsort_tracker import DeepSort

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
# host = params.defip.split('//')[1].split(':')[0]



# Device setup
device = torch.device(0 if torch.cuda.is_available()  else "cpu")
logger.info(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} device.")
logger.info(f"Version : 10.0.0 Up 26/3/2025")

# Frame Buffers: One buffer for each RTSP source
frame_buffers = {f"/rt{i+1}": Queue(maxsize=10) for i, _ in enumerate(params.rtps)}
global buffer_key 

# YOLO Models
class YOLOModels:
    def __init__(self, plate_model_path, char_model_path, arvand_model_path):
        logger.info("Loading YOLO models...")
        self.model_plate = torch.hub.load('yolov5', 'custom', plate_model_path, source='local', device=device, force_reload=True)
        self.model_char = torch.hub.load('yolov5', 'custom', char_model_path, source='local', force_reload=True)
       
        self.carmodel=YOLO('model/yolo11n.pt',verbose=False).to(device)

models = YOLOModels(params.modelPlate_path, params.modelCharX_path, params.modelArvand_path)
print('start server')
deep_sort_tracker = DeepSort(max_age=30)
object_area_history = {}
track_histories = {}  # track_id -> deque of last states
MAX_HISTORY = 5  # Number of frames to observe
ENTRANCE_THRESHOLD = 1.2
EXIT_THRESHOLD = 0.8
POSITION_SHIFT_Y = 15  # Pixels


# Memory management function
def clear_memory():
    logger.warning("Clearing memory to prevent leaks and restarting...")
    torch.cuda.empty_cache()
    gc.collect()
    os.execv(sys.executable, ['python'] + sys.argv)
# Frame Producer
def frame_producer(source, buffer):
    while True:
        try:
            logger.info(f"Connecting to RTSP: {source}")

            cap = cv2.VideoCapture(source)
            # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not cap.isOpened():
                logger.warning(f"Failed to open {source}. Retrying in 5s...")
                time.sleep(5)
                continue

            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.warning(f"Lost connection or empty frame from {source}. Reconnecting...")
                    break

                if buffer.full():
                    buffer.get()
                buffer.put(frame)
        except Exception as e:
            logger.error(f"[frame_producer] Error for {source}: {e}")
            time.sleep(5)
        
        # Check memory usage and clear if necessary
        if torch.cuda.memory_reserved(device) > 2e9:  # Adjust memory threshold as needed
            clear_memory()


#correcting angles
def correct_perspective(image, scale_factor):
    try:

        # پیشپردازش
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # تشخیص لبه
        edges = cv2.Canny(gray, 30, 150)
        
        # تشخیص خطوط
        lines = cv2.HoughLinesP(edges, 
                               rho=1, 
                               theta=np.pi/180, 
                               threshold=30, 
                               minLineLength=20, 
                               maxLineGap=5)
        
        if lines is None:
            return image, (0,0,0,0)

        # محاسبه زاویه
        angles = []
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            dx = x2_l - x1_l
            dy = y2_l - y1_l
            angle = np.degrees(np.arctan2(dy, dx))
            if -45 <= angle <= 45 or 135 <= abs(angle) <= 180:
                angles.append(angle)
        
        if not angles:
            return image, (0,0,0,0)
        
        median_angle = np.median(angles)
        
        # اصلاح زاویه عمودی
        if abs(median_angle) > 45:
            median_angle = 90 - median_angle
        
        if abs(median_angle) < 2:
            return image, (0,0,0,0)
            
        # چرخش تصویر
        (h, w) = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # محاسبه اندازه جدید
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0,2] += (new_w - w)/2
        M[1,2] += (new_h - h)/2
        
        deskewed = cv2.warpAffine(image, M, (new_w, new_h), 
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        
        # تبدیل مختصات با در نظر گرفتن مقیاس
        original_points = np.array([
            [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
        ], dtype=np.float32)
        
        transformed_points = cv2.transform(
            original_points.reshape(1, -1, 2), M
        ).squeeze().astype(float)
        
        # اعمال بزرگنمایی
        deskewed = cv2.resize(deskewed, None, 
                            fx=scale_factor, 
                            fy=scale_factor, 
                            interpolation=cv2.INTER_LANCZOS4)
        
        # مقیاس‌گذاری مختصات
        transformed_points *= scale_factor
        
        new_x1 = int(transformed_points[:,0].min())
        new_y1 = int(transformed_points[:,1].min())
        new_x2 = int(transformed_points[:,0].max())
        new_y2 = int(transformed_points[:,1].max())
        
        return deskewed, (new_x1, new_y1, new_x2, new_y2)
    
    except Exception as e:
        print(f"Error: {e}")
        return image,(0,0,0,0)

# Character Detection
def detect_plate_chars(cropped_plate):
    chars, confidences, char_detected = [], [], []
    results = models.model_char(cropped_plate)
    detections = sorted(results.pred[0], key=lambda x: x[0])  # Sort by x-coordinate
    for det in detections:
        conf = det[4]
        
        
        if conf > 0.5:
            cls = int(det[5].item())
            char = params.char_id_dict.get(str(cls), '')
            chars.append(char)
            confidences.append(conf.item())
            char_detected.append(det.tolist())
    char_conf_avg = round(statistics.mean(confidences) * 100) if confidences else 0
    return ''.join(chars), char_conf_avg

# WebSocket Frame Transmitter
async def transmit_frames(websocket, path):
    print(path)
    logger.info(f"Client connected to {path}")
    
    if path not in frame_buffers.keys():
        logger.warning(f"Invalid path: {path}")
        await websocket.close()
        return

    buffer = frame_buffers[path]
    try:
        while True:
            if not buffer.empty():
                frame = buffer.get()
                if frame is None or frame.size == 0:
                    logger.warning(f"[{path}] Skipped empty frame")
                    continue

                # Create a lower-resolution copy for detection
                lowres_for_detection = cv2.resize(
                    frame,
                    (640, int(frame.shape[0] * 640 / frame.shape[1])),
                    interpolation=cv2.INTER_AREA
                )
                scale_x = frame.shape[1] / lowres_for_detection.shape[1]
                scale_y = frame.shape[0] / lowres_for_detection.shape[0]

                # Detect vehicles in low-res
                car_res = models.carmodel(lowres_for_detection, device=device, classes=[2, 5, 7])
                detections = []
                if len(car_res[0]) > 0:
                    for box in car_res[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                        # Scale back to original resolution
                        conf = float(box.conf[0])
                        clse = int(box.cls[0])
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cropped_car = frame[y1:y2, x1:x2]
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, clse))
                    
                    tracks = deep_sort_tracker.update_tracks(detections, frame=frame)
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        x1, y1, x2, y2 = map(int, track.to_ltrb())
                        current_area = (x2 - x1) * (y2 - y1)
                        current_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # Initialize track history if new
                        if track_id not in track_histories:
                            track_histories[track_id] = deque(maxlen=MAX_HISTORY)

                        # Append current state
                        track_histories[track_id].append({'area': current_area, 'center': current_center})

                        # Analyze movement if enough history exists
                        if len(track_histories[track_id]) >= MAX_HISTORY:
                            areas = [h['area'] for h in track_histories[track_id]]
                            centers_y = [h['center'][1] for h in track_histories[track_id]]

                            avg_area_old = sum(areas[:-1]) / (MAX_HISTORY - 1)
                            latest_area = areas[-1]

                            area_ratio = latest_area / avg_area_old if avg_area_old != 0 else 1
                            delta_y = centers_y[-1] - centers_y[0]

                            status = "Stationary"

                            if area_ratio > ENTRANCE_THRESHOLD and delta_y > POSITION_SHIFT_Y:
                                status = "Entrance"
                                print(f"[Track {track_id}] Status: {status}, Area ratio: {area_ratio:.2f}, ΔY: {delta_y}")
                            elif area_ratio < EXIT_THRESHOLD and delta_y < -POSITION_SHIFT_Y:
                                status = "Exit"
                                print(f"[Track {track_id}] Status: {status}, Area ratio: {area_ratio:.2f}, ΔY: {delta_y}")

                           
                        
                        
                        plate_results = models.model_plate(cropped_car).pandas().xyxy[0]
                        if not plate_results.empty:
                            for _, plate in plate_results.iterrows():
                                plate_conf = int(plate['confidence'] * 100)
                                if plate_conf >= int(params.plateConf):
                                    x_min, y_min, x_max, y_max = (
                                        int(plate['xmin']), int(plate['ymin']),
                                        int(plate['xmax']), int(plate['ymax'])
                                    )
                                    cropped_plate = cropped_car[y_min:y_max, x_min:x_max]
                                    plate_text, char_conf_avg = detect_plate_chars(cropped_plate)

                                    cv2.rectangle(cropped_car, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                                    plate_text = plate_text.replace('Taxi', 'x')
                                    confidance = float(params.charConf) * 100

                                    if char_conf_avg >= confidance and len(plate_text) >= 8:
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
                                        deskewed_plate, (newx1, newy1, newx2, newy2) = correct_perspective(cropped_plate, 1.0)
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
                                        cropped_plate_nesf = deskewed_plate[newy1:tempyMax, newx1:newx2]
                                        plate_text_arvnad, char_conf_arvnad = detect_plate_chars(cropped_plate_nesf)

                                        if len(plate_text_arvnad) >= 5 and char_conf_arvnad >= confidance - 3:
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
                                

                # Encode full-res frame with annotations
                try:
                    _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    data = base64.b64encode(encoded).decode('utf-8')
                    await websocket.send(data)
                    del frame, data
                   
                    torch.cuda.empty_cache()
                except cv2.error as e:
                    logger.error(f"OpenCV Memory Error: {e}")
                    clear_memory()
            else:
                await asyncio.sleep(0.03)
    except websockets.ConnectionClosed:
        logger.info(f"Client disconnected from {path}")
    # except Exception as e:
    #     logger.error(f"Unexpected error: {e}")
        


# WebSocket Server


async def ws_handler(websocket):
    """Handle WebSocket connections."""
    path=websocket.request.path


    await transmit_frames(websocket, path)


async def websocket_server():
    """Start the WebSocket server."""
    logger.info(f"Starting WebSocket server at ws://{host}:{port}")
    print(f'start websocket server at ws://{host}:{port}')
    
    server = await websockets.serve(
        ws_handler,
        host,
        port,
    )
    
    await asyncio.Future()  # run forever
# Main
if __name__ == "__main__":
    # Start frame producer threads for each RTSP source
    for i, source in enumerate(params.rtps):
        buffer_key = f"/rt{i+1}"
        
        threading.Thread(target=frame_producer, args=(source, frame_buffers[buffer_key]), daemon=True).start()

    # Run WebSocket server
    asyncio.run(websocket_server())
