import asyncio
import gc
import logging
import os
import time
from fastapi import Request
import numpy as np
import cv2
import warnings
import psutil
import torch
import statistics
from ultralytics import YOLO
from configParams import Parameters
from database.db_entries_utils import db_entries_time
from watchdog.observers import Observer
from camera import FreshestFrame
import threading

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
logger.info(f"Version : 10.1.1 Up 05/19/2025")


# Frame rate limiter (FPS)
TARGET_FPS = 30  # Adjust based on your needs
FRAME_DELAY = 1.0 / TARGET_FPS
# Constants for health monitoring
RETRY_LIMIT = 5
RETRY_DELAY = 3  # seconds

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        threading.Thread.__init__(
            self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        threading.Thread.join(self)
        return self._return


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
observer = Observer()
observer.start()

# Memory management function


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Initialize cameras - call once at startup


def graceful_shutdown():

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


def process_frame(frame, path):

    try:
        # Create a lower-resolution copy for detection
        # lowres_for_detection=frame
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

                            cropped_plate = cropped_car[y_min:y_max,
                                                        x_min:x_max]

                            # Skip if the cropped plate is empty
                            if cropped_plate.size == 0:
                                continue

                            plate_text, char_conf_avg = detect_plate_chars(
                                cropped_plate)

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
        try:
            del plate_results, cropped_car, car_res
        except Exception as e:
            pass
        return frame

    except Exception as e:
        return frame


def kill_processes_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.net_connections()
            for conn in connections:
                if conn.laddr.port == port:
                    print(
                        f"Killing PID {proc.pid} ({proc.name()}) on port {port}")
                    proc.kill()
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


async def generate_frames(camera_idx, source, request: Request):
    """Generate frames from a specific camera feed"""

    def open_capture(source):
        cap = cv2.VideoCapture(source)
        return cap if cap.isOpened() else None

    retries = 0
    cap = open_capture(source)

    while cap is None and retries < RETRY_LIMIT:
        print(f"[Camera {camera_idx}] Failed to open source. Retrying ({retries + 1}/{RETRY_LIMIT})...")
        await asyncio.sleep(RETRY_DELAY)
        retries += 1
        cap = open_capture(source)

    if cap is None:
        print(f"[Camera {camera_idx}] Could not open source after {RETRY_LIMIT} retries.")
        return
    fresh = FreshestFrame(cap)

    try:
        while fresh.is_alive():

            if await request.is_disconnected():
                print("Client disconnected, releasing camera.")
                break
            success, frame = fresh.read()
            if not success:

                # Generate blank frame if we can't read from camera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No signal", (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                frame = process_frame(frame, f'/rt{camera_idx}')

            # Encode and yield the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(e)
        graceful_shutdown()
        
    finally:
        fresh.release()
        cap.release()
        print("Camera released.")
