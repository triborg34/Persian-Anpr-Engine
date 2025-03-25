import gc
import logging
import os
import sys
import time
import numpy as np
import cv2
import warnings
import asyncio
import base64
import threading
from queue import Queue
import statistics
import onnxruntime as ort
import websockets
from configParams import Parameters
from database.db_entries_utils import db_entries_time

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CCTV-Server")
warnings.filterwarnings("ignore", category=UserWarning)

# Parameters
params = Parameters()
port = int(params.socketport)
host = '0.0.0.0'

# Frame Buffers
frame_buffers = {f"/rt{i+1}": Queue(maxsize=10) for i, _ in enumerate(params.rtps)}

def clear_memory():
    logger.warning("Clearing memory to prevent leaks and restarting...")
    gc.collect()
    os.execv(sys.executable, ['python'] + sys.argv)

# Preprocessing and Postprocessing

def preprocess_frame(image, size=640):
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, image.shape[3], image.shape[2]

def scale_coords(box, input_w, input_h, original_w, original_h):
    x, y, w, h = box
    x = int(x * original_w / input_w)
    y = int(y * original_h / input_h)
    w = int(w * original_w / input_w)
    h = int(h * original_h / input_h)
    return [x, y, w, h]

def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.45):
    boxes, confidences, class_ids = [], [], []
    for pred in prediction:
        if pred[4] < conf_thresh:
            continue
        scores = pred[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_thresh:
            x_center, y_center, width, height = pred[:4]
            x = x_center - width / 2
            y = y_center - height / 2
            boxes.append([int(x), int(y), int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
    detections = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        box = boxes[i]
        detections.append({"box": box, "confidence": confidences[i], "class_id": class_ids[i]})
    return detections

# Plate Deskewing

def correct_perspective(image, scale_factor=1.0):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        gray = cv2.medianBlur(gray, 3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        edges = cv2.Canny(gray, 30, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=5)
        if lines is None:
            return image, (0, 0, 0, 0)
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            if -45 <= angle <= 45 or 135 <= abs(angle) <= 180:
                angles.append(angle)
        if not angles:
            return image, (0, 0, 0, 0)
        median_angle = np.median(angles)
        if abs(median_angle) > 45:
            median_angle = 90 - median_angle
        if abs(median_angle) < 2:
            return image, (0, 0, 0, 0)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        deskewed = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        transformed_points = cv2.transform(np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(1, -1, 2), M).squeeze()
        deskewed = cv2.resize(deskewed, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
        transformed_points *= scale_factor
        new_x1 = int(transformed_points[:, 0].min())
        new_y1 = int(transformed_points[:, 1].min())
        new_x2 = int(transformed_points[:, 0].max())
        new_y2 = int(transformed_points[:, 1].max())
        return deskewed, (new_x1, new_y1, new_x2, new_y2)
    except Exception as e:
        print(f"Deskew error: {e}")
        return image, (0, 0, 0, 0)

# Model Class
class YOLOModels:
    def __init__(self, plate_model_path, char_model_path, car_model_path):
        self.plate_session = ort.InferenceSession(plate_model_path,providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.plate_input = self.plate_session.get_inputs()[0].name

        self.char_session = ort.InferenceSession(char_model_path,providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.char_input = self.char_session.get_inputs()[0].name

        self.car_session = ort.InferenceSession(car_model_path,providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.car_input = self.car_session.get_inputs()[0].name
        self.car_outputs = [o.name for o in self.car_session.get_outputs()]

models = YOLOModels('model/plateYolo.onnx', 'model/CharsYolo.onnx', "model/yolo11n.onnx")

# Frame Producer

def frame_producer(source, buffer):
    while True:
        try:
            logger.info(f"Connecting to RTSP: {source}")
            cap = cv2.VideoCapture(source)
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

                gc.collect()

        except Exception as e:
            logger.error(f"[frame_producer] Error for {source}: {e}")
            time.sleep(5)

def preprocess_frame(image, size=640):
    if image is None or image.size == 0:
        raise ValueError("Attempted to preprocess an empty frame.")
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image, image.shape[3], image.shape[2]
# Character Detection

def detect_plate_chars(cropped_plate):
    input_tensor, _, _ = preprocess_frame(cropped_plate)
    result = models.char_session.run(None, {models.char_input: input_tensor})
    detections = result[0][0]  # assuming first output is detections
    detections = sorted(detections, key=lambda x: x[0])
    chars, confidences = [], []
    for det in detections:
        conf = det[4]
        if conf > 0.5:
            cls = int(det[5])
            char = params.char_id_dict.get(str(cls), '')
            chars.append(char)
            confidences.append(conf)
    avg_conf = round(statistics.mean(confidences) * 100) if confidences else 0
    return ''.join(chars), avg_conf

# WebSocket Frame Transmitter


async def transmit_frames(websocket, path):
    logger.info(f"Client connected to {path}")
    if path not in frame_buffers:
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
                try:
                    input_tensor, w, h = preprocess_frame(frame)
                except ValueError as ve:
                    logger.warning(f"[{path}] Invalid frame for preprocessing: {ve}")
                    continue

                output = models.car_session.run(models.car_outputs, {models.car_input: input_tensor})
                preds = output[0][0]
                detections = non_max_suppression(preds, 0.4)
                for det in detections:
                    x, y, w, h = scale_coords(det['box'], 640, 640, frame.shape[1], frame.shape[0])
                    x2, y2 = x + w, y + h
                    cropped_car = frame[y:y2, x:x2]
                    try:
                        plate_tensor, _, _ = preprocess_frame(cropped_car)
                    except ValueError as ve:
                        # logger.warning(f"[{path}] Invalid cropped_car for preprocessing: {ve}")
                        continue

                    plate_out = models.plate_session.run(None, {models.plate_input: plate_tensor})
                    plate_preds = non_max_suppression(plate_out[0][0], 0.4)
                    for plate in plate_preds:
                        px, py, pw, ph = scale_coords(plate['box'], 640, 640, cropped_car.shape[1], cropped_car.shape[0])
                        px2, py2 = px + pw, py + ph
                        cropped_plate = cropped_car[py:py2, px:px2]
                        if cropped_plate is None or cropped_plate.size == 0:
                            continue
                        plate_text, conf = detect_plate_chars(cropped_plate)
                        if conf >= 50 and len(plate_text) >= 8:
                            db_entries_time(number=plate_text, charConfAvg=conf, plateConfAvg=int(plate['confidence']*100),
                                            croppedPlate=cropped_plate, status="Active", frame=frame, isarvand='notarvand', rtpath=path)
                            break
                        else:
                            deskewed, (nx1, ny1, nx2, ny2) = correct_perspective(cropped_plate)
                            if (nx2 - nx1) > 0 and (ny2 - ny1) > 0:
                                fallback_crop = deskewed[ny1:ny2, nx1:nx2]
                                if fallback_crop is None or fallback_crop.size == 0:
                                    continue
                                plate_text, conf = detect_plate_chars(fallback_crop)
                                if conf >= 47 and len(plate_text) >= 6:
                                    db_entries_time(number=plate_text, charConfAvg=conf, plateConfAvg=int(plate['confidence']*100),
                                                    croppedPlate=fallback_crop, status="Active", frame=frame, isarvand='arvand', rtpath=path)
                                    break
                _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                await websocket.send(base64.b64encode(encoded).decode('utf-8'))
            else:
                await asyncio.sleep(0.015)
    except websockets.ConnectionClosed:
        logger.info(f"Client disconnected from {path}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        clear_memory()

async def ws_handler(websocket):
    await transmit_frames(websocket, websocket.request.path)

async def websocket_server():
    logger.info(f"Starting WebSocket server at ws://{host}:{port}")
    server = await websockets.serve(ws_handler, host, port)
    await asyncio.Future()

if __name__ == "__main__":
    for i, source in enumerate(params.rtps):
        threading.Thread(target=frame_producer, args=(source, frame_buffers[f"/rt{i+1}"]), daemon=True).start()
    asyncio.run(websocket_server())
