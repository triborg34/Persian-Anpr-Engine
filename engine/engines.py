
import gc
import json
import logging
import os
import queue
import socket
import subprocess
import time
from urllib.parse import urlparse
import webbrowser
import numpy as np
import cv2
import requests
import torch
import statistics
from ultralytics import YOLO
from database.db_entries_utils import db_entries_time
from camera import FreshestFrame
import threading
from configParams import Parameters

logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
logging.info(cv2.__version__)
logging.info(torch.__version__)


class CcTvMonitor:
    def __init__(self) -> None:
        self.process = None
        self.loadDb()
        self.regionMode = self.isRegionMode()
        if self.regionMode:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.k = []
      
        self.carConf = 0.6
        self.iou = 0.5
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3
        self.params = Parameters()

        self.lock = threading.Lock()
        self.model_car, self.model_plate, self.model_char, self.dolatimodel = self.loadModels()
        self.quality, self.charConfidence, self.plateConfidence, self.port = self.loadConfig()[
            0:4]
        self._warmup_models()
        self.loadWebBrowser(self.port)

    def loadWebBrowser(self, port: int) -> None:
        webbrowser.open(f'http://127.0.0.1:{port}/web/app')

    def isRegionMode(self) -> bool:
        if os.path.isfile('regions.json'):
            return True
        else:
            return False

    def loadDb(self) -> None:

        try:

            self.process = subprocess.Popen(
                ["pocketbase", "serve", "--http=0.0.0.0:8090"], creationflags=subprocess.CREATE_NO_WINDOW,)
            logging.info(f"PocketBase stater {self.process.pid}")
        except Exception as e:
            logging.info(e)

    def checkrecordMode(self) -> bool:
        if os.path.isfile('recordingmode'):
            return True
        else:
            return False

    def chechOnnx(self) -> str:
        directory = 'model'
        if not os.path.isdir(directory):
            logging.warning("Model directory not found, defaulting to 'pt'")
            return 'pt'
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if filename == "onnx":
                    logging.info("Found 'onnx' sentinel file")
                    return "onnx"
                elif filename == "pt":
                    logging.info("Found 'pt' sentinel file")
                    return "pt"
        return 'pt'

    def chechOpenvino(self) -> bool:
        directory = 'model'
        if not os.path.isdir(directory):
            return False
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if 'openvino' in filename.lower() and not filename.startswith('.'):
                    logging.info(f"Found OpenVINO model: {filename}")
                    return True
        return False

    def loadModels(self) -> tuple:
        fileEx = self.chechOnnx()
        logging.info("Loading YOLO models...")
        model_car = None
        model_plate = None
        model_char = None
        dolatimodel = None

        try:
            if self.chechOpenvino() and self.device.type == 'cpu':
                logging.info("Loading openvino")
                model_char = torch.hub.load(
                    'yolov5', 'custom', f'model/CharsYolo_openvino_model', source='local', device=self.device, force_reload=True)
                model_plate = torch.hub.load(
                    'yolov5', 'custom', f'model/plateYolo_openvino_model', source='local', device=self.device, force_reload=True)
                model_car = YOLO(f'model/yolov8n_openvino_model', task='detect')
                dolatimodel = YOLO('model/dolditector_openvino_model', task='detect')
            else:
                logging.info("Loading onnx/pt")
                model_char = torch.hub.load(
                    'yolov5', 'custom', f'model/CharsYolo.{fileEx}', source='local', device=self.device, force_reload=True)
                model_plate = torch.hub.load(
                    'yolov5', 'custom', f'model/plateYolo.{fileEx}', source='local', device=self.device, force_reload=True)
                model_car = YOLO(f'model/yolov8n.{fileEx}', task='detect')
                dolatimodel = YOLO(f'model/dolditector.{fileEx}', task='detect')
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            if model_car is None or model_plate is None or model_char is None:
                logging.critical("Core models failed to load. System cannot operate.")
                raise RuntimeError("Failed to load core YOLO models") from e
            logging.warning("Some models failed to load, running with partial capabilities")

        logging.info("Models loaded successfully")
        with self.lock:
            return model_car, model_plate, model_char, dolatimodel

    def _warmup_models(self) -> None:
        logging.info("Warming up models...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            if self.model_car is not None:
                self.model_car(dummy, device=self.device, verbose=False)
            if self.model_plate is not None:
                self.model_plate(dummy)
            if self.model_char is not None:
                self.model_char(dummy)
            if self.dolatimodel is not None:
                self.dolatimodel(dummy, verbose=False)
        except Exception as e:
            logging.warning(f"Model warmup failed: {e}")
        logging.info("Model warmup complete")

    def loadConfig(self) -> tuple:
        url = 'http://127.0.0.1:8090/api/collections/setting/records'
        response = requests.get(url).json()
        with self.lock:
            quality = response['items'][0]['quality']
            charConfidence = response['items'][0]['charConf']
            plateConfidence = response['items'][0]['plateConf']
            port = response['items'][0]['port']
            return quality, charConfidence, plateConfidence, port

    def graceful_shutdown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait(1)

        logging.info("Cleanup complete. Shutting down.")


class CameraManager:
    def __init__(self, source: str, config: CcTvMonitor, camera_id: int):
        self.source = source
        self.config = config
        self.camera_id = camera_id

        # ---------- STATE ----------
        self.running = False
        self.client_count = 0
        self.client_lock = threading.Lock()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self.result_frame = None
        # ---------- THREADS ----------
        self.capture_thread = None
        self.process_thread = None
        self.stop_event = threading.Event()

        # ========== LOCK-FREE BUFFERS ==========
        self.capture_buffer = [None, None]
        self.display_buffer = [None, None]
        self.capture_write_idx = 0
        self.capture_read_idx = 0
        self.display_write_idx = 0
        self.display_read_idx = 0

        self.capture_version = 0
        self.display_version = 0

        # ========== OPTIMIZED QUEUES ==========
        self.frame_queue = queue.Queue(maxsize=2)

        # ---------- DATA ----------
        self.processed_tracks = set()

        # ========== REGION MASK CACHE ==========
        self._cached_region_masks = None
        self._cached_regions_key = None
        self._cached_frame_shape = None
        self._combined_region_mask = None
        self._combined_regions_key = None

        # ========== CACHED CV OBJECTS ==========
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._hsv_lower1 = np.array([0, 70, 50])
        self._hsv_upper1 = np.array([10, 255, 255])
        self._hsv_lower2 = np.array([170, 70, 50])
        self._hsv_upper2 = np.array([180, 255, 255])

        if self.config.regionMode:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.k = []

    def start(self):
        self.running = True
        self.stop_event.clear()

        self.capture_thread = threading.Thread(
            target=self.generate_frames, args=[
                self.camera_id, self.source], daemon=True
        )
        self.process_thread = threading.Thread(
            target=self.process_frame, daemon=True
        )

        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        self.running = False
        self.stop_event.set()

    def add_client(self):
        with self.client_lock:
            self.client_count += 1

            if self.client_count == 1:
                self.start()

    def remove_client(self):
        with self.client_lock:
            self.client_count -= 1

            if self.client_count == 0:
                self.stop()

    def sendFrames(self):
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
        while self.running:
            with self._frame_lock:
                frame = self.result_frame

            if frame is None:
                time.sleep(0.005)
                continue

            success, jpeg = cv2.imencode(".jpg", frame, jpeg_params)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    def generate_frames(self, camera_idx: int, source: str):
        """Generate frames from a specific camera feed"""
        if not self.is_connection_alive(source):
            logging.warning(f"[Camera {camera_idx}] Connection not available")
            return

        counter = 0
        if self.config.regionMode:
            regions = self.loadRegions(soruce=source)

            if not hasattr(self, 'k'):
                self.k = []
        else:
            regions = None

        fresh = FreshestFrame(source)

        try:
            while self.running and not self.stop_event.is_set():

                success, frame = fresh.read()

                if frame is None:
                    continue
                with self._frame_lock:
                    self._latest_frame = frame

                try:
                    self.frame_queue.put_nowait(
                        (f'/rt{camera_idx}', counter, regions))
                except queue.Full:
                    pass
                counter += 1

        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
        finally:
            logging.info("Releasing camera resources")
            fresh.release()
            cv2.destroyAllWindows()

    def process_frame(self):
        last_capture_version = -1
        """Process a single frame for object detection"""
        if self.config.regionMode:
            self._combined_region_mask = None
            self._combined_regions_key = None

        while self.running:
            try:

                item = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                if not self.running:
                    break
                continue
            if item is None:
                logging.info("process_frame shutdown signal received")
                break
            path, counter, regions = item
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None or frame.size == 0:
                continue

            try:
                processed_frame = frame

                if self.config.regionMode:
                    regions_key = id(regions) if regions else None
                    frame_shape_key = (frame.shape[0], frame.shape[1])
                    if (self._cached_region_masks is None or
                            self._cached_regions_key != regions_key or
                            self._cached_frame_shape != frame_shape_key):
                        self._cached_region_masks = self.generate_region_masks(
                            frame.shape, regions)
                        self._cached_regions_key = regions_key
                        self._cached_frame_shape = frame_shape_key
                        self._combined_region_mask = None

                    region_masks = self._cached_region_masks

                    if self._combined_region_mask is None:
                        combined_mask = np.zeros(
                            processed_frame.shape[:2], dtype=np.uint8)
                        for mask in region_masks.values():
                            cv2.bitwise_or(combined_mask, mask, dst=combined_mask)
                        self._combined_region_mask = combined_mask

                    masked_frame = cv2.bitwise_and(
                        processed_frame, processed_frame, mask=self._combined_region_mask)
                    self.k.clear()
                    current_regions = []

                with torch.inference_mode():
                    car_res = self.config.model_car(
                        masked_frame if self.config.regionMode else processed_frame, device=self.config.device, classes=[2, 5, 7], verbose=False, conf=self.config.carConf, iou=self.config.iou)

                for res in car_res:
                    for i in range(len(res.boxes.xyxy)):
                        x1, y1, x2, y2 = res.boxes.xyxy[i].int().tolist()

                        if self.config.regionMode:
                            region_name = self.get_detection_region(
                                (x1, y1, x2, y2), region_masks)
                            if region_name and region_name in regions:
                                region_data = regions[region_name]
                                if region_data not in current_regions:
                                    current_regions.append(region_data)
                        else:
                            region_data = None

                        cv2.rectangle(processed_frame, (x1, y1),
                                      (x2, y2), (255, 0, 0), 2)
                        cropped_car = masked_frame[y1:y2,
                                                   x1:x2] if self.config.regionMode else processed_frame[y1:y2, x1:x2]
                        if cropped_car.size == 0:
                            continue

                        plate_results = self.config.model_plate(
                            cropped_car).pandas().xyxy[0]

                        if not plate_results.empty:
                            for _, plate in plate_results.iterrows():
                                plate_conf = int(plate['confidence'] * 100)
                                if plate_conf >= int(self.config.plateConfidence*100):
                                    x_min, y_min, x_max, y_max = (
                                        int(plate['xmin']), int(plate['ymin']),
                                        int(plate['xmax']), int(plate['ymax'])
                                    )

                                    if (y_min >= y_max or x_min >= x_max or
                                        y_min < 0 or x_min < 0 or
                                        y_max > cropped_car.shape[0] or
                                            x_max > cropped_car.shape[1]):
                                        continue

                                    cropped_plate = cropped_car[y_min:y_max,
                                                                x_min:x_max]

                                    if cropped_plate.size == 0:
                                        continue

                                    plate_text, char_conf_avg = self.detect_plate_chars(
                                        cropped_plate)

                                    cv2.rectangle(
                                        cropped_car, (x_min, y_min), (x_max, y_max), (60, 119, 0), 2)
                                    plate_text = plate_text.replace(
                                        'Taxi', 'x')
                                    confidence = float(
                                        self.config.charConfidence) * 100

                                    if char_conf_avg >= confidence and len(plate_text) >= 8:
                                        cv2.putText(cropped_car, f"Plate: {plate_text}", (x_min, y_min - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

                                        db_entries_time(
                                            number=plate_text,
                                            charConfAvg=char_conf_avg,
                                            plateConfAvg=plate_conf,
                                            croppedPlate=cropped_plate,
                                            status="Active",
                                            frame=processed_frame,
                                            isarvand='notarvand',
                                            rtpath=path,
                                            quality=self.config.quality
                                        )
                                        break

                                    else:
                                        deskewed_plate, (newx1, newy1, newx2, newy2) = self.correct_perspective(
                                            cropped_plate, 1.0)
                                        if deskewed_plate.size == 0:
                                            continue

                                        newx1 = max(0, newx1)
                                        newy1 = max(0, newy1)
                                        newx2 = min(
                                            deskewed_plate.shape[1], newx2)
                                        newy2 = min(
                                            deskewed_plate.shape[0], newy2)

                                        if (newx2 <= newx1) or (newy2 <= newy1):
                                            newx1, newy1 = 0, 0
                                            newx2, newy2 = deskewed_plate.shape[1], deskewed_plate.shape[0]

                                        d = newy2 - newy1
                                        tempyMax = newy1 + int(d / 2)

                                        if (tempyMax > newy1 and newx2 > newx1 and
                                            newy1 >= 0 and newx1 >= 0 and
                                            tempyMax <= deskewed_plate.shape[0] and
                                                newx2 <= deskewed_plate.shape[1]):

                                            cropped_plate_nesf = deskewed_plate[newy1:tempyMax, newx1:newx2]

                                            if cropped_plate_nesf.size > 0:
                                                plate_text_arvnad, char_conf_arvnad = self.detect_plate_chars(
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
                                                        frame=processed_frame,
                                                        isarvand='arvand',
                                                        rtpath=path,
                                                        quality=self.config.quality
                                                    )

                if self.config.regionMode:
                    self.k = current_regions
                    self.onDisplay(self.k, frame)
                    display_frame = self.draw_regions_on_frame(
                        processed_frame, regions)
                else:
                    display_frame = processed_frame

                with self._frame_lock:
                    self.result_frame = display_frame

            except Exception as ex:
                logging.error(f"Error in process_frame: {ex}")
                with self._frame_lock:
                    self.result_frame = frame

    def is_connection_alive(self, source: str) -> bool:
        """Check if network connection to source is alive using socket"""
        hostname = urlparse(source).hostname
        if not hostname:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((hostname, 554))
            sock.close()
            return result == 0
        except (socket.error, OSError):
            return False

    def draw_regions_on_frame(self, frame: np.ndarray, regions: dict) -> np.ndarray:
        """Draw region boundaries on frame"""
        overlay = frame.copy()

        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            color_name = region_data.get('color', 'red')
            shape_type = region_data.get('shape_type', 'polygon')

            # Convert color name to BGR
            color_map = {
                'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
                'yellow': (0, 255, 255), 'purple': (128, 0, 128),
                'orange': (0, 165, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255)
            }
            color = color_map.get(color_name, (0, 0, 255))

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, color, 2)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

            # Add region label
            if points:
                center_x = int(sum(p[0] for p in points) / len(points))
                center_y = int(sum(p[1] for p in points) / len(points))

                # Add background for text
                text = f"{region_name} (ID: {region_data.get('id', 'N/A')})"
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                # cv2.rectangle(overlay, (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                #               (center_x + text_size[0]//2 + 5, center_y + 5), (0, 0, 0), -1)
                # cv2.putText(overlay, text, (center_x - text_size[0]//2, center_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return overlay

    def onDisplay(self, region: list, frame: np.ndarray) -> None:
        """Display region names on frame"""
        if not region:  # More pythonic than len(region) == 0
            return

        # Display up to the first few regions with proper spacing
        y_offset = 30  # Starting Y position
        line_height = 50  # Space between lines

        # Limit to 5 regions to avoid overcrowding
        for i, reg in enumerate(region[:5]):
            if 'name' in reg:
                y_pos = y_offset + (i * line_height)
                cv2.putText(frame, reg['name'], (10, y_pos),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

    def get_detection_region(self, detection_box: tuple, region_masks: dict) -> str | None:

        cx = int((detection_box[0] + detection_box[2]) / 2)
        cy = int((detection_box[1] + detection_box[3]) / 2)
        for region_name, mask in region_masks.items():

            if cy < mask.shape[0] and cx < mask.shape[1] and mask[cy, cx] > 0:
                return region_name  # First match wins
        return None

    def generate_region_masks(self, frame_shape: tuple, regions: dict) -> dict:
        """Create binary masks for each region (once)"""
        h, w, _ = frame_shape
        masks = {}
        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            shape_type = region_data.get('shape_type', 'polygon')

            mask = np.zeros((h, w), dtype=np.uint8)

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)  # use thickness

            masks[region_name] = mask
        return masks

    def correct_perspective(self, image: np.ndarray, scale_factor: float) -> tuple[np.ndarray, tuple]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            gray = cv2.medianBlur(gray, 3)
            gray = self._clahe.apply(gray)

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
            logging.error(f"Error in correct_perspective: {e}")
            return image, (0, 0, 0, 0)

    def is_red_plate(self, img: np.ndarray) -> bool:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self._hsv_lower1, self._hsv_upper1)
        mask2 = cv2.inRange(hsv, self._hsv_lower2, self._hsv_upper2)
        mask = cv2.add(mask1, mask2)
        red_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
        return red_ratio > 0.15

    def dolatireader(self, img: np.ndarray):
        with torch.inference_mode():
            results = self.config.dolatimodel(img, conf=0.7)

        boxes = results[0].boxes

        bbox_char = boxes.xyxy
        cls_char = boxes.cls
        conf_char = boxes.conf

        if len(cls_char) > 0:
            keys = cls_char.cpu().numpy().astype(np.int32)
            x_positions = bbox_char[:, 0].cpu().numpy().astype(np.int32)
            confidences = conf_char.cpu().numpy()

            sorted_indices = np.argsort(x_positions)
            sorted_keys = keys[sorted_indices]
            sorted_confidences = confidences[sorted_indices]

            plate_text = ''.join([
                self.config.params.charclasssnames[k]
                for k in sorted_keys
            ])

            char_conf_avg = round(float(np.mean(sorted_confidences)) * 100)
            return plate_text, char_conf_avg

    def detect_plate_chars(self, cropped_plate: np.ndarray) -> tuple[str, int]:
        if self.is_red_plate(cropped_plate):
            plate_text, char_conf_avg = self.dolatireader(cropped_plate)
            if plate_text and len(plate_text.strip()) > 0:
                return plate_text, char_conf_avg

        chars, confidences = [], []
        with torch.inference_mode():
            results = self.config.model_char(cropped_plate)
        detections = sorted(results.pred[0], key=lambda x: x[0])
        for det in detections:
            conf = det[4]
            if conf > 0.5:
                cls = int(det[5].item())
                char = self.config.params.char_id_dict.get(str(cls), '')
                chars.append(char)
                confidences.append(conf.item())
        char_conf_avg = round(statistics.mean(confidences)
                              * 100) if confidences else 0
        return ''.join(chars), char_conf_avg

    def realseFreshest(self) -> None:
        if not self.running:
            return

        self.running = False
        logging.info("Camera pipeline stopped")

    def loadRegions(self, soruce: str, file_path: str = 'regions.json') -> dict:
        url = urlparse(soruce).hostname
        """Load regions from JSON file"""
        try:
            with open(file_path, 'r') as f:
                datas = json.load(f)
                for data in datas:
                    if url == data['ip']:
                        return data.get('regions', {})
                    else:
                        pass

        except Exception as e:
            logging.error(f"Error loading regions: {e}")
            return {}


def emailHandler(email: str, plateNumber: str, edate: str, etime: str) -> None:

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    SENDER_EMAIL = os.environ.get("ANPR_EMAIL", "")
    SENDER_PASSWORD = os.environ.get("ANPR_EMAIL_PASSWORD", "")

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        logging.error("ANPR_EMAIL and ANPR_EMAIL_PASSWORD environment variables not set")
        return

    RECIPIENT_EMAIL = email

    subject = f"{edate} شناسایی پلاک در تاریخ "
    body = f""" 
    پلاک:\n{plateNumber}
    تاریخ:\n{edate}
    زمان:\n{etime}
     """

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    server = None
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logging.info("Email sent successfully!")

    except Exception as e:
        logging.error(f"Failed to send email: {e}")

    finally:
        if server:
            server.quit()


