
import gc
import json
import logging
import os
import queue
import re
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
import uuid
from concurrent.futures import ThreadPoolExecutor
from configParams import Parameters

# Configure logging only if nothing else (e.g. api.py) already did. This avoids
# the previous bug where db_entries_utils imported first and silently forced
# DEBUG level globally, flooding log.txt on every frame.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("log.txt", mode='a', encoding='utf-8'),
            logging.StreamHandler(),
        ],
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
        self.dolatiConf = 0.4
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3
        self.params = Parameters()

        self.lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.model_car, self.model_plate, self.model_char, self.dolatimodel = self.loadModels()
        self.quality, self.charConfidence, self.plateConfidence, self.port = self.loadConfig()
        self._warmup_models()
        self.loadWebBrowser(self.port)
        # self._settings_thread = threading.Thread(target=self._settings_listener, daemon=True)
        # self._settings_thread.start()

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
                ["pocketbase", "serve", "--http=0.0.0.0:8090"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            logging.info(f"PocketBase started with PID {self.process.pid}")
        except FileNotFoundError:
            logging.error("PocketBase executable not found in PATH")
        except Exception as e:
            logging.error(f"Failed to start PocketBase: {e}")

    def _check_pocketbase_health(self) -> bool:
        """Check if PocketBase process is still alive, restart if needed."""
        if self.process is None:
            return False
        if self.process.poll() is not None:
            logging.warning(
                f"PocketBase process exited with code {self.process.returncode}, restarting..."
            )
            self.loadDb()
            return self.process is not None and self.process.poll() is None
        return True

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
            use_openvino = self.chechOpenvino() and self.device.type == 'cpu'
            if use_openvino:
                logging.info("Loading openvino")
                model_char = torch.hub.load(
                    'yolov5', 'custom', 'model/CharsYolo_openvino_model', source='local', device=self.device)
                model_plate = torch.hub.load(
                    'yolov5', 'custom', 'model/plateYolo_openvino_model', source='local', device=self.device)
                model_car = YOLO('model/yolov8n_openvino_model', task='detect')
                dolatimodel = YOLO('model/dolditector_openvino_model', task='detect')
            else:
                logging.info("Loading onnx/pt")
                model_char = torch.hub.load(
                    'yolov5', 'custom', f'model/CharsYolo.{fileEx}', source='local', device=self.device)
                model_plate = torch.hub.load(
                    'yolov5', 'custom', f'model/plateYolo.{fileEx}', source='local', device=self.device)
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

    def set_detection_params(self, car_conf: float = None, iou: float = None,
                             dolati_conf: float = None) -> None:
        with self.lock:
            if car_conf is not None:
                self.carConf = car_conf
            if iou is not None:
                self.iou = iou
            if dolati_conf is not None:
                self.dolatiConf = dolati_conf

    def loadConfig(self, max_retries: int = 5, initial_delay: float = 1.0) -> tuple:
        url = 'http://127.0.0.1:8090/api/collections/setting/records'
        last_error = None
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                items = data.get('items', [])
                if not items:
                    raise ValueError("Empty settings collection")
                with self.lock:
                    quality = items[0]['quality']
                    charConfidence = items[0]['charConf']
                    plateConfidence = items[0]['plateConf']
                    port = items[0]['port']
                    return quality, charConfidence, plateConfidence, port
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logging.warning(
                        f"loadConfig attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 10)
        logging.critical(f"loadConfig failed after {max_retries} attempts: {last_error}")
        raise RuntimeError(f"Failed to load config from PocketBase after {max_retries} attempts") from last_error
    def updateSetting(self):
        try:
            self.quality, self.charConfidence, self.plateConfidence = self.loadConfig()[0:3]
            return f"Sucsess : {self.quality=} , {self.charConfidence=} , {self.plateConfidence}"
        except Exception as e:
          
            return str(e)


    # def _settings_listener(self):
    #     base_url = 'http://127.0.0.1:8090'
    #     backoff = 1

    #     while not self._shutdown_event.is_set():
    #         try:
    #             logging.info("[Realtime] Connecting to PocketBase realtime...")
    #             resp = requests.get(
    #                 f'{base_url}/api/realtime',
    #                 stream=True,
    #                 timeout=(5, 30)
    #             )
    #             logging.info("[Realtime] Connected to PocketBase realtime")

    #             client_id = None
    #             event_type = None
    #             raw_data = None
    #             for line in resp.iter_lines(decode_unicode=True):
    #                 if line is None or self._shutdown_event.is_set():
    #                     break

    #                 if line.startswith('event:'):
    #                     event_type = line[len('event:'):].strip()
    #                 elif line.startswith('data:'):
    #                     raw_data = line[len('data:'):].strip()
    #                 elif line == '':
    #                     if event_type == 'PB_CONNECT' and raw_data:
    #                         parsed = json.loads(raw_data)
    #                         client_id = parsed.get("clientId", parsed) if isinstance(parsed, dict) else parsed
    #                         logging.info(f"[Realtime] Got clientId={client_id!r}")

    #                         sub_payload = json.dumps({
    #                             "clientId": client_id,
    #                             "subscriptions": ["setting/*"]
    #                         })
    #                         logging.info(f"[Realtime] Sending subscribe: {sub_payload}")
    #                         sub_resp = requests.post(
    #                             f'{base_url}/api/realtime',
    #                             data=sub_payload,
    #                             headers={"Content-Type": "application/json"},
    #                             timeout=5
    #                         )
    #                         logging.info(f"[Realtime] Subscribe response: {sub_resp.status_code} {sub_resp.text}")
    #                         backoff = 1

                        
    #                     elif event_type == 'PB_RECORDS' or event_type.startswith("setting") and raw_data:
    #                         try:
    #                             data = json.loads(raw_data)
    #                             record = data.get('record', {})
    #                             with self.lock:
    #                                 if 'quality' in record:
    #                                     self.quality = record['quality']
    #                                 if 'charConf' in record:
    #                                     self.charConfidence = record['charConf']
    #                                 if 'plateConf' in record:
    #                                     self.plateConfidence = record['plateConf']
                
    #                             logging.info(
    #                                 f"[Realtime] Settings updated - quality={self.quality}, "
    #                                 f"charConf={self.charConfidence}, plateConf={self.plateConfidence}"
    #                             )
    #                         except Exception as e:
    #                             logging.error(f"[Realtime] Failed to parse update: {e}")

    #                     event_type = None
    #                     raw_data = None

    #         except Exception as e:
    #             if self._shutdown_event.is_set():
    #                 break
    #             logging.warning(f"[Realtime] Connection lost: {e}")

    #         if not self._shutdown_event.is_set():
    #             self._check_pocketbase_health()
    #             logging.info(f"[Realtime] Reconnecting in {backoff}s...")
    #             self._shutdown_event.wait(timeout=backoff)
    #             backoff = min(backoff * 2, 30)
    def graceful_shutdown(self) -> None:
        self._shutdown_event.set()

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
        # Per-connection tokens (see add_client): a set, not a counter, so
        # the two cleanups of one request (watch_disconnect + sendFrames'
        # finally) can't steal a *new* request's slot on fast reconnect.
        self._clients = set()
        self.client_lock = threading.Lock()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self.result_frame = None
        # ---------- THREADS ----------
        self.capture_thread = None
        self.process_thread = None
        self.stop_event = threading.Event()

        # ========== OPTIMIZED QUEUES ==========
        self.frame_queue = queue.Queue(maxsize=2)

        # ========== DB WRITE QUEUE ==========
        self.db_queue = queue.Queue()
        self.db_writer_thread = None

        # ========== OCR WORKER POOL ==========
        # Runs plate char recognition (and the expensive correct_perspective
        # deskew) off the detection thread so multiple cars in one frame are
        # processed in parallel. Ultralytics/PyTorch inference is thread-safe
        # for concurrent predict() calls and cv2 releases the GIL, so this
        # overlaps CPU work across detections instead of serializing it.
        self._ocr_pool = ThreadPoolExecutor(max_workers=3)

        # YOLOv5 (model_plate / model_char, loaded via torch.hub) is NOT
        # thread-safe: it caches grid/anchor tensors in module state during
        # forward, so concurrent calls with different input sizes corrupt each
        # other ("size of tensor a must match tensor b"). Serialize those calls.
        # RLock because _process_car_ocr may hold it across the detect_plate_chars
        # call (which also acquires it) on the same thread.
        self._model_lock = threading.RLock()

        # ========== STREAM JPEG PARAMS ==========
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 50, cv2.IMWRITE_JPEG_OPTIMIZE, 0]

        # ========== DB WRITE DEDUP ==========
        # single-writer (process_frame thread), so no lock needed
        self._recent_db_puts = {}
        self._last_dedup_cleanup = time.time()
        self._DEDUP_CLEANUP_INTERVAL = 30  # seconds

        # ========== REGION MASK CACHE ==========
        self._cached_region_masks = None
        self._cached_regions_key = None
        self._cached_frame_shape = None
        self._combined_region_mask = None
        self._combined_regions_key = None
        self._cached_full_frame_region = None

        # ========== CACHED CV OBJECTS ==========
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._hsv_lower1 = np.array([0, 70, 50])
        self._hsv_upper1 = np.array([10, 255, 255])
        self._hsv_lower2 = np.array([170, 70, 50])
        self._hsv_upper2 = np.array([180, 255, 255])

        if self.config.regionMode:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.k = []

    REGION_COLORS = {
        'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
        'yellow': (0, 255, 255), 'purple': (128, 0, 128),
        'orange': (0, 165, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255)
    }

    def _ensure_ocr_pool(self) -> None:
        """Recreate the OCR pool if it was shut down by a previous stop().

        stop() shuts the pool down so in-flight OCR cancels promptly.
        A fresh pool is required before the next start(), otherwise
        submit() raises "cannot schedule new futures after shutdown".
        """
        pool = getattr(self, '_ocr_pool', None)
        if pool is None or getattr(pool, '_shutdown', False):
            self._ocr_pool = ThreadPoolExecutor(max_workers=3)

    def _drain_queues(self) -> None:
        for q in (self.frame_queue, self.db_queue):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def _join_old_threads(self, timeout: float = 2.0) -> None:
        for t in (self.capture_thread, self.process_thread,
                  self.db_writer_thread):
            try:
                if t is not None and t.is_alive():
                    t.join(timeout=timeout)
            except Exception:
                pass

    def start(self):
        # Called with client_lock held from add_client(). Guard against
        # double-start and against restarting while old threads linger.
        if self.running:
            proc = getattr(self, 'process_thread', None)
            cap = getattr(self, 'capture_thread', None)
            if ((proc is not None and proc.is_alive())
                    or (cap is not None and cap.is_alive())):
                return
        # Reap any stale threads from the previous run before spawning new
        # ones, otherwise two capture loops would read the same RTSP source.
        self._join_old_threads(timeout=2.0)

        self.running = True
        self.stop_event.clear()
        self._ensure_ocr_pool()
        self._drain_queues()

        self.capture_thread = threading.Thread(
            target=self.generate_frames, args=[
                self.camera_id, self.source], daemon=True
        )
        self.process_thread = threading.Thread(
            target=self.process_frame, daemon=True
        )
        self.db_writer_thread = threading.Thread(
            target=self._db_writer, daemon=True
        )

        self.capture_thread.start()
        self.process_thread.start()
        self.db_writer_thread.start()

    def stop(self):
        # Called with client_lock held from remove_client(). Must NOT block:
        # it runs on the FastAPI event loop via watch_disconnect, so a
        # blocking shutdown(wait=True) would freeze all API responses.
        self.running = False
        self.stop_event.set()
        # Unblock process_frame (polls with timeout, sentinel is best-effort)
        # and _db_writer (blocks forever on get()).
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        except Exception:
            pass
        try:
            self.db_queue.put(None)
        except Exception:
            pass
        pool = getattr(self, '_ocr_pool', None)
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _db_writer(self):
        while True:
            try:
                item = self.db_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running or self.stop_event.is_set():
                    break
                continue
            if item is None:
                break
            try:
                db_entries_time(**item)
            except Exception as e:
                logging.error(
                    f"[Camera {self.camera_id}] DB write failed for plate "
                    f"{item.get('number', '?')}: {e}"
                )

    def _queue_db_entry(self, entry: dict) -> None:
        """Enqueue a plate write for the background worker.

        Mirrors db_entries_time's 10s reserve_plate window here in the
        producer, so consecutive frames of the same plate don't pile up in
        the queue while the worker is busy writing the first one to disk.
        """
        now = time.time()

        # Periodic cleanup of stale entries to prevent memory growth
        if now - self._last_dedup_cleanup >= self._DEDUP_CLEANUP_INTERVAL:
            self._recent_db_puts = {
                k: ts for k, ts in self._recent_db_puts.items() if now - ts <= 10
            }
            self._last_dedup_cleanup = now

        key = (entry['rtpath'], entry['number'])
        if key in self._recent_db_puts:
            return
        self._recent_db_puts[key] = now
        self.db_queue.put(entry)

    def add_client(self) -> str:
        """Register one streaming connection. Returns its token."""
        token = uuid.uuid4().hex
        with self.client_lock:
            was_empty = not self._clients
            self._clients.add(token)
            if was_empty:
                self.start()
        return token

    def remove_client(self, token: str | None = None) -> None:
        # Each request cleans up twice (watch_disconnect + sendFrames'
        # finally). Tokens make the second call a no-op for that request
        # instead of stealing a new request's slot after a fast reconnect.
        with self.client_lock:
            if token is None:
                # Legacy fallback: drop a single arbitrary client, if any.
                if not self._clients:
                    return
                self._clients.pop()
            else:
                if token not in self._clients:
                    return
                self._clients.discard(token)

            if not self._clients:
                self.stop()

    def has_client(self, token: str) -> bool:
        with self.client_lock:
            return token in self._clients

    def sendFrames(self, token: str | None = None):
        jpeg_params = self._jpeg_params
        try:
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
        finally:
            if token is not None:
                self.remove_client(token)

    def generate_frames(self, camera_idx: int, source: str):
        """Generate frames from a specific camera feed"""
        # Wait for the camera to become reachable instead of exiting at
        # once: a transient network drop must not permanently kill the
        # pipeline and force an app restart.
        while self.running and not self.stop_event.is_set():
            try:
                if self.is_connection_alive(source):
                    break
            except Exception:
                pass
            logging.warning(
                f"[Camera {camera_idx}] Connection not available, retrying..."
            )
            self.stop_event.wait(timeout=3.0)
        if not self.running or self.stop_event.is_set():
            return

        counter = 0
        if self.config.regionMode:
            regions = self.loadRegions(soruce=source)

            if not hasattr(self, 'k'):
                self.k = []
        else:
            regions = None

        fresh = None
        try:
            fresh = FreshestFrame(source)
            last_seq = -1

            while self.running and not self.stop_event.is_set():
                # Timeout so stop() can interrupt the loop promptly;
                # without it read() blocks forever and restart duplicates
                # capture threads.
                seq, frame = fresh.read(timeout=0.5)

                if not self.running or self.stop_event.is_set():
                    break
                # Reader thread died unexpectedly -> rebuild it so the
                # stream recovers instead of freezing on a stale frame.
                if not fresh.is_alive():
                    logging.warning(
                        f"[Camera {camera_idx}] Frame reader died, recreating..."
                    )
                    try:
                        fresh.release()
                    except Exception:
                        pass
                    self.stop_event.wait(timeout=2.0)
                    if not self.running or self.stop_event.is_set():
                        break
                    fresh = FreshestFrame(source)
                    last_seq = -1
                    continue

                if frame is None:
                    continue
                if seq == last_seq:
                    continue  # read() timed out, no new frame yet
                last_seq = seq

                with self._frame_lock:
                    self._latest_frame = frame

                try:
                    self.frame_queue.put_nowait(
                        (f'/rt{camera_idx}', counter, regions))
                except queue.Full:
                    pass
                counter += 1

        except Exception as e:
            logging.error(
                f"[Camera {camera_idx}] Error in generate_frames: {e}",
                exc_info=True
            )
        finally:
            logging.info(f"[Camera {camera_idx}] Releasing camera resources")
            if fresh is not None:
                try:
                    fresh.release()
                except Exception as e:
                    logging.warning(f"Error releasing FreshestFrame: {e}")
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logging.warning(f"Error closing OpenCV windows: {e}")

    def process_frame(self):
     
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
                    if not regions:
                        # camera has no entry (or empty regions) in
                        # regions.json: detect over the whole frame instead
                        regions = self._full_frame_region(frame.shape)
                    regions_key = id(regions)
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

                # detection_input: region-masked frame for YOLO inference
                # processed_frame: original frame used for drawing boxes/labels
                detection_input = masked_frame if self.config.regionMode else processed_frame

                with torch.inference_mode():
                    car_res = self.config.model_car(
                        detection_input, device=self.config.device, classes=[2, 5, 7],
                        verbose=False, conf=self.config.carConf, iou=self.config.iou)

                ocr_futures = []
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

                        cv2.rectangle(processed_frame, (x1, y1),
                                      (x2, y2), (255, 0, 0), 2)
                        cropped_car = detection_input[y1:y2, x1:x2]
                        if cropped_car.size == 0:
                            continue

                        # Offload plate OCR + deskew to the worker pool so
                        # multiple cars in this frame are recognized in parallel.
                        if not self.running or self.stop_event.is_set():
                            break
                        try:
                            pool = self._ocr_pool
                            ocr_futures.append(
                                pool.submit(
                                    self._process_car_ocr, processed_frame, cropped_car, path))
                        except RuntimeError:
                            # Pool is shutting down (stop() called during
                            # disconnect); skip this car. The next start()
                            # recreates the pool, so this is transient.
                            if self.running and not self.stop_event.is_set():
                                logging.warning(
                                    f"[Camera {self.camera_id}] OCR pool shut down, "
                                    f"recreating..."
                                )
                                try:
                                    self._ensure_ocr_pool()
                                except Exception:
                                    pass
                            pass

                # Wait for all plate recognition for this frame to finish before
                # publishing, so every label/box is present and there are no
                # cross-frame races on the shared frame buffer.
                for _fut in ocr_futures:
                    try:
                        _fut.result(timeout=5.0)
                    except Exception as e:
                        logging.warning(
                            f"[Camera {self.camera_id}] OCR worker failed: {e}"
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
                logging.error(
                    f"[Camera {self.camera_id}] Error in process_frame: {ex}",
                    exc_info=True
                )
                with self._frame_lock:
                    self.result_frame = frame

    def is_connection_alive(self, source) -> bool:
        """Check if network connection to source is alive using socket"""
        # Local webcam index (int 0 or "0") has no hostname to probe.
        try:
            if isinstance(source, int):
                return True
            if isinstance(source, str) and source.strip().isdigit():
                return True
            hostname = urlparse(source).hostname
        except Exception:
            return False
        if not hostname:
            return False
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((hostname, 554))
            return result == 0
        except (socket.error, OSError):
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def draw_regions_on_frame(self, frame: np.ndarray, regions: dict) -> np.ndarray:
        """Draw region boundaries directly on the frame"""
        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            color_name = region_data.get('color', 'red')
            shape_type = region_data.get('shape_type', 'polygon')

            color = self.REGION_COLORS.get(color_name, (0, 0, 255))

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [pts], True, color, 2)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        return frame

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

    def _process_car_ocr(self, frame, cropped_car, path):
        """Detect and OCR plates inside one car crop.

        Runs on a worker thread (self._ocr_pool) so several cars in the same
        frame are recognized concurrently instead of one after another. It
        draws the plate box/label directly onto ``cropped_car`` (a view of the
        working frame) and enqueues DB writes. ``process_frame`` joins every
        car future before publishing the frame, so there are no cross-frame
        races on the shared frame buffer.

        Note: this relies on Ultralytics/PyTorch inference being thread-safe
        for concurrent ``predict()`` calls, and cv2 releasing the GIL.
        """
        plate_min = int(self.config.plateConfidence * 100)
        char_min = float(self.config.charConfidence) * 100
        try:
            with self._model_lock, torch.inference_mode():
                plate_res = self.config.model_plate(cropped_car)
            # (x1, y1, x2, y2, conf, cls) rows, no pandas involved
            for pbox in plate_res.xyxy[0].tolist():
                x_min, y_min, x_max, y_max = (
                    int(pbox[0]), int(pbox[1]),
                    int(pbox[2]), int(pbox[3])
                )
                plate_conf = int(pbox[4] * 100)

                if plate_conf < plate_min:
                    continue

                if (y_min >= y_max or x_min >= x_max or
                        y_min < 0 or x_min < 0 or
                        y_max > cropped_car.shape[0] or
                        x_max > cropped_car.shape[1]):
                    continue

                cropped_plate = cropped_car[y_min:y_max, x_min:x_max]
                if cropped_plate.size == 0:
                    continue

                plate_text, char_conf_avg = self.detect_plate_chars(
                    cropped_plate)

                cv2.rectangle(
                    cropped_car, (x_min, y_min), (x_max, y_max), (60, 119, 0), 2)
                plate_text = plate_text.replace('Taxi', 'x')

                if char_conf_avg >= char_min and len(plate_text) >= 8:
                    cv2.putText(cropped_car, f"Plate: {plate_text}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

                    self._queue_db_entry(
                        dict(
                            number=plate_text,
                            charConfAvg=char_conf_avg,
                            plateConfAvg=plate_conf,
                            croppedPlate=cropped_plate,
                            status="Active",
                            frame=frame,
                            isarvand='notarvand',
                            rtpath=path,
                            quality=self.config.quality
                        ))
                    break

                else:
                    deskewed_plate, (newx1, newy1, newx2, newy2) = self.correct_perspective(
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

                    if (tempyMax > newy1 and newx2 > newx1 and
                            newy1 >= 0 and newx1 >= 0 and
                            tempyMax <= deskewed_plate.shape[0] and
                            newx2 <= deskewed_plate.shape[1]):

                        cropped_plate_nesf = deskewed_plate[newy1:tempyMax, newx1:newx2]

                        if cropped_plate_nesf.size > 0:
                            plate_text_arvnad, char_conf_arvnad = self.detect_plate_chars(
                                cropped_plate_nesf)

                            if len(plate_text_arvnad) >= 5 and char_conf_arvnad >= char_min - 3:
                                cv2.putText(cropped_car, f"Plate: {plate_text_arvnad}", (x_min, y_min - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

                                self._queue_db_entry(
                                    dict(
                                        number=plate_text_arvnad,
                                        charConfAvg=char_conf_arvnad,
                                        plateConfAvg=plate_conf,
                                        croppedPlate=cropped_plate,
                                        status="Active",
                                        frame=frame,
                                        isarvand='arvand',
                                        rtpath=path,
                                        quality=self.config.quality
                                    ))
        except Exception as e:
            logging.error(
                f"[Camera {self.camera_id}] Error in _process_car_ocr: {e}",
                exc_info=True
            )

    # def is_red_plate(self, img: np.ndarray) -> bool:
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     mask1 = cv2.inRange(hsv, self._hsv_lower1, self._hsv_upper1)
    #     mask2 = cv2.inRange(hsv, self._hsv_lower2, self._hsv_upper2)
    #     mask = cv2.add(mask1, mask2)
    #     red_ratio = cv2.countNonZero(mask) / (img.shape[0] * img.shape[1])
    #     return red_ratio > 0.15


    def _looks_like_plate(self,text: str) -> bool:
        """Trust gate for the fast dolatimodel read (before falling back to the
        slower char OCR). The detector only outputs digits + a single provincial
        letter 'A', so a real plate is: digits, one 'A' somewhere in the middle,
        more digits, total length 7-9 (tolerant of +/-1 detection errors that the
        strict ^.{2}A.{5}$ rejected)."""
        if not text or text.count('A') != 1:
            return False
        if not text.replace('A', '').isdigit():
            return False
        a = text.index('A')
        return 7 <= len(text) <= 9 and 1 <= a <= len(text) - 2

    def dolatireader(self, img: np.ndarray):
        try:
            with self._model_lock, torch.inference_mode():
                results = self.config.dolatimodel(img, conf=self.config.dolatiConf)

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
        except Exception as e:
            logging.warning(f"dolatireader inference failed: {e}")
            return None

    def detect_plate_chars(self, cropped_plate: np.ndarray) -> tuple[str, int]:
        result = self.dolatireader(cropped_plate)
        if result is not None:
            plate_text, char_conf_avg = result
            if plate_text and len(plate_text.strip()) > 0:
                if self._looks_like_plate(plate_text):
                    return plate_text, char_conf_avg

        chars, confidences = [], []
        with self._model_lock, torch.inference_mode():
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

        except Exception as e:
            logging.error(f"Error loading regions: {e}")

        logging.warning(
            f"No regions configured for camera {url}, using full-screen region")
        return {}

    def _full_frame_region(self, frame_shape: tuple) -> dict:
        """Full-screen fallback region for cameras with no regions.json entry.

        Cached per frame size so the dict identity (and therefore the region
        mask cache key) stays stable across frames.
        """
        h, w = frame_shape[:2]
        if self._cached_full_frame_region is None or self._cached_full_frame_region[0] != (h, w):
            self._cached_full_frame_region = ((h, w), {
                'full_frame': {
                    'id': 'auto',
                    'name': 'FullFrame',
                    'points': [[0, 0], [w, 0], [w, h], [0, h]],
                    'shape_type': 'rectangle',
                    'color': 'green',
                },
            })
        return self._cached_full_frame_region[1]


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


