# ANPR System — Stability, Reliability & Code Quality Fixes

**Date:** 2026-09-07  
**Scope:** `engine/api.py`, `engine/engines.py`, `engine/camera.py`, `engine/database/db_entries_utils.py`  
**Constraint:** Preserve FPS / inference pipeline performance. No model, resolution, or threshold changes.

---

## 1. Architecture Overview

| Component | File | Role |
|---|---|---|
| FastAPI server | `engine/api.py` | Endpoints `/video_feed`, `/rtsp_feed`, ONVIF, relay, email, health. Owns `camera_registry` + `CcTvMonitor` singleton |
| Core engine | `engine/engines.py` | `CcTvMonitor` (model load, config, PocketBase, settings listener) + `CameraManager` (capture → YOLO car → YOLO plate → OCR → DB queue → MJPEG publish) |
| Frame capture | `engine/camera.py` | `FreshestFrame(threading.Thread)` — RTSP `VideoCapture` with `cond` + `latestnum` |
| DB writer | `engine/database/db_entries_utils.py` | `reserve_plate` / `db_entries_time` → PocketBase `database` collection, `ThreadPoolExecutor(4)` |
| Config | `engine/configParams.py` | Char maps (unchanged) |

Pipeline: `Camera → FreshestFrame → frame_queue(2) → process_frame (YOLO car) → _ocr_pool(3) → _queue_db_entry → db_queue → _db_writer → PocketBase`

---

## 2. Problems Found (Audit)

### Critical (crash / freeze)
1. `engines.py:579` — `for _fut in ocr_futures: _fut.result()` re-raised worker exception → killed `process_frame` thread → camera frozen until app restart.
2. `engines.py:914` — `dolatireader()` no `try/except`, no `_model_lock`. YOLOv5 hub model not thread-safe → concurrent `predict()` corrupted grid cache.
3. `engines.py:178` — `loadConfig()` single `requests.get().json()` no timeout/retry. Raced PocketBase `Popen` → startup crash.
4. `camera.py:56` — `FreshestFrame.run()` fixed 2s reconnect, `release()` blocked `join(timeout=None)` forever.
5. `engines.py:419` — `CameraManager.stop()` was `shutdown(wait=False)` originally, then changed to `wait=True,cancel_futures=True` → blocked FastAPI event loop (`watch_disconnect` runs on loop) → all API responses frozen. Pool never recreated → `RuntimeError: cannot schedule new futures after shutdown` on reconnect.
6. `engines.py:526` / `api.py:115` — `client_count:int` counter. Each HTTP request does two cleanups (`watch_disconnect` + `sendFrames` finally). Fast reconnect: A cleans `1→0→stop`, B adds `0→1→start`, A's late `finally` steals B's slot `1→0→stop` → immediate release.

### High (leak / stall)
7. `engines.py:393` + `db_entries_utils.py:32` — dedup dicts never periodically cleaned → unbounded growth on long run.
8. `engines.py:190` — `_settings_listener` `while True`, `timeout=(5,None)` infinite read, `time.sleep()` not interruptible, no shutdown signal.
9. `engines.py:564` — `generate_frames` `if not is_connection_alive: return` → transient network drop permanently killed pipeline. `fresh.read()` no timeout → `stop()` could not interrupt. `fresh=None` not handled in `finally`.
10. `engines.py:503` `is_connection_alive` — `urlparse(source).hostname` crashed on `int` source `0`.

### Medium
11. `api.py:105` — `cctv.carConf/iou/dolatiConf` written directly per-request while `process_frame` reads → race.
12. `engines.py:606` — `masked_frame` vs `processed_frame` naming confusion, extra ternary per car crop.
13. `engines.py:551` / `camera.py:13` — `cv2.destroyAllWindows()` in headless / `cap.release()` throw.

---

## 3. Fixes Applied

### 3.1 `engine/camera.py`

**`FreshestFrame` lifecycle `camera.py:17-110`:**
- Removed 30-attempt give-up. Now infinite reconnect with capped exponential backoff `RECONNECT_INITIAL_DELAY=2.0` → `RECONNECT_MAX_DELAY=30.0` (`camera.py:18`).
- Backoff sleep interruptible in 0.5s chunks checking `self.running` (`camera.py:94`), so `release()` stays responsive even during 30s wait.
- `cap.read()` wrapped `try/except` (`camera.py:75`), `None` image treated as failure.
- `_create_capture()` wrapped, logs `isOpened()` failure (`camera.py:51`).
- `release(timeout=5.0)` (`camera.py:60`): `join(timeout)` + alive check + `cap.release()` try/except. `thread daemon=True`.

### 3.2 `engine/engines.py` — `CcTvMonitor`

**Config load `engines.py:200`:**
- `loadConfig(max_retries=5, initial_delay=1.0)` retries `requests.get(timeout=5)` with `raise_for_status()`, empty `items` check, `delay = min(delay*2,10)`, raises `RuntimeError` only after 5 attempts.

**Thread-safe params `engines.py:191`:**
- `set_detection_params(car_conf,iou,dolati_conf)` with `self.lock`.

**Logging `engines.py:28`:**
- Guard `if not logging.getLogger().handlers:` prevents duplicate handlers when `api.py` imports first.

**PocketBase `engines.py:80`:**
- `loadDb()` logs `PID`, handles `FileNotFoundError`, adds `_check_pocketbase_health()` (`engines.py:92`) called before settings reconnect.

**Settings listener `engines.py:210`:**
- Added `self._shutdown_event:threading.Event` (`engines.py:63`), `while not _shutdown_event.is_set()`, `timeout=(5,30)`, `iter_lines` break on event, `except` checks event, `wait(timeout=backoff)` instead of `sleep`.

**Shutdown `engines.py:320`:**
- `graceful_shutdown()` sets `_shutdown_event` first, then `empty_cache`/`gc`, then `process.kill()`.

### 3.3 `engine/engines.py` — `CameraManager`

**OCR pool lifecycle `engines.py:400`:**
- `_ensure_ocr_pool()` recreates `ThreadPoolExecutor(3)` if `pool is None or pool._shutdown`.
- `_drain_queues()` drains `frame_queue` + `db_queue` on `start()`.
- `_join_old_threads(2.0)` joins stale capture/process/db threads before spawning new ones.
- `start()` idempotent: if `running` and threads alive → return; else join stale, `running=True`, `stop_event.clear()`, ensure pool, drain queues, spawn 3 daemons.
- `stop()` non-blocking (`wait=False,cancel_futures=True`) + `frame_queue.put_nowait(None)` + `db_queue.put(None)` to unblock both consumers. Does NOT block event loop.

**Client tracking `engines.py:340` + `engines.py:526`:**
- Replaced `client_count:int` with `self._clients:set` of `uuid4` tokens.
- `add_client()->str` token, `was_empty → start()`. `remove_client(token)` discards only that token, no-op on second cleanup of same request. `has_client(token)`. `sendFrames(token)` finally removes only its token. Backward compat `token=None` drops arbitrary entry.

**Queues `engines.py:344` + `engines.py:486`:**
- `frame_queue maxsize=2` kept. `_db_writer` now `get(timeout=0.5)` + break if not running, so old writer is reapable.
- `_recent_db_puts` cleaned every `30s` (`engines.py:448`), not every enqueue.

**Frame pipeline `engines.py:569` + `engines.py:614`:**
- `generate_frames`: retry loop waits for `is_connection_alive` with `stop_event.wait(3s)` instead of instant `return`. `fresh` init `None`-safe. `fresh.read(timeout=0.5)`, skip `seq==last_seq` (timeout duplicate), recreate `FreshestFrame` if `not is_alive()`, log. `finally` guards `fresh is not None`.
- `is_connection_alive(source)` handles `int`/`"0"` → `True` (`engines.py:692`).
- `process_frame`: clarified `detection_input = masked_frame if regionMode else processed_frame` (`engines.py:611`), `cropped_car = detection_input[y1:y2,x1:x2]` single access. `ocr_futures` submit checks `running/stop_event`, recreates pool if `RuntimeError` while still running. `for _fut: try _fut.result(timeout=5.0) except` logs `OCR worker failed` per `engines.py:652`.

**Region handling:**
- `onDisplay`, `draw_regions_on_frame`, `generate_region_masks` unchanged logic, only extra `exc_info=True` on errors.

**Error logging:**
- `generate_frames`, `process_frame`, `_process_car_ocr`, `_db_writer` all log `[Camera {id}]` + `exc_info=True`.

**Dolatireader `engines.py:990`:**
- Wrapped entire inference with `self._model_lock, torch.inference_mode()`, `try/except` return `None`, logs `warning`.

### 3.4 `engine/api.py`

**`video_feed` `api.py:97`:**
- `cctv.set_detection_params(car_conf=0.1,iou=0.5,dolati_conf=0.6)` instead of direct writes.
- `token = cam.add_client()` → `watch_disconnect` checks `is_disconnected()` then `cam.remove_client(token)`, also breaks if `not has_client(token)` (generator already closed). `sendFrames(token)`.

### 3.5 `engine/database/db_entries_utils.py`

- `recent_plates` cleanup every `60s` (`_last_cleanup`, `_CLEANUP_INTERVAL=60`) instead of every call (`db_entries_utils.py:12`).
- `_upload_to_pocketbase` separate `Timeout`/`ConnectionError` logs, includes plate number in log (`db_entries_utils.py:68`).

---

## 4. What Was NOT Changed (FPS Protection)

- Model files / `chechOnnx` / `chechOpenvino` / `loadModels` / `_warmup_models` logic.
- YOLO inference args (`classes=[2,5,7]`, `conf`, `iou`) — only made thread-safe.
- `frame_queue maxsize=2`, `ThreadPoolExecutor(3)` size, `correct_perspective` / `detect_plate_chars` algorithms.
- `sendFrames` MJPEG `imencode` quality `50`, `is_connection_alive` probe port `554`.
- No extra per-frame log, copy, resize, or DB roundtrip in hot path.

---

## 5. Verification

```powershell
python -m py_compile "D:\Codes\anprv8\engine\engines.py"  # OK
python -m py_compile "D:\Codes\anprv8\engine\camera.py"   # OK
python -m py_compile "D:\Codes\anprv8\engine\api.py"      # OK
python -m py_compile "D:\Codes\anprv8\engine\database\db_entries_utils.py"  # OK
```

Manual tests required (no automated FPS harness in repo):
- Normal single camera stream
- Tab disconnect → reconnect quickly (race)
- Two tabs same source (shared pipeline)
- RTSP drop 10s → auto-recover
- No vehicle / multiple vehicles / invalid crop
- Repeated plate (dedup 10s window)
- PocketBase down at startup → retry
- Long run (2h) check `psutil` RAM/thread count, `frame_queue` size, `recent_plates` size.

---

## 6. Known Trade-off

`_ocr_pool` recreation adds one `ThreadPoolExecutor` alloc per camera restart (tab reconnect). Cost <1ms, no FPS impact. Alternative (single global pool) would require cross-camera locking.

---

## 7. File List

- `engine/engines.py` — 90% of changes
- `engine/camera.py` — reconnect + release
- `engine/api.py` — 8 lines (token plumbing)
- `engine/database/db_entries_utils.py` — 20 lines (cleanup + error handling)

---

## 8. Technology Stack — Complete Inventory

### 8.1 Language & Runtime

| Technology | Version | Purpose | Used in |
|---|---|---|---|
| **Python** | 3.10+ (`str\|None` syntax) | Main language, GIL concurrency | all `engine/*.py` |
| **python-dotenv** | 1.1.0 | Load `.env` → `ANPR_API_KEY`, `ANPR_EMAIL` | `api.py:4`, `engines.py:1018` |

### 8.2 Web Framework & Server

| Technology | Version | Purpose | Used in |
|---|---|---|---|
| **FastAPI** | 0.138.0 | Async HTTP, dependency injection, auto docs | `api.py:22` |
| **Starlette** | 1.3.1 | Underlies FastAPI, `StreamingResponse` for MJPEG/SSE | `api.py:24`, `api.py:126` |
| **Uvicorn** | 0.49.0 | ASGI server `0.0.0.0:{port}` | `api.py:32`, `api.py:290` |
| **Pydantic** | 2.13.4 + `pydantic_core` 2.46.4 | `BaseModel` validation `Relay/EmailClass/RtspFields` | `api.py:55` |
| **anyio / h11** | 4.14.0 / 0.16.0 | Async primitives, HTTP/1.1 | transitive via FastAPI |

### 8.3 Concurrency & Async

| Technology | Where | How |
|---|---|---|
| **threading.Thread / Lock / RLock / Event / Condition** | `engines.py:526`, `camera.py:17` | 3-per-camera threads (capture/process/db) + 1-per-camera `FreshestFrame` + `CcTvMonitor._settings_thread`. See §3.3 details |
| **concurrent.futures.ThreadPoolExecutor** | `engines.py:356 (3 workers)`, `db_entries_utils.py:10 (4 workers)` | OCR pool (parallel deskew+char) + DB upload pool. Fix: `_ensure_ocr_pool()` recreates after shutdown |
| **asyncio + asynccontextmanager** | `api.py:42` `lifespan` | `watch_disconnect` `await request.is_disconnected()` `api.py:123`, `discover_onvif_stream` probe |
| **queue.Queue(maxsize=2) / Queue()** | `engines.py:344` | Bounded frame queue (freshest only) + unbounded DB queue |
| **GIL-aware libs** | `torch`, `cv2` | Release GIL during `model()` / `HoughLinesP` → real parallelism despite GIL |

### 8.4 Computer Vision & AI

| Technology | Version | Purpose | Used in |
|---|---|---|---|
| **OpenCV (opencv-python)** | 4.10.0.84 + FFmpeg `CAP_FFMPEG` | `VideoCapture` RTSP, `imencode`, `cvtColor`, `Canny`, `HoughLinesP`, `warpAffine`, `CLAHE`, masks | `camera.py:7`, `engines.py:14`, all vision in `engines.py:690` |
| **PyTorch (torch+cu126 / torchvision)** | 2.12.1 | Inference `torch.inference_mode()`, `torch.device(0/cuda)`, `set_float32_matmul_precision` | `engines.py:16`, `engines.py:914` |
| **Ultralytics YOLO** | 8.4.72 | `YOLO('yolov8n.*')` car detection `classes=[2,5,7]` | `engines.py:18`, `engines.py:615` |
| **YOLOv5 (torch.hub)** | vendored `engine/yolov5/` + `hubconf.py` | `torch.hub.load('yolov5','custom', 'CharsYolo.*'/'plateYolo.*')` | `engines.py:137` |
| **ONNX Runtime / OpenVINO** | 1.24.1 / 2026.2.1 | `model/*.onnx` / `*_openvino_model` loading via `chechOnnx`/`chechOpenvino` | `engines.py:97`, `engines.py:134` |
| **NumPy** | 2.4.6 | Frame arrays `(H,W,3) uint8`, masks, `argsort`, angle math | all vision code |
| **Pillow** | 12.2.0 | JPEG encode fallback, `PIL` images (transitive) | via `cv2.imencode` path, `ultralytics` |

### 8.5 Networking & Streaming

| Technology | Used for | File |
|---|---|---|
| **RTSP over TCP** `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` | Camera ingest `FreshestFrame(source)` | `camera.py:11`, `engines.py:579` |
| **MJPEG multipart/x-mixed-replace; boundary=frame** | `StreamingResponse(cam.sendFrames())` live view | `api.py:126`, `engines.py:545`, `api.py:156` |
| **SSE text/event-stream** | ONVIF scan `discover_onvif_stream()` + PocketBase realtime `GET /api/realtime` | `api.py:235`, `engines.py:197` |
| **WebSocket (bottle-websocket/gevent-websocket)** | Transitive via `Eel`/`bottle` legacy | `requirements.txt:8,24` |
| **Socket probe** `connect_ex((host,554), timeout 3)` | `is_connection_alive()` RTSP liveness | `engines.py:692` |
| **ONVIF + WSDiscovery + zeep** `onvif_zeep 0.2.12 / WSDiscovery 2.1.2 / zeep 4.3.3 / lxml 6.1.1` | `get_rtsp_url()` via `ONVIFCamera` + `media_service.GetStreamUri` | `onvifmaneger.py:1` |
| **NrcDevice (nrcpy 1.2.3)** | IP relay `relayContact(300)` on TCP 23 | `api.py:33`, `api.py:190` |
| **SMTP (smtplib, email.mime)** | Gmail TLS 587 `emailHandler()` | `engines.py:1010` |

### 8.6 Data & Storage

| Technology | Purpose | File |
|---|---|---|
| **PocketBase** `pocketbase.exe serve --http=0.0.0.0:8090` (Go binary, `creationflags=CREATE_NO_WINDOW`) | `setting` + `database` collections, realtime subscriptions `setting/*` | `engines.py:80`, `engines.py:197`, `db_entries_utils.py:84` |
| **Requests** 2.34.2 + `urllib3` 2.7.0 | PocketBase REST `GET/POST /api/collections/...`, timeout/retry | `engines.py:179`, `db_entries_utils.py:68` |
| **SQLite** (embedded in PocketBase `pb_data/`) | Persistence (not directly in Python) | — |
| **JSON + hostname.json / regions.json** | Port + ROI polygons persisted to disk | `engines.py:973`, `api.py:274` |
| **OpenPyXL / pandas / polars implied** `dummy.xlsx` `csvReader.py` | CSV/XLSX import (legacy) | `csvReader.py`, `requirements.txt:53,57` |

### 8.7 Utilities & System

| Technology | Version | Purpose | File |
|---|---|---|---|
| **psutil** | 7.2.2 | `cpu_percent() / virtual_memory() / disk_usage()` `GET /system/utils` | `api.py:27`, `api.py:257` |
| **CORS Middleware** | Starlette | `allow_origins=["*"]` for Flutter web | `api.py:78` |
| **StaticFiles** | Starlette | `mount("/web/app", "build/web")` Flutter | `api.py:270` |
| **psutil / gc / webbrowser** | stdlib | `gc.collect()` + `torch.cuda.empty_cache()` shutdown, `webbrowser.open(http://127.0.0.1:{port}/web/app)` | `engines.py:320`, `engines.py:71` |
| **uuid** | stdlib | Per-connection token `uuid4().hex` | `engines.py:526` (`import uuid`) |
| **re / statistics / datetime / time / json / os / subprocess / webbrowser** | stdlib | Regex plate gate, `mean(confidences)`, timestamps, backoff | throughout |
| **multiprocessing.cpu_count()** | stdlib | `cv.setNumThreads(cpu_count)` `camera.py:14` | `camera.py:14` |

### 8.8 Packaging & Dev

| Technology | Version | Purpose |
|---|---|---|
| **PyInstaller / pyinstaller-hooks-contrib** | 6.21.0 / 2026.6 | `rtsp_streaming.spec` → exe |
| **auto-py-to-exe** | 2.50.1 | GUI for PyInstaller |
| **GitPython / gitdb / smmap** | 3.1.50 / 4.0.12 | Repo ops |
| **Matplotlib / seaborn / scipy / pandas** | 3.11 / 0.13 / 1.18 / 3.0 | Legacy training/plotting in `yolov5/` |
| **ONNX / onnxruntime-gpu / openvino-telemetry** | 1.22 / 1.24.1 / 2025.2 | Model export `yolov5/export.py` |
| **Eel / bottle / gevent / greenlet** | 0.18 / 0.13 / 26.5 / 3.5 | Legacy desktop wrapper `starter.py` (not in hot path) |
| **Flutter (build/web)** | prebuilt | Frontend served at `/web/app` |

### 8.9 Models (disk `engine/model/`)

| File pattern | Loader | Task |
|---|---|---|
| `yolov8n.{pt,onnx,_openvino_model}` | `ultralytics.YOLO` | car `classes=[2,5,7]` conf `carConf` |
| `plateYolo.{pt,onnx,_openvino_model}` + `dolditector.{pt,onnx,_openvino_model}` | `ultralytics.YOLO` | plate box `plateConf` / fast char `dolatiConf` |
| `CharsYolo.{pt,onnx,_openvino_model}` | `torch.hub YOLOv5` | fallback char OCR `charConf` |

Sentinel file `model/pt` or `model/onnx` selects extension; `*openvino*` forces CPU path `engines.py:97`.

