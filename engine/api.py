
import os
os.environ['ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS'] = '1'
from dotenv import load_dotenv
load_dotenv()
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import socket
import threading
import time
import cv2
from fastapi import FastAPI, Query, Request, Security, HTTPException
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import psutil
from pydantic import BaseModel
from engines import CcTvMonitor, emailHandler, CameraManager
from camera import FreshestFrame
from onvifmaneger import get_rtsp_url
import uvicorn
from nrcpy import NrcDevice

API_KEY = os.environ.get("ANPR_API_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

cctv = None
camera_registry = {}
camera_registry_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cctv
    cctv = CcTvMonitor()

    yield
    updatePort()
    cctv.graceful_shutdown()


app = FastAPI(lifespan=lifespan)


class Relay(BaseModel):
    ip: str
    port: str
    username: str
    password: str
    relay_number: str


class EmailClass(BaseModel):
    plateNumber: str
    eDate: str
    eTime: str


class RtspFields(BaseModel):
    ip: str
    port: str
    username: str
    password: str


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH",
                   "DELETE"],
    allow_headers=["Origin", "X-Requested-With",
                   "Content-Type", "Accept"],
)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not API_KEY:
        return True
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True


@app.get("/video_feed/{camera_id}")
async def video_feed(
    camera_id: str,
    request: Request,
    source: str = Query(...),
    _: bool = Security(verify_api_key),
):

    cctv.carConf = 0.1
    cctv.iou = 0.5
    logging.info("NORMAL MODE")
    if source == "0":
        source = int(source)
    camera_idx = int(camera_id[2:])
    with camera_registry_lock:
        if source not in camera_registry:
            camera_registry[source] = CameraManager(source, cctv, camera_idx)

        cam = camera_registry[source]

    cam.add_client()

    async def watch_disconnect():
        while True:
            if await request.is_disconnected():
                cam.remove_client()
                break
            await asyncio.sleep(0.2)

    asyncio.create_task(watch_disconnect())

    return StreamingResponse(
        cam.sendFrames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/rtsp_feed/{camera_id}")
async def rtsp_feed(camera_id: str, request: Request, source: str = Query(...), _: bool = Security(verify_api_key)):
    if source == '0':
        source = int(source)

    return StreamingResponse(
        generate_rtsp(source, request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store"
        }
    )


async def generate_rtsp(source, request):
    cam = FreshestFrame(source) if isinstance(source, str) else FreshestFrame(str(source))
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
    try:
        while True:
            if await request.is_disconnected():
                break
            success, frame = cam.read()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            _, jpeg = cv2.imencode(".jpg", frame, jpeg_params)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
    finally:
        cam.release()


@app.post("/utils/iprelay")
def onOff(data: Relay, _: bool = Security(verify_api_key)):
    ip = data.ip.strip()
    port = int(data.port.strip())
    username = data.username.strip()
    password = data.password.strip()
    relay_number = int(data.relay_number.strip())

    handle_relay_operations(ip, port, username, password, relay_number)
    return {"message": "relay operation completed"}


def handle_relay_operations(ip='192.168.1.200', port=23, username='admin', password='admin', relay_number=1):
    """Handle IP relay operations - single execution"""
    try:
        logging.info(f"Executing relay operation for {ip}")
        nrc = NrcDevice((ip, port, username, password))

        nrc.connect()
        if nrc.login():
            nrc.relayContact(relay_number, 300)
            logging.info(f"Relay operation completed for {ip}")
        nrc.disconnect()
    except Exception as e:
        logging.error(f"Relay error for {ip}: {e}")


@app.post('/email')
def sendEmail(request: EmailClass, email: str, _: bool = Security(verify_api_key)):
    try:
        emailHandler(email, request.plateNumber, request.eDate, request.eTime)
        return {"message": "email send approved"}
    except Exception as e:
        logging.error(f"Email error: {e}")
        return {"message": "failed"}


def discover_onvif_stream():
    try:

        ip_base = socket.gethostbyname(socket.gethostname())
        ip_base = ip_base.split('.')
        ip_base = '.'.join(ip_base[0:3])
    except Exception:
        ip_base = "192.168.1"

    def event_generator():
        for i in range(1, 255):
            ip = f"{ip_base}.{i}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.3)
                result = sock.connect_ex((ip, 80))
                if result == 0:
                    yield f"data: {json.dumps({'ip': ip, 'port': 80})}\n\n"
                sock.close()
            except:
                continue
            time.sleep(0.1)
    return event_generator()


@app.get("/onvif/get-stream")
async def get_camera_stream(_: bool = Security(verify_api_key)):
    return StreamingResponse(discover_onvif_stream(), media_type="text/event-stream")


@app.post('/onvif/get-rtsp')
async def get_camra_rtsp(request: RtspFields, _: bool = Security(verify_api_key)):
    logging.info(f"ONVIF RTSP request for {request.ip}")
    request.port = int(request.port)
    rtspUrl = get_rtsp_url(request.ip, request.port,
                           request.username, request.password)
    return {'rtsp': rtspUrl}


@app.get('/system/utils')
async def get_system_health(_: bool = Security(verify_api_key)):
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent,
    }
    



app.mount("/web/app", StaticFiles(directory="build/web",
          html=True), name="flutter")


def updatePort():
    port = cctv.loadConfig()[3]
    with open('hostname.json', 'w') as file:
        json.dump({'port': port}, file, indent=4)
    return 0


def readPort():
    with open('hostname.json', 'r') as file:
        data = json.load(file)
        return data['port']

if __name__ == "__main__":
    host = '0.0.0.0'
    port = int(readPort())

    uvicorn.run("api:app", log_level='info', log_config=None,
                reload=False, port=port, host=host)
