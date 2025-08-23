
from contextlib import asynccontextmanager
import json
import os
import socket
import time
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from engines import CcTvMonitor, emailHandler
from TcpConnector import TcpConnector
from onvifmaneger import get_rtsp_url
import uvicorn
import webbrowser


cctv = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cctv,port
    cctv=CcTvMonitor()
    
    yield
    cctv.graceful_shutdown()


app = FastAPI(lifespan=lifespan)
connection = TcpConnector()


class Relay(BaseModel):
    isconnect: bool


class EmailClass(BaseModel):
    plateNumber: str
    eDate: str
    eTime: str


class RtspFields(BaseModel):
    ip: str
    port: str
    username: str
    password: str


origins = ["*"]  # Change this to specific domains in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH",
                   "DELETE"],  # Allowed HTTP methods
    allow_headers=["Origin", "X-Requested-With",
                   "Content-Type", "Accept"],  # Allowed headers
)


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str, request: Request, source: str = Query(...)):

    if source == '0':
        source = int(source)
    """Stream video from a specific camera"""

    try:
        # Extract camera index from ID (rt1 -> 1)
        camera_idx = int(camera_id[2:])

        return StreamingResponse(

            cctv.generate_frames(camera_idx, source, request),


            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store"
            }
        )
    except ValueError:
        return Response(f"Invalid camera ID: {camera_id}. Use format: rt1, rt2, etc.",
                        status_code=400)


@app.get("/rtsp_feed/{camera_id}")
async def video_feed(camera_id: str, request: Request, source: str = Query(...)):
    if source == '0':
        source = int(source)
    """Stream video from a specific camera"""

    return StreamingResponse(

        cctv.generate_rtsp(source, request),


        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store"
        }
    )


@app.post("/iprelay")
def connectrelay(request: Relay, ip, port):

    connection.setConnectionProperties(f"{ip}", int(port))
    if (request.isconnect):
        # on
        if (connection.connectToServer()):

            return {"massage": "connect"}
        else:

            return {"massage": "problem connect"}
    else:
        if (connection.closeConnection()):
            return {"massage": "disconnect"}
        else:
            return {"massage": "problem dissconnect"}


@app.get("/iprelay")
def onOff(onOff, relay):
    # on
    if (onOff == "true"):
        if (int(relay) == 1):
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x01\x01\x02'  # relay 1
            connection.sendPacket(bData=data)
            # if (connection.sendPacket(bData=data)):
            #     return {'massage': connection.receivePacket(23, 2)}
            # else:
            #     return {"massage":f"problem : {connection.receivePacket(23, 2)}"}
        else:
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x02\x01\x02'  # relay 2
            connection.sendPacket(bData=data)
            # if ():
            #     return {'massage': connection.receivePacket(23, 2)}
            # else:
            #     return {"massage":f"problem : {connection.receivePacket(23, 2)}"}
            ########################
            # of
    else:
        if (int(relay) == 1):
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x01\x00\x00'  # relay 1
            if (connection.sendPacket(bData=data)):
                return {'massage': connection.receivePacket(23, 2)}
            else:
                return {"massage": f"problem : {connection.receivePacket(23, 2)}"}
        else:
            data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00\x01\x01\x01\x00\x00\x00\x00\x00\x03\x02\x00\x00'  # relay 2
            if (connection.sendPacket(bData=data)):
                return {'massage': connection.receivePacket(23, 2)}
            else:
                return {"massage": f"problem : {connection.receivePacket(23, 2)}"}

    # vioz mxiw nedg rybh


@app.post('/email')
def sendEmail(request: EmailClass, email):
    try:

        emailHandler(email, request.plateNumber, request.eDate, request.eTime)
        return {"massage": "email send Apporved"}
    except:
        return {"massage": "Failed"}


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
def get_camera_stream():
    return StreamingResponse(discover_onvif_stream(), media_type="text/event-stream")


@app.post('/onvif/get-rtsp')
async def get_camra_rtsp(request: RtspFields):
    print(request.ip)
    request.port = int(request.port)
    rtspUrl = get_rtsp_url(request.ip, request.port,
                           request.username, request.password)
    return {'rtsp': rtspUrl}


app.mount("/web/app", StaticFiles(directory="build/web",
          html=True), name="flutter")


if __name__ == "__main__":
    host = '0.0.0.0'
    port=8000
    webbrowser.open(f'http://127.0.0.1:{port}/web/app')
    uvicorn.run("api:app", log_level='info', log_config=None,
                reload=False, port=port, host=host)
    # if KeyboardInterrupt:
    #     graceful_shutdown()
    # DO BUT DONT FORGER PORT
