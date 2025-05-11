import cv2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import uvicorn
from afterengine import process_frame

app = FastAPI()

def generate_frames():
    cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.89:554/mainstream')  # Or your RTSP stream
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame=process_frame(frame,'rt1')
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    print("UPDATE 4132025")
    host:str='0.0.0.0'
    uvicorn.run("lets:app", log_level="info")