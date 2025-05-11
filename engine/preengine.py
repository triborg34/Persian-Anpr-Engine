import os
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from configParams import Parameters
from lets import generate_frames, graceful_shutdown
import uvicorn
params = Parameters()
port = int(params.socketport)
host = '0.0.0.0'


app = FastAPI()





@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str):
    """Stream video from a specific camera"""
    if not camera_id.startswith("rt"):
        return Response("Invalid camera ID format. Use rt1, rt2, etc.", status_code=400)

    try:
        # Extract camera index from ID (rt1 -> 1)
        camera_idx = int(camera_id[2:])

        # Check if camera index is valid
        if camera_idx < 1 or camera_idx > len(params.rtps):
            return Response(f"Camera {camera_id} not found. Valid range: rt1-rt{len(params.rtps)}",
                            status_code=404)

        return StreamingResponse(
            generate_frames(camera_idx),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except ValueError:
        return Response(f"Invalid camera ID: {camera_id}. Use format: rt1, rt2, etc.",
                        status_code=400)

if __name__ =="__main__":
    try:
        uvicorn.run("preengine:app",)
    except KeyboardInterrupt:
        graceful_shutdown()
        os._exit(0)
        