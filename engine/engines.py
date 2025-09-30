import asyncio
from asyncio import subprocess
import gc
import logging
import multiprocessing
import os
import platform
import time
from urllib.parse import urlparse
import webbrowser
from fastapi import Request
import numpy as np
import cv2
import requests
import torch
import subprocess
import statistics
from ultralytics import YOLO
from configParams import Parameters
from database.db_entries_utils import db_entries_time
from camera import FreshestFrame
import threading


# Logging configuration
logging.getLogger('torch').setLevel(logging.ERROR)
# warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)


cv2.setNumThreads(multiprocessing.cpu_count())


class CcTvMonitor:
    def __init__(self):
        self.process = None
        # self.loadDb()
        self.params = Parameters()
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
        self.RETRY_LIMIT = 5
        self.RETRY_LIMIT = 3
        self.lock = threading.Lock()
        self.model_car, self.model_plate, self.model_char = self.loadModels()
        self.quality, self.charConfidence, self.plateConfidence,self.port = self.loadConfig()[0:4]
        # self.loadWebBrowser(self.port)
        
    
    def loadWebBrowser(self,port):
        webbrowser.open(f'http://127.0.0.1:{port}/web/app')
        
        

    def loadDb(self):

        try:

            self.process = subprocess.Popen(
                ["pocketbase", "serve", "--http=0.0.0.0:8090"], creationflags=subprocess.CREATE_NO_WINDOW,)
            logging.info(f"PocketBase stater {self.process.pid}")
        except Exception as e:
            logging.info(e)

    def loadModels(self):

        logging.info("Loading YOLO models...")
        model_char = torch.hub.load(
            'yolov5', 'custom', 'model/CharsYolo.pt', source='local', device=self.device, force_reload=True)
        model_plate = torch.hub.load(
            'yolov5', 'custom', 'model/plateYolo.pt', source='local', device=self.device, force_reload=True)
        model_car = YOLO('model/yolo11n.pt', verbose=False).to(self.device)
        logging.info("Models loaded successfully")
        with self.lock:
            return model_car, model_plate, model_char

    def loadConfig(self):
        url = 'http://127.0.0.1:8090/api/collections/setting/records'
        response = requests.get(url).json()
        with self.lock:
            quality = response['items'][0]['quality']
            charConfidence = response['items'][0]['charConf']
            plateConfidence = response['items'][0]['plateConf']
            port = response['items'][0]['port']
            return quality, charConfidence, plateConfidence, port

    def graceful_shutdown(self):

        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Stop observer if running
        # os.system(f'taskkill /PID {self.process.pid} /F')
        self.process.kill()
        self.process.wait(1)
        # if self.process.returncode is not None:
            
        #    self.process.kill()
        #    self.process.wait(1)
        # # self.process.terminate()

        logging.info("Cleanup complete. Shutting down.")

        # Use os._exit instead of sys.exit for more forceful termination
        os._exit(0)

    def correct_perspective(self, image, scale_factor):
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
            logging.error(f"Error in correct_perspective: {e}")
            return image, (0, 0, 0, 0)

    # Character Detection
    def detect_plate_chars(self, cropped_plate):
        chars, confidences, char_detected = [], [], []
        results = self.model_char(cropped_plate)
        # Sort by x-coordinate
        detections = sorted(results.pred[0], key=lambda x: x[0])
        for det in detections:
            conf = det[4]

            if conf > 0.5:
                cls = int(det[5].item())
                char = self.params.char_id_dict.get(str(cls), '')
                chars.append(char)
                confidences.append(conf.item())
                char_detected.append(det.tolist())
        char_conf_avg = round(statistics.mean(confidences)
                              * 100) if confidences else 0
        return ''.join(chars), char_conf_avg

    async def process_frame(self, frame, path):

        try:
            # Create a lower-resolution copy for detection
            # lowres_for_detection=frame
            lowres_for_detection = frame.copy()
            scale_x = frame.shape[1] / lowres_for_detection.shape[1]
            scale_y = frame.shape[0] / lowres_for_detection.shape[0]

            # Detect vehicles in low-res
            car_res = self.model_car(
                lowres_for_detection, device=self.device, classes=[2, 5, 7])

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

                    plate_results = self.model_plate(
                        cropped_car).pandas().xyxy[0]

                    if not plate_results.empty:
                        for _, plate in plate_results.iterrows():
                            plate_conf = int(plate['confidence'] * 100)
                            if plate_conf >= int(self.plateConfidence*100):
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

                                plate_text, char_conf_avg = self.detect_plate_chars(
                                    cropped_plate)

                                cv2.rectangle(
                                    cropped_car, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                                plate_text = plate_text.replace('Taxi', 'x')
                                confidence = float(self.charConfidence) * 100

                                if char_conf_avg >= confidence and len(plate_text) >= 8:
                                    cv2.putText(cropped_car, f"Plate: {plate_text}", (x_min, y_min - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
                                    await db_entries_time(
                                        number=plate_text,
                                        charConfAvg=char_conf_avg,
                                        plateConfAvg=plate_conf,
                                        croppedPlate=cropped_plate,
                                        status="Active",
                                        frame=frame,
                                        isarvand='notarvand',
                                        rtpath=path,
                                        quality=self.quality
                                    )
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

                                    # Safety check before cropping
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
                                                await db_entries_time(
                                                    number=plate_text_arvnad,
                                                    charConfAvg=char_conf_arvnad,
                                                    plateConfAvg=plate_conf,
                                                    croppedPlate=cropped_plate,
                                                    status="Active",
                                                    frame=frame,
                                                    isarvand='arvand',
                                                    rtpath=path,
                                                    quality=self.quality
                                                )

            try:
                del plate_results, cropped_car, car_res
            except Exception as e:
                pass
            return frame

        except Exception as e:
            return frame

    def realseFreshest(self, fresh: FreshestFrame, cap: cv2.VideoCapture):
        try:
            fresh.release()
            cap.release()
        except Exception as e:
            logging.error(f"Error to Realse Cameras : {e}")

    async def generate_frames(self, camera_idx, source, request: Request):
        if not self.isConnectionAlive(source):
            return
        """Generate frames from a specific camera feed"""

        check_interval = 60  # seconds
        last_check = 0

        def open_capture(source):
            cap = cv2.VideoCapture(source)
            return cap if cap.isOpened() else None

        retries = 0
        tryconnection = 0
        cap = open_capture(source)

        while cap is None and retries < self.RETRY_LIMIT:
            print(
                f"[Camera {camera_idx}] Failed to open source. Retrying ({retries + 1}/{self.RETRY_LIMIT})...")
            await asyncio.sleep(self.RETRY_DELAY)
            retries += 1
            cap = open_capture(source)

        if cap is None:
            print(
                f"[Camera {camera_idx}] Could not open source after {self.RETRY_LIMIT} retries.")
            return
        fresh = FreshestFrame(cap)

        try:
            while fresh.is_alive():
                now = time.time()
                if now - last_check >= check_interval:

                    if not self.isConnectionAlive(source):
                        break
                    last_check = now

                if await request.is_disconnected():
                    print("Client disconnected, releasing camera.")
                    self.realseFreshest(fresh, cap)
                    break
                success, frame = fresh.read()
                if not success:

                    # Generate blank frame if we can't read from camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No signal", (220, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    frame = await self.process_frame(frame, f'/rt{camera_idx}')

                # Encode and yield the frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.info(e)
            self.graceful_shutdown()
        finally:

            self.realseFreshest(fresh, cap)
            print("Camera released.")

    async def generate_rtsp(self, source, request: Request):
        if not self.isConnectionAlive(source):
            return
        """Generate frames from a specific camera feed"""

        def open_capture(source):
            cap = cv2.VideoCapture(source)
            return cap if cap.isOpened() else None

        retries = 0
        cap = open_capture(source)

        while cap is None and retries < self.RETRY_LIMIT:

            await asyncio.sleep(self.RETRY_DELAY)
            retries += 1
            cap = open_capture(source)

        if cap is None:

            return
        try:
            fresh = FreshestFrame(cap)
        except Exception:
            self.graceful_shutdown()

        try:
            while fresh.is_alive():
                if not self.isConnectionAlive(source):
                    break

                if await request.is_disconnected():
                    print("Client disconnected, releasing camera.")
                    self.realseFreshest(fresh, cap)
                    break
                success, frame = fresh.read()
                if not success:

                    # Generate blank frame if we can't read from camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No signal", (220, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Encode and yield the frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(e)
            self.graceful_shutdown()
        finally:
            self.realseFreshest(fresh, cap)
            print("Camera released.")

    def isConnectionAlive(self, source):
        ulr = urlparse(source).hostname
        param = "-n" if platform.system().lower() == "windows" else "-c"

        # Build the ping command
        command = ["ping", param, "1", ulr]

        try:
            # Execute the ping command
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=10)

            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def emailHandler(email, plateNumber, edate, etime):

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    # Gmail SMTP server settings
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    # Sender's email credentials
    SENDER_EMAIL = "amnafarin4@gmail.com"
    SENDER_PASSWORD = "vioz mxiw nedg rybh"

    # Recipient email
    RECIPIENT_EMAIL = email

    # Email content
    subject = f"{edate} شناسایی پلاک در تاریخ "
    body = f""" 
    پلاک:\n{plateNumber}
    تاریخ:\n{edate}
    زمان:\n{etime}
     """

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Upgrade the connection to secure
        server.login(SENDER_EMAIL, SENDER_PASSWORD)  # Log in to the server

        # Send the email
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")

    finally:
        server.quit()  # Close the connection
