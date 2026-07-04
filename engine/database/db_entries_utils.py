
import logging
import os
import threading
import cv2
import datetime
import time
import requests
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=4)

recent_plates = {}
recent_lock = threading.Lock()

TIME_THRESHOLD = 10  # seconds

OUTPUT_DIRS_CREATED = False


logging.basicConfig(
    level=logging.DEBUG,

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def _ensure_output_dirs():
    global OUTPUT_DIRS_CREATED
    if not OUTPUT_DIRS_CREATED:
        os.makedirs('output/cropedplate', exist_ok=True)
        os.makedirs('output/screenshot', exist_ok=True)
        OUTPUT_DIRS_CREATED = True


def getQuality() -> int:
    try:
        response = requests.get(
            f"http://127.0.0.1:8090/api/collections/setting/records",
            timeout=5)
        response.raise_for_status()
        items = response.json().get('items', [])
        if items:
            return int(items[-1]['quality'])
    except requests.RequestException as e:
        logging.error(f"Failed to fetch quality setting: {e}")
    return 80


def reserve_plate(plate: str, rtpath: str) -> bool:
    now = time.time()

    with recent_lock:
        expired = [
            key
            for key, ts in recent_plates.items()
            if now - ts > TIME_THRESHOLD
        ]

        for key in expired:
            del recent_plates[key]

        key = f"{rtpath}_{plate}"

        if key in recent_plates:
            return False

        recent_plates[key] = now

        return True


def release_plate(plate: str, rtpath: str):
    key = f"{rtpath}_{plate}"

    with recent_lock:
        recent_plates.pop(key, None)


def savePicture(frame, croppedPlate, number, quality):

    frame_loc = f'output/screenshot/s.{number}.jpg'
    crop_loc = f'output/cropedplate/c.{number}.jpg'

    encode_params_frame = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
    encode_params_crop = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_OPTIMIZE, 0]

    cv2.imwrite(frame_loc, frame, encode_params_frame)
    cv2.imwrite(crop_loc, croppedPlate, encode_params_crop)

    return frame_loc, crop_loc


def _upload_to_pocketbase(url, files, data, number, rtpath):
    try:
        response = requests.post(url, files=files, data=data, timeout=15)
        if response.status_code in [200, 201]:
            logging.info(response.json().get('id', 'unknown'))
        else:
            logging.error(
                f"PocketBase error {response.status_code}: "
                f"{response.text}"
            )
            release_plate(number, rtpath)
    except Exception as e:
        logging.error(f"Failed to upload plate {number}: {e}")
        release_plate(number, rtpath)


def db_entries_time(number, charConfAvg, plateConfAvg, croppedPlate, status, frame, isarvand, rtpath, quality):
    url = "http://127.0.0.1:8090/api/collections/database/records"
    

    timeNow = datetime.datetime.now()
    display_time = timeNow.strftime("%H:%M:%S")
    display_date = timeNow.strftime("%Y-%m-%d")

    _ensure_output_dirs()
    if not reserve_plate(number, rtpath):
        return
    frame_loc = None
    crop_loc = None
    try:

        frame_loc, crop_loc = savePicture(
            frame, croppedPlate, number, quality)

        with open(crop_loc, "rb") as f1, open(frame_loc, "rb") as f2:
            crop_bytes = f1.read()
            frame_bytes = f2.read()

        files = {
            "scrnPath": (frame_loc, frame_bytes, "image/jpeg"),
            "imgpath": (crop_loc, crop_bytes, "image/jpeg"),
        }

        data = {
            "plateNum": number,
            "eDate": display_date,
            "eTime": display_time,
            "status": status,
            "isarvand": isarvand,
            "rtpath": rtpath,
            "charPercent": charConfAvg,
            "platePercent": plateConfAvg,
        }

        _executor.submit(_upload_to_pocketbase, url, files, data, number, rtpath)

    except Exception as e:
        logging.exception(
            f"Failed to save plate {number}"
        )
        release_plate(number, rtpath)

    finally:

        for path in (frame_loc, crop_loc):

            if path and os.path.exists(path):

                try:
                    os.remove(path)
                except Exception:
                    logging.exception(
                        f"Failed to delete temporary file {path}")
