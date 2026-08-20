
import logging
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


def _encode_jpeg(img, quality: int):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
    ok, buf = cv2.imencode('.jpg', img, params)
    return buf if ok else None


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

    if not reserve_plate(number, rtpath):
        return

    try:
        frame_bytes = _encode_jpeg(frame, quality)
        crop_bytes = _encode_jpeg(croppedPlate, 100)
        if frame_bytes is None or crop_bytes is None:
            logging.error(f"Failed to encode plate images for {number}")
            release_plate(number, rtpath)
            return

        files = {
            "scrnPath": (f"s.{number}.jpg", frame_bytes.tobytes(), "image/jpeg"),
            "imgpath": (f"c.{number}.jpg", crop_bytes.tobytes(), "image/jpeg"),
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
