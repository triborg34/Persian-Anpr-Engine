
import logging
import os
import threading
import cv2
import datetime
import time
import requests
from PIL import Image


recent_plates = {}
recent_lock = threading.Lock()

TIME_THRESHOLD = 10  # seconds


logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)


def getQuality() -> int:
    response = requests.get(
        f"http://127.0.0.1:8090/api/collections/setting/records")

    quality = response.json()['items'][len(
        response.json()['items'])-1]['quality']

    return int(quality)


def reserve_plate(plate: str, rtpath: str) -> bool:
    """
    Reserve a plate for processing.

    Returns:
        True  -> process it
        False -> recently processed
    """

    now = time.time()

    with recent_lock:

        # Cleanup expired entries
        expired = [
            key
            for key, ts in recent_plates.items()
            if now - ts > TIME_THRESHOLD
        ]

        for key in expired:
            del recent_plates[key]

        key = f"{rtpath}_{plate}"

        # Already processed recently
        if key in recent_plates:
            return False

        # Reserve it
        recent_plates[key] = now

        return True


def release_plate(plate: str, rtpath: str):
    """
    Release reservation when upload fails.
    """

    key = f"{rtpath}_{plate}"

    with recent_lock:
        recent_plates.pop(key, None)


def savePicture(frame, croppedPlate, number, quality):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame_loc = f'output/screenshot/s.{number}.jpg'
    frame.save(
        f'{frame_loc}', "JPEG", quality=quality, optimize=True)
    # cropp
    croppedPlate = cv2.cvtColor(croppedPlate, cv2.COLOR_BGR2RGB)
    croppedPlate = Image.fromarray(croppedPlate)
    crop_loc = f'output/cropedplate/c.{number}.jpg'
    croppedPlate.save(
        f'{crop_loc}', "JPEG", quality=100, optimize=True)

    return frame_loc, crop_loc


# class RecentEntry(NamedTuple):
#     platenum: str

#     time: datetime.datetime


# recent_names: list[RecentEntry] = []
# TIME_THRESHOLD = 10


# def clean_old_entries():
#     now = datetime.datetime.now()
#     recent_names[:] = [
#         entry for entry in recent_names
#         if (now - entry.time).total_seconds() < TIME_THRESHOLD
#     ]


# def should_insert(name):
#     now = datetime.datetime.now()
#     clean_old_entries()

#     # for entry in recent_names:
#     #     if name == "unknown" and entry.platenum == "unknown":
#     #         if (now - entry.time).total_seconds() < TIME_THRESHOLD:
#     #             return False
#     for entry in recent_names:
#         if entry.platenum == name:
#             if (now - entry.time).total_seconds() < TIME_THRESHOLD:
#                 return False

#     return True


def db_entries_time(number, charConfAvg, plateConfAvg, croppedPlate, status, frame, isarvand, rtpath, quality):
    url = "http://127.0.0.1:8090/api/collections/database/records"

    timeNow = datetime.datetime.now()
    display_time = timeNow.strftime("%H:%M:%S")
    display_date = timeNow.strftime("%Y-%m-%d")

    
  
    os.makedirs('output/cropedplate',exist_ok=True)
    os.makedirs('output/screenshot',exist_ok=True)
    if not reserve_plate(number, rtpath):
        return
    frame_loc = None
    crop_loc = None
    try:

        frame_loc, crop_loc = savePicture(
            frame, croppedPlate, number, quality)

        with open(crop_loc, "rb") as file1, open(frame_loc, "rb") as file2:

            files = {
                # Change field name if needed
                "scrnPath": (frame_loc, file2, "image/jpeg"),
                # Change field name if needed
                "imgpath": (crop_loc, file1, "image/jpeg"),
            }

            response = requests.post(url, files=files, data={
                "plateNum": number,
                "eDate": display_date,
                "eTime": display_time,
                "status": status,
                "isarvand": isarvand,
                "rtpath": rtpath,
                "charPercent": charConfAvg,
                "platePercent": plateConfAvg,
            }, timeout=15)

            if response.status_code in [200, 201]:

                logging.info(response.json()['id'])
            else:
                logging.error(
                    f"PocketBase error {response.status_code}: "
                    f"{response.text}"
                )
                release_plate(number, rtpath)
            # os.remove(frame_loc)
            # os.remove(crop_loc)

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
