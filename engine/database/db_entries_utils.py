
import logging
import os
from typing import NamedTuple
import cv2
import datetime
import time
import requests

from configParams import Parameters
from PIL import Image


params = Parameters()
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


# def dbGetPlateLatestEntry(plateNumber):
#     params = Parameters()
#     base_url = f"http://127.0.0.1:8090/api/collections/database/records"

#     try:
#         params = {
#             'filter': f"plateNum='{plateNumber}'",
#             'sort': '-eDate',
#             'perPage': 1
#         }

#         response = requests.get(
#             url=base_url,
#             params=params,
#             timeout=10
#         )
#         response.raise_for_status()

#         data = response.json()

#         if data.get('totalItems', 0) > 0:
#             # Map API response to Entries constructor parameters
#             item = data['items'][0]

#             # Create dictionary with required fields
#             FullData = {
#                 'platePercent': item['platePercent'],
#                 'charPercent': item['charPercent'],
#                 'eDate': item['eDate'],
#                 'eTime': item['eTime'],
#                 'plateNum': item['plateNum'],
#                 'status': item['status'],
#                 'imgpath': item['imgpath'],
#                 'scrnpath': item['scrnPath'],  # Note case difference
#                 'isarvand': item['isarvand'],
#                 'rtpath': item['rtpath']
#             }

#             return Entries(**FullData)

#         return None

#     except Exception as e:
#         print(f"API request failed: {str(e)}")
#         return None
#     except KeyError as e:
#         print(f"Missing expected field in response: {str(e)}")
#         return None


# def insterToPocket(plateImgName2, screenshot_path, number, display_date, display_time, status, isarvand, rtpath, charConfAvg, plateConfAvg):
#     POCKETBASE_URL = f"http://127.0.0.1:8090"
#     COLLECTION_NAME = "database"
#     url = f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records"

#     with open(plateImgName2, "rb") as file1, open(screenshot_path, "rb") as file2:
#         files = {
#             # Change field name if needed
#             "scrnPath": (screenshot_path, file2, "image/jpeg"),
#             # Change field name if needed
#             "imgpath": (plateImgName2, file1, "image/jpeg"),
#         }

#         response = requests.post(url, files=files, data={
#             "plateNum": number,
#             "eDate": display_date,
#             "eTime": display_time,
#             "status": status,
#             "isarvand": isarvand,
#             "rtpath": rtpath,
#             "charPercent": charConfAvg,
#             "platePercent": plateConfAvg,
#         })

#     # Check response
#     if response.status_code in [200, 201]:

#         return response.json()['id']
#     else:
#         print("Error:", response.text)


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


class RecentEntry(NamedTuple):
    platenum: str

    time: datetime.datetime


recent_names: list[RecentEntry] = []
TIME_THRESHOLD = 10


def clean_old_entries():
    now = datetime.datetime.now()
    recent_names[:] = [
        entry for entry in recent_names
        if (now - entry.time).total_seconds() < TIME_THRESHOLD
    ]


def should_insert(name):
    now = datetime.datetime.now()
    clean_old_entries()

    # for entry in recent_names:
    #     if name == "unknown" and entry.platenum == "unknown":
    #         if (now - entry.time).total_seconds() < TIME_THRESHOLD:
    #             return False
    for entry in recent_names:
        if entry.platenum == name:
            if (now - entry.time).total_seconds() < TIME_THRESHOLD:
                return False

    return True


def db_entries_time(number, charConfAvg, plateConfAvg, croppedPlate, status, frame, isarvand, rtpath,quality):
    url = f"http://127.0.0.1:8090/api/collections/database/records"


    timeNow = datetime.datetime.now()
    display_time = timeNow.strftime("%H:%M:%S")
    display_date = timeNow.strftime("%Y-%m-%d")

    if not os.path.exists('output'):
        os.makedirs('output')
        os.makedirs('output/cropedplate')
        os.makedirs('output/screenshot')
    else:
        pass
    if should_insert(number):
        frame_loc, crop_loc = savePicture(
            frame, croppedPlate, number, quality)
        recent_names.append(RecentEntry(
            number=number, time=datetime.datetime.now()))
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
        })
        if response.status_code in [200, 201]:
            os.remove(crop_loc)
            os.remove(frame_loc)
            logging.info(response.json()['id'])
        else:
            logging.error("Error:", response.text)

    # result = dbGetPlateLatestEntry(number)
    # if number != '':
    #     if result is not None:
    #         strTime = result.getTime()
    #         strDate = result.getDate()
    #         timediff = timeDifference(strTime, strDate, False)

    #     else:
    #         strTime = time.strftime("%H:%M:%S")
    #         strDate = time.strftime("%Y-%m-%d")
    #         timediff = timeDifference(strTime, strDate, True)

    #     if timediff:
    #         display_time = timeNow.strftime("%H:%M:%S")
    #         display_date = timeNow.strftime("%Y-%m-%d")
    #         screenshot_path = f"output/screenshot/{number}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    #     # Save the full frame as a screenshot if `frame` is provided
    #         if frame is not None:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             frame = Image.fromarray(frame)
    #             frame.save(screenshot_path, "JPEG",
    #                        quality=quality, optimize=True)
    #         #    cv2.imwrite(screenshot_path, frame)

    #         plateImgName2 = f'output/cropedplate/{number}_{datetime.datetime.now().strftime("%m-%d")}.jpg'
    #         cv2.imwrite(plateImgName2, croppedPlate)
    #         insterToPocket(status=status, rtpath=rtpath, plateImgName2=plateImgName2, screenshot_path=screenshot_path, charConfAvg=charConfAvg,
    #                        display_date=display_date, display_time=display_time, isarvand=isarvand, number=number, plateConfAvg=plateConfAvg)


# def timeDifference(strTime, strDate, isnone):
#     # Uncomment the following if you want to calculate the actual time difference
#     start_time = datetime.datetime.strptime(
#         strTime + ' ' + strDate, "%H:%M:%S %Y-%m-%d")
#     end_time = datetime.datetime.strptime(
#         datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"), "%H:%M:%S %Y-%m-%d")
#     delta = end_time - start_time
#     sec = delta.total_seconds()
#     if isnone:
#         min = 2
#     else:
#         min = (sec / 60).__ceil__()

#     # min = 2  # Set to 2 for testing purposes

#     if min > 1:
#         return True
#     else:
#         return False
