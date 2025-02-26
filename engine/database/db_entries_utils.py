
import cv2
import datetime
import time
from requests.exceptions import RequestException
import requests

from configParams import Parameters
from database.classEntries import Entries
from helper.text_decorators import check_similarity_threshold
from PIL import Image

# def create_database_if_not_exists(db_path):
#     # Check if the database file already exists
#     if not os.path.exists(db_path):
#         # Create a new database and define the tables
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # Create `entry` table for `insterMyentry` function
#         cursor.execute('''CREATE TABLE IF NOT EXISTS entry (
#                             platePercent REAL,
#                             charPercent REAL,
#                             eDate TEXT,
#                             eTime TEXT,
#                             plateNum TEXT,
#                             status TEXT,
#                             imgpath TEXT,
#                             scrnpath TEXT,
#                           	isarvand  TEXT,
#                             rtpath TEXT


#                           )''')

#         # Create `entries` table for `insertEntries` function
#         # cursor.execute('''CREATE TABLE IF NOT EXISTS entries (
#         #                     platePercent REAL,
#         #                     charPercent REAL,
#         #                     eDate TEXT,
#         #                     eTime TEXT,
#         #                     plateNum TEXT PRIMARY KEY,
#         #                     status TEXT
#         #                   )''')

#         # Commit changes and close connection
#         conn.commit()
#         conn.close()
#         print(f"Database created at {db_path} with tables `entry` and `entries`.")
#     else:
#         print(f"Database already exists at {db_path}.")




params = Parameters()


def getQuality() -> int:
    response=requests.get( f"http://{params.defip}:{params.defport}/api/collections/setting/records")
    
    quality=response.json()['items'][len(response.json()['items'])-1]['quality']
    
    return int(quality)




# create_database_if_not_exists(db_path=db_path)


# def insterMyentry(platePercent, charPercent, eDate, eTime, plateNum, status, imagePath,scrnpath,isarvand,rtpath):
#     sqlconnect = sqlite3.connect('./database/entrieses.db')
#     sqlcuurser = sqlconnect.cursor()
#     excute = 'INSERT INTO entry VALUES (:platePercent, :charPercent, :eDate, :eTime, :plateNum, :status , :imgpath,:scrnpath,:isarvand,:rtpath)'
#     sqlcuurser.execute(excute, (platePercent, charPercent, eDate, eTime, plateNum, status, imagePath,scrnpath,isarvand,rtpath))

#     sqlconnect.commit()
#     sqlcuurser.close()

# def insertEntries(entry):
#     sqlConnect = sqlite3.connect(dbEntries)
#     sqlCursor = sqlConnect.cursor()

#     sqlCursor.execute(
#         "INSERT OR IGNORE INTO entries VALUES (:platePercent, :charPercent, :eDate, :eTime, :plateNum, :status)",
#         vars(entry))

#     sqlConnect.commit()
#     sqlConnect.close()

def dbGetPlateLatestEntry(plateNumber):
    params = Parameters()
    base_url = f"http://{params.defip}:{params.defport}/api/collections/database/records"
    
    try:
        params = {
            'filter': f"plateNum='{plateNumber}'",
            'sort': '-eDate',
            'perPage': 1
        }

        response = requests.get(
            url=base_url,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('totalItems', 0) > 0:
            # Map API response to Entries constructor parameters
            item = data['items'][0]
            
            # Create dictionary with required fields
            FullData = {
                'platePercent': item['platePercent'],
                'charPercent': item['charPercent'],
                'eDate': item['eDate'],
                'eTime': item['eTime'],
                'plateNum': item['plateNum'],
                'status': item['status'],
                'imgpath': item['imgpath'],
                'scrnpath': item['scrnPath'],  # Note case difference
                'isarvand': item['isarvand'],
                'rtpath': item['rtpath']
            }
          
            return Entries(**FullData)
            
        return None
        
    except RequestException as e:
        print(f"API request failed: {str(e)}")
        return None
    except KeyError as e:
        print(f"Missing expected field in response: {str(e)}")
        return None



def insterToPocket(plateImgName2, screenshot_path,number,display_date,display_time,status,isarvand,rtpath,charConfAvg,plateConfAvg):
    params = Parameters()
    POCKETBASE_URL = f"http://{params.defip}:{params.defport}"
    COLLECTION_NAME = "database"
    url = f"{POCKETBASE_URL}/api/collections/{COLLECTION_NAME}/records"

    with open(plateImgName2, "rb") as file1, open(screenshot_path, "rb") as file2:
        files = {
            "scrnPath": (screenshot_path, file2, "image/jpeg"),  # Change field name if needed
            "imgpath": (plateImgName2, file1, "image/jpeg"),      # Change field name if needed
        }
        
        response = requests.post(url, files=files,data={
              "plateNum": number,
  "eDate": display_date,
  "eTime": display_time,
  "status": status,
  "isarvand": isarvand,
  "rtpath": rtpath,
  "charPercent": charConfAvg,
  "platePercent": plateConfAvg,
        })

    # Check response
    if response.status_code in [200, 201]:
        
        return response.json()['id']
    else:
        print("Error:", response.text)
        
    
similarityTemp = ''
def db_entries_time(number, charConfAvg, plateConfAvg, croppedPlate, status, frame,isarvand,rtpath) :
    
    
    
    global similarityTemp
    isSimilar = check_similarity_threshold(similarityTemp, number)
    quality=getQuality()
    quality = 10 if quality<10 else quality
    
    # Only proceed if the plate number is unique
    if not isSimilar:
        similarityTemp = number
        timeNow = datetime.datetime.now()
        
        
      

        # Database operations for plate detection 
        result = dbGetPlateLatestEntry(number)
        if number != '':
            if result is not None:
                strTime = result.getTime()
                strDate = result.getDate()
                timediff=timeDifference(strTime, strDate,False)
                
            else:
                strTime = time.strftime("%H:%M:%S")
                strDate = time.strftime("%Y-%m-%d")
                timediff=timeDifference(strTime, strDate,True)
                
                
                
            
            if timediff:
                display_time = timeNow.strftime("%H:%M:%S")
                display_date = timeNow.strftime("%Y-%m-%d")
                screenshot_path = f"output/screenshot/{number}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            # Save the full frame as a screenshot if `frame` is provided
                if frame is not None:
                   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                   frame=Image.fromarray(frame) 
                   frame.save(screenshot_path, "JPEG", quality=quality, optimize=True)
                #    cv2.imwrite(screenshot_path, frame)
                   print(f"Screenshot saved to {screenshot_path} for plate {number} with character confidence {charConfAvg}%.")


                plateImgName2 = f'output/cropedplate/{number}_{datetime.datetime.now().strftime("%m-%d")}.jpg'
                cv2.imwrite(plateImgName2, croppedPlate)
                insterToPocket(status=status,rtpath=rtpath,plateImgName2=plateImgName2,screenshot_path=screenshot_path,charConfAvg=charConfAvg,display_date=display_date,display_time=display_time,isarvand=isarvand,number=number,plateConfAvg=plateConfAvg)
                
                

#                 sendData=requests.patch(f'http://127.0.0.1:8090/api/collections/database/records/{id}',data={

#   "plateNum": number,
#   "eDate": display_date,
#   "eTime": display_time,
#   "status": status,
#   "isarvand": isarvand,
#   "rtpath": rtpath,
#   "charPercent": charConfAvg,
#   "platePercent": plateConfAvg,

#                 })
#                 print(sendData.json())
                
                # insterMyentry(plateConfAvg, charConfAvg, display_date, display_time, number, status, plateImgName2,screenshot_path,isarvand,rtpath)
                # insertEntries(entries)
        # else:
        #     if number != '':
        #         print("Here ")
        #         display_time = time.strftime("%H:%M:%S")
        #         display_date = time.strftime("%Y-%m-%d")
                
        #         screenshot_path = f"output/screenshot/{number}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        #         if frame is not None:
        #                  cv2.imwrite(screenshot_path, frame)
        #                  print(f"Screenshot saved to {screenshot_path} for plate {number} with character confidence {charConfAvg}%.")

        #         plateImgName2 = f'output/cropedplate/{number}_{datetime.datetime.now().strftime("%m-%d")}.jpg'
        #         cv2.imwrite(plateImgName2, croppedPlate)

          
        #         # insertEntries(entries)
        #         insterMyentry(plateConfAvg, charConfAvg, display_date, display_time, number, status, plateImgName2,screenshot_path,isarvand,rtpath)

def getFieldNames(fieldsList):
    fieldNamesOutput = []
    for value in fieldsList:
        fieldNamesOutput.append(params.fieldNames[value])
    return fieldNamesOutput

def timeDifference(strTime, strDate,isnone):
    # Uncomment the following if you want to calculate the actual time difference
    start_time = datetime.datetime.strptime(strTime + ' ' + strDate, "%H:%M:%S %Y-%m-%d")
    end_time = datetime.datetime.strptime(datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d"), "%H:%M:%S %Y-%m-%d")
    delta = end_time - start_time
    sec = delta.total_seconds()
    if isnone:
        min=2
    else:
        min = (sec / 60).__ceil__()

    # min = 2  # Set to 2 for testing purposes


    if min > 1:
        return True
    else:
        return False
