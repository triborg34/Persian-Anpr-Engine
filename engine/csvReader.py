import pandas as pd
import datetime

import requests

alphabetP1 = {

            "A": "آ",
            "B": "ب",
            "D": "د",
            "Gh": "ق",
            "H": "ه",
            "J": "ج",
            "L": "ل",
            "M": "م",
            "N": "ن",
            "P": "پ",
            "PuV": "ع",
            "PwD": "ژ",
            "Sad": "ص",
            "Sin": "س",
            "T": "ط",
            "Taxi": "ت",
            "V": "و",
            "Y": "ی",
        }
alphabetP2 = {
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
    "آ": "A",
    "ب": "B",
    "د": "D",
    "ق": "Gh",
    "ه": "H",
    "ج": "J",
    "ل": "L",
    "م": "M",
    "ن": "N",
    "پ": "P",
    "ع": "PuV",
    "ژ": "PwD",
    "ص": "Sad",
    "س": "Sin",
    "ط": "T",
    "ت": "Taxi",
    "و": "V",
    "ی": "Y",
}

URL='http://127.0.0.1:8090/api/collections/registredDb/records'
date=datetime.datetime.now().date()
time=datetime.datetime.now().time()

df = pd.read_excel('dummy.xlsx')
count = df.shape[0]
names = df['name'].tolist()
plateFarsi = df['platenumber'].tolist()
plateEnglish = []
carname = df['carName'].tolist()
role = df['role'].tolist()
arvand = df['arvand'].tolist()

for plate in plateFarsi:
    # Translate each character; if not found, keep original character
    translated = ''.join(alphabetP2.get(ch, ch) for ch in str(plate))
    plateEnglish.append(translated)


for i in range(count):
    
    if arvand[i]:
        isarvand= 'arvand' 
        firstTwoDigit=""
        threeDigit=""
        lastTwoDigit="" 
        englishAlphabet="" 
        persinalAlphabet=""
        
    else:
        isarvand= 'notarvand' 
        firstTwoDigit=plateEnglish[i][0:2]
        threeDigit= plateEnglish[i][3:6]
        lastTwoDigit=plateEnglish[i][6:9]
        englishAlphabet=plateEnglish[i][2]
        persinalAlphabet= alphabetP1.get(plateEnglish[i][2])
        
    body = {
        "name": names[i],
        "carName": carname[i],
        "eDate": date,
        "eTime": time,
        "role": role[i],
        "rtpath": "/rt1",
        "plateNumber": plateEnglish[i],
        "isarvand": isarvand ,
        "firstTwoDigit": firstTwoDigit,
        "threeDigit":  threeDigit,
        "lastTwoDigit":  lastTwoDigit,
        "englishAlphabet": englishAlphabet,
        "persinalAlphabet":  persinalAlphabet
    }
    
    res=requests.post(URL,body)
    if res.status_code==200:
        print(res.json()['id'])

