import requests
import json

from configParams import Parameters
params=Parameters()


quality=requests.get(f"http://{params.defip}:{params.defport}/api/collections/setting/records")
print(quality.json()['items'][len(quality.json()['items'])-1]["quality"])




#     response=requests.get(f'{url}/{substring}')
#     print(response.json())
# except:

#     url="http://192.168.1.114:8090/"
#     response=requests.get(f'{url}/{substring}')
#     print(response.json())
    


