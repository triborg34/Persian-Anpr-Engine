import requests
import json

substring='/api/collections/ipconfig/records'



with open('ip.json') as json_data:
    json = json.load(json_data)
    

print(json['ip'])
url=json['ip']
if len(url) <= 1:
    url=f'http://127.0.0.1:8090'

else:
    url=f'http://{url}:8090'



print(url)
response=requests.get(f'{url}/{substring}')
print(response.json())
# try:
#     url='http://127.0.0.1:8090/'




#     response=requests.get(f'{url}/{substring}')
#     print(response.json())
# except:

#     url="http://192.168.1.114:8090/"
#     response=requests.get(f'{url}/{substring}')
#     print(response.json())
    


