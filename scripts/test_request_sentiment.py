import requests

url = "http://0.0.0.0:9090/"
data = "This game was great"
response = requests.post(url, data=data)
print(response.text)
