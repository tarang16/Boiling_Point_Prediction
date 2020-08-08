import requests
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

url = 'http://localhost:5000/predict_api'
[[a,b,c]]=model1.transform([[18, 647, 0.35]])
r = requests.post(url,json={'weight':a, 'ctemp':b, 'afactor':c})

print(r.json())