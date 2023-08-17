import cv2
from roboflow import Roboflow

rf = Roboflow(api_key="nVylWKmHkJCkIKm4GEd7")
project = rf.workspace().project("malvas")
model = project.version(5).model

# infer on a local image

imagen = cv2.imread("5.jpg")
response = model.predict("5.jpg", confidence=80, overlap=30).json()

print(response)
#print(response['predictions'][0]['x'])

x1 = response['predictions'][0]['x'] - int(response['predictions'][0]['width']/2)
y1 = response['predictions'][0]['y'] - int(response['predictions'][0]['height']/2) #tabien

x2 = x1 + response['predictions'][0]['width']
y2 = y1 + response['predictions'][0]['height'] #tabien

cv2.rectangle(imagen,(x1,y1),(x2,y2),(0,255,0),3)#(imagen,(x1,y1),(x2,y2),(B,G,R),grosor)

cv2.imwrite("PRUEBA.jpg", imagen)
