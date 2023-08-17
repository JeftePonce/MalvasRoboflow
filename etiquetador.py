import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="-")
project = rf.workspace().project("malvas")
model = project.version(5).model

imagen = cv2.imread("1.jpg") #imagen que usaremos para predecir

response = model.predict("1.jpg", confidence=80, overlap=30).json() #json en donde estan las predicciones del modelo

contMalvaMala = 0
contMalvaBuena = 0
font = cv2.FONT_HERSHEY_TRIPLEX 

#print(len(response['predictions']))

for pred in response['predictions']: #recorrer el json de predictions

    x1 = pred['x'] - int(pred['width']/2) #coordenada X en donde empieza se empieza a dibujar el rectangulo para seleccionar malva 
    y1 = pred['y'] - int(pred['height']/2) #coordenada Y en donde empieza se empieza a dibujar el rectangulo para seleccionar malva 

    x2 = x1 + pred['width'] #coordenada X en donde empieza se termina de dibujar el rectangulo para seleccionar malva 
    y2 = y1 + pred['height'] #coordenada Y en donde empieza se termina de dibujar el rectangulo para seleccionar malva 
    
    if pred['class'] == 'malvaBuena': 
        cv2.rectangle(imagen,(x1,y1),(x2,y2),(0,102,0),3)#(imagen,(x1,y1),(x2,y2),(B,G,R),grosor) funcion para dibujar rectangulos en la imagen
        cv2.putText(imagen, 'malvaBuena', (x1,y1-5), font,1,(0,102,0),2,cv2.LINE_AA) #funcion para escribir texto en la imagen
        contMalvaBuena = contMalvaBuena + 1

    if pred['class'] == 'malvaMala':
        cv2.rectangle(imagen,(x1,y1),(x2,y2),(0,0,255),3)#(imagen,(x1,y1),(x2,y2),(B,G,R),grosor)
        cv2.putText(imagen, 'malvaMala', (x1,y1-5), font,1,(0,0,255),2,cv2.LINE_AA)
        contMalvaMala = contMalvaMala + 1
    
print("La cantidad de malvas buenas es: " + str(contMalvaBuena))
print("La cantidad de malvas malas es: " + str(contMalvaMala))

cv2.putText(imagen, f'Malvas buenas: {contMalvaBuena}', (10,30), font,1,(0,0,0),2,cv2.LINE_AA)
cv2.putText(imagen, f'Malvas malas: {contMalvaMala}', (10,60), font,1,(0,0,0),2,cv2.LINE_AA)

cv2.imwrite("rectangulos1.jpg", imagen) #funcion para guardar la imagen en la ruta especificada
