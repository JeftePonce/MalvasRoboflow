# -**SOLUCION PARCHE**- LA IDEA PRINCIPAL DE ESTE CODIGO ES DESCOMPONER EL VIDEO POR FRAMES Y ANALIZAR FRAME POR FRAME EL VIDEO PARA CONSEGUIR ETIQUETAR LAS MALVAS 

import cv2
from roboflow import Roboflow

# Initialize Roboflow instance and access project and model as before
rf = Roboflow(api_key="-")
project = rf.workspace().project("malvas")
model = project.version(5).model

# Open the video file
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create VideoWriter object to save the output
output_path = "videoSalida2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

contMalvaMala = 0 
contMalvaBuena = 0
contFrame = 0
font = cv2.FONT_HERSHEY_TRIPLEX 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform inference on the frame
    response = model.predict(frame, confidence=70, overlap=30).json()
    
    
    # Process the prediction response and draw bounding boxes
    """for prediction in response['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(x1 + prediction['width'])
        y2 = int(y1 + prediction['height'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)"""

    for pred in response['predictions']: #recorrer el json de predictions
        contFrame = contFrame + 1
        x1 = int(pred['x'] - pred['width'] / 2)
        y1 = int(pred['y'] - pred['height'] / 2)

        x2 = int(x1 + pred['width'])
        y2 = int(y1 + pred['height'])
    
        if pred['class'] == 'malvaBuena': 
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,102,0),1)#(imagen,(x1,y1),(x2,y2),(B,G,R),grosor) funcion para dibujar rectangulos en la imagen
            cv2.putText(frame, 'malvaBuena', (x1,y1-5), font,1,(0,102,0),2,cv2.LINE_AA) #funcion para escribir texto en la imagen
            contMalvaBuena = contMalvaBuena + 1

        if pred['class'] == 'malvaMala':
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.putText(frame, 'malvaMala', (x1,y1-5), font,1,(0,0,255),2,cv2.LINE_AA)
            contMalvaMala = contMalvaMala + 1
    
    cv2.putText(frame, f'Malvas buenas: {contMalvaBuena}', (10,30), font,1,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, f'Malvas malas: {contMalvaMala}', (10,60), font,1,(0,0,0),2,cv2.LINE_AA)
    # Write the frame with bounding boxes to the output video
    out.write(frame)

# Release the capture and output objects
cap.release()
out.release()

cv2.destroyAllWindows()

print("La cantidad de malvas buenas es: " + str(contMalvaBuena))
print("La cantidad de malvas malas es: " + str(contMalvaMala))
print("La cantidad de frames es: " + str(contFrame))
