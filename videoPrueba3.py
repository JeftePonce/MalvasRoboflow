import cv2
from roboflow import Roboflow

# Resto de tu código de inicialización de Roboflow
# Initialize Roboflow instance and access project and model as before
rf = Roboflow(api_key="-")
project = rf.workspace().project("malvas")
model = project.version(5).model

# Resto de tu código de apertura de video y obtención de propiedades
# Open the video file
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Define the codec and create VideoWriter object to save the output
output_path = "videoSalidaFrame.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

contMalvaMala = 0 
contMalvaBuena = 0
contFrame = 0
font = cv2.FONT_HERSHEY_TRIPLEX 

frames_to_skip = 5  # Analizar cada décimo fotograma
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    if frame_counter % frames_to_skip != 0:
        continue
    
    # Resto del código de procesamiento e inferencia
    response = model.predict(frame, confidence=70, overlap=30).json()
    # Resto del código de dibujo de cajas y recuento de malvas
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

print("La cantidad de malvas buenas es: " + str(contMalvaBuena/10))
print("La cantidad de malvas malas es: " + str(contMalvaMala/10))
print("La cantidad de frames es: " + str(contFrame))
