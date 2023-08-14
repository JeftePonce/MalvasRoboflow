from roboflow import Roboflow
rf = Roboflow(api_key="nVylWKmHkJCkIKm4GEd7")
project = rf.workspace().project("malvas")
model = project.version(5).model

response = model.predict("3.jpg", confidence=70, overlap=30).json()

contMalvaMala = 0
contMalvaBuena = 0

for pred in response['predictions']:
    
    if pred['class'] == 'malvaBuena':
        contMalvaBuena = contMalvaBuena + 1

    if pred['class'] == 'malvaMala':
        contMalvaMala = contMalvaMala + 1
    
print("La cantidad de malvas buenas es: " + str(contMalvaBuena))
print("La cantidad de malvas malas es: " + str(contMalvaMala))
