from roboflow import Roboflow
rf = Roboflow(api_key="nVylWKmHkJCkIKm4GEd7")
project = rf.workspace().project("malvas")
model = project.version(5).model

# infer on a local image
print(model.predict("3.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("3.jpg", confidence=40, overlap=70).save("prediction3.jpg")

# infer on an image hosted elsewhere
#print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())