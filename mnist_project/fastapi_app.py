# fastapi_app.py
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model import ConvNet

app = FastAPI()

class ImageRequest(BaseModel):
    image: list

class PredictionResponse(BaseModel):
    prediction: int

# Charger le modèle
model_path = "convnet_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet(input_size=28*28, n_kernels=6, output_size=10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict(image: np.ndarray):
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_image(request: ImageRequest):
    image = np.array(request.image).astype(np.float32)
    prediction = predict(image)
    return PredictionResponse(prediction=prediction)
