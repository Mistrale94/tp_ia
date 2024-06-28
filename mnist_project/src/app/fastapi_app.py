# src/app/fastapi_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
from src.model.model import ConvNet
import uvicorn
from torchvision import transforms

app = FastAPI()

class ImageRequest(BaseModel):
    image: list

class PredictionResponse(BaseModel):
    prediction: int

VERSION="0.0.1"
# Charger le modèle
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', f"mnist.{VERSION}.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(image: np.ndarray) -> np.ndarray:
    """
    Run model and get result
    :param package: dict from fastapi state including model and processing objects
    :param input: list of input values or an image in a suitable format
    :return: numpy array of model output
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image)
    image = torch.tensor(image, dtype=torch.float32)

    image = image.to(device)
   

    perm=torch.arange(0, 784).long()
    image = image.view(-1, 28*28)
    image = image[:, perm]
    image = image.view(-1, 1, 28, 28)

    # Run the model
    model = ConvNet(input_size=28*28, n_kernel=6, output_size=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred = model(image)

    pred = torch.argmax(y_pred, dim=1)
    pred = pred.cpu().numpy()
    print(f"Server side prediction : {pred}")

    return pred

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_image(request: ImageRequest):
    image = np.array(request.image).astype(np.float32) / 255.0

    prediction = predict(image)
    return PredictionResponse(prediction=prediction)



if __name__ == '__main__':
    # server api
    uvicorn.run("fastapi_app:app", host="localhost", port=8000, reload=True)