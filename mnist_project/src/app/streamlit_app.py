# src/app/streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import uvicorn
import cv2

st.title("Reconnaissance de chiffres manuscrits")

# Zone de dessin
st.subheader("Dessinez un chiffre")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = cv2.resize(canvas_result.image_data.astype("uint8"),
                     (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_rescaled = cv2.resize(
        image, (28*8, 28*8), interpolation=cv2.INTER_NEAREST)
    st.write("Model input")
    st.image(image_rescaled)

# Bouton pour envoyer l'image
if st.button("Prédire"):
    if canvas_result.image_data is not None:
        image_array = image.tolist()

        response = requests.post(
            "http://backend:8000/api/v1/predict",
            json={"image": image_array},
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Prédiction : {prediction}")
        else:
            st.error("Erreur lors de la prédiction")
    else:
        st.warning("Veuillez dessiner un chiffre d'abord")

