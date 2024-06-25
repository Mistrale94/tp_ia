# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

st.title("Reconnaissance de chiffres manuscrits")

# Zone de dessin
st.subheader("Dessinez un chiffre")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, :3] * 255).astype(np.uint8))
    gray_image = image.convert("L")
    resized_image = gray_image.resize((28, 28))
    st.image(resized_image, caption="Image redimensionnée (28x28)", width=140)

# Bouton pour envoyer l'image
if st.button("Prédire"):
    if canvas_result.image_data is not None:
        image_array = np.array(resized_image).astype(np.float32) / 255.0
        image_array = image_array.tolist()

        response = requests.post(
            "http://localhost:8000/api/v1/predict",
            json={"image": image_array},
        )

        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Prédiction : {prediction}")
        else:
            st.error("Erreur lors de la prédiction")
    else:
        st.warning("Veuillez dessiner un chiffre d'abord")
