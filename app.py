
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("models/brainscan_cnn.keras")

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# configurer la page
st.set_page_config(page_title="BrainScan", page_icon=None)

st.title("BrainScan - Classification d'IRM cerebrales")
st.write("Charge une image IRM pour obtenir une prediction du modele.")


uploaded_file = st.file_uploader(
    "Choisir une image IRM",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    # lire et afficher l'image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargee", use_container_width=True)

    # preparer l'image comme pendant l'entrainement
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # predire la classe
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    # afficher le resultat
    st.subheader("Resultat de la prediction")
    st.write(f"Classe predite : **{predicted_class}**")
    st.write(f"Score du modele : **{confidence * 100:.2f}%**")

    # afficher les scores de toutes les classes
    scores = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }
    st.subheader("Scores par classe")
    st.bar_chart(scores)
