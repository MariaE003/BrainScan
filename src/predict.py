
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# charger le modele entraine
model = load_model("../models/brainscan_cnn.keras")

# garder le meme ordre que LabelEncoder
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def predict_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError("impossible de charger cette image.")

    # convertir BGR en RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    # normaliser les pixels entre 0 et 1
    image = image.astype("float32") / 255.0

    # ajouter la dimension du batch
    image = np.expand_dims(image, axis=0)

    # predire les probabilites
    probabilities = model.predict(image, verbose=0)

    # recuperer la classe la plus probable
    predicted_index = np.argmax(probabilities[0])
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[0][predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": dict(zip(
            class_names,
            probabilities[0].tolist()
        ))
    }
