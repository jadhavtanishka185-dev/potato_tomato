import gdown
import hmac
import io
import os
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


# Placeholder classes for tomato and potato disease labels.
CLASS_LABELS: List[str] = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv("MODEL_PATH", "model.h5")
API_KEY_ENV_NAME = "API_KEY"

app = FastAPI(title="Plant Disease Prediction API", version="1.0.0")

# Load model once at module import so every request reuses it.
try:
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading from Google Drive...")

        # Replace with your actual file ID
        FILE_ID = '1PFQHgD2V_av1m3zO4XyLurvAyMU0iOmU'

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        print("Model downloaded successfully.")

    # Load model after ensuring it exists
    model = tf.keras.models.load_model(MODEL_PATH)

except Exception as exc:
    raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {exc}") from exc

def get_api_key() -> str:
    key = os.getenv(API_KEY_ENV_NAME)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV_NAME} is not set. Add it to your deployment environment variables."
        )
    return key


def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    expected = get_api_key()
    if not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    return array


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image_tensor = preprocess_image_bytes(image_bytes)

    try:
        predictions = model.predict(image_tensor, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    pred_index = int(np.argmax(predictions))
    confidence = float(predictions[pred_index])
    predicted_class = CLASS_LABELS[pred_index] if pred_index < len(CLASS_LABELS) else str(pred_index)

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4),
    }

