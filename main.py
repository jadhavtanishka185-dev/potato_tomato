import gdown
import hmac
import io
import json
import os
import shutil
import tempfile
from typing import List

import h5py
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

def _remove_quantization_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for nested_value in value.values():
            _remove_quantization_config(nested_value)
    elif isinstance(value, list):
        for item in value:
            _remove_quantization_config(item)


def _build_compat_model_copy(model_path: str) -> str:
    fd, temp_model_path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    shutil.copyfile(model_path, temp_model_path)

    with h5py.File(temp_model_path, "r+") as h5_file:
        raw_model_config = h5_file.attrs.get("model_config")
        if raw_model_config is None:
            return temp_model_path

        if isinstance(raw_model_config, bytes):
            model_config = json.loads(raw_model_config.decode("utf-8"))
        else:
            model_config = json.loads(raw_model_config)

        _remove_quantization_config(model_config)
        h5_file.attrs["model_config"] = json.dumps(model_config).encode("utf-8")

    return temp_model_path


def load_prediction_model(model_path: str):
    if not os.path.exists(model_path):
        print("Model not found. Downloading from Google Drive...")

        file_id = "1PFQHgD2V_av1m3zO4XyLurvAyMU0iOmU"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
        print("Model downloaded successfully.")

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        compat_model_path = _build_compat_model_copy(model_path)
        try:
            return tf.keras.models.load_model(compat_model_path, compile=False)
        finally:
            if os.path.exists(compat_model_path):
                os.remove(compat_model_path)


# Load model once at module import so every request reuses it.
try:
    model = load_prediction_model(MODEL_PATH)
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

