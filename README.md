# Plant Disease Prediction API (FastAPI)

Production-ready FastAPI backend for serving a TensorFlow/Keras model (`model.h5`) with API key protection.

## Files Included

- `main.py` - FastAPI app with `/predict` and `/health`
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime pin for Render (`python-3.10.13`)
- `Procfile` - Web process command for deployment
- `.env.example` - Example environment variable template
- `test_api.py` - Local API test script with `requests`

## 1) Setup Environment Variables

Set these environment variables:

- `API_KEY` (required): your secure API key
- `MODEL_PATH` (optional): defaults to `model.h5`

### Windows PowerShell

```powershell
$env:API_KEY="REPLACE_WITH_YOUR_API_KEY"
$env:MODEL_PATH="model.h5"
```

### Linux/macOS

```bash
export API_KEY="REPLACE_WITH_YOUR_API_KEY"
export MODEL_PATH="model.h5"
```

## 2) Install Dependencies

```bash
pip install -r requirements.txt
```

## 3) Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4) Predict Endpoint

- **Method**: `POST`
- **URL**: `/predict`
- **Header**: `x-api-key: <YOUR_API_KEY>`
- **Body**: `multipart/form-data` with key `file` (image)

### Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "x-api-key: YOUR_API_KEY" \
  -F "file=@sample.jpg"
```

### Response format

```json
{
  "class": "Tomato___Late_blight",
  "confidence": 0.95
}
```

## 5) Test with Python Script

```bash
export API_KEY="YOUR_API_KEY"
export API_URL="http://127.0.0.1:8000/predict"
export IMAGE_PATH="sample.jpg"
python test_api.py
```

On Windows PowerShell:

```powershell
$env:API_KEY="YOUR_API_KEY"
$env:API_URL="http://127.0.0.1:8000/predict"
$env:IMAGE_PATH="sample.jpg"
python test_api.py
```

## 6) Render Deployment (TensorFlow-Compatible)

Create a new **Web Service** on Render and point it to this repository root.

1. Confirm runtime pin in `runtime.txt`:
   - `python-3.10.13`
2. Configure service commands:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}`
3. Set required environment variables in Render:
   - `API_KEY` = your secure API key (required)
   - `PYTHON_VERSION` = `3.10.13` (recommended explicit pin)
4. Optional environment variable:
   - `MODEL_PATH` = `model.h5` (defaults to `model.h5`)

Notes:

- This project uses `tensorflow-cpu==2.13.0` for better compatibility on Render.
- Ensure `model.h5` exists in the deployed filesystem (repo, disk, or mounted path).
