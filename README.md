# Plant Disease Prediction API (FastAPI)

Production-ready FastAPI backend for serving a TensorFlow/Keras model (`model.h5`) with API key protection.

## Files Included

- `main.py` - FastAPI app with `/predict` and `/health`
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime for deployment
- `Procfile` - Web process command for deployment
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

## 6) Render Deployment

Create a new Web Service and point it to this repo.

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`

Set Render environment variables:

- `API_KEY` = your secure API key
- `MODEL_PATH` = `model.h5` (or your actual model location)

Also ensure `model.h5` is present in deployment storage/repo or mounted location.
