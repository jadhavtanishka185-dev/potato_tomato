import os
import sys

import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
API_KEY = os.getenv("API_KEY", "")
IMAGE_PATH = os.getenv("IMAGE_PATH", "sample.jpg")


def main() -> None:
    if not API_KEY:
        raise ValueError("Set API_KEY environment variable before running test_api.py")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
        headers = {"x-api-key": API_KEY}
        response = requests.post(API_URL, files=files, headers=headers, timeout=60)

    print(f"Status: {response.status_code}")
    try:
        print("Response JSON:", response.json())
    except Exception:
        print("Response Text:", response.text)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

