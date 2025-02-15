from fastapi import FastAPI, HTTPException
import requests
from loguru import logger

app = FastAPI()

BACKEND_URL = "http://backend:8000"

# front end api details
@app.post("/predict/image")
async def predict_image(image_url: str):
    response = requests.post(f"{BACKEND_URL}/predict/image", json={"image_url": image_url})
    logger.info(
        f'Received response with params from_ms: {response}'
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()

@app.post("/predict/text")
async def predict_text(text: str):
    response = requests.post(f"{BACKEND_URL}/predict/text", json={"text": text})
    logger.info(
        f'Received response with params from_ms: {response}'
    )
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()