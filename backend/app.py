from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.image_model import classify_image
from models.text_model import analyze_sentiment
from loguru import logger

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

class TextRequest(BaseModel):
    text: str



@app.post("/predict/image")
async def predict_image(request: ImageRequest):
    """
    method to call predict image
    :param request:
    :return:
    """
    try:
        result = classify_image(request.image_url)
        logger.info(
            f'Received request with params from_ms: {result}'
        )
        return {"result": result}
    except Exception as e:
        logger.info(
            f'Received exception from_ms: {e}'
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/text")
async def predict_text(request: TextRequest):
    """
    method to call predict text
    :param request:
    :return:
    """
    try:
        result = analyze_sentiment(request.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))