from fastapi import APIRouter, UploadFile, File  # type: ignore[import]
from PIL import Image

from app.services.prediction import predict

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    return predict(image)