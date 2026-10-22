from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Models

class Coordinates(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectedObject(BaseModel):
    label: str
    confidence: float
    coordinates: Coordinates

class APIResponse(BaseModel):
    message: str
    total_objects: int
    detected_objects: list[DetectedObject]


# 2. API Initialization
app = FastAPI(
    title="Object Detection API", 
    description="An API for detecting objects in images using YOLOv8.", 
    version="1.0.0"
    )

# model = YOLO(f'./runs/detect/train/weights/best.pt')  # Load the YOLOv8 model
MODELS = {
        "yolov8n": YOLO(f'./runs/detect/train-v8n/weights/best.pt'),
        "yolo26n": YOLO(f'./runs/detect/train-26n/weights/best.pt'),
        "yolo26s-1280": YOLO(f'./runs/detect/train-26s-1280/weights/best.pt'),
    }

# 3. API Endpoints

@app.post("/detect_objects", response_model=APIResponse)
async def detect_objects(file: UploadFile = File(...), model_name: str = Form("yolov8n")):

    model = MODELS.get(model_name)  # Change this to "yolo26" or "yolo26n" as needed

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img)
    result = results[0]  # Get the first result

    detected_objects = []
    if len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])  # Get the class ID
            label = model.names[cls_id]  # Get the class label
            confidence = float(box.conf[0])  # Get the confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates

            detected_objects.append(DetectedObject(
                label=label,
                confidence=confidence,
                coordinates=Coordinates(x1=x1, y1=y1, x2=x2, y2=y2)
            ))

    return APIResponse(
        message="Objects detected successfully.",
        total_objects=len(detected_objects),
        detected_objects=detected_objects
    )