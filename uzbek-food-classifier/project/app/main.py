from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import torch
import torch.nn.functional as F

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from app.model import load_model
from app.utils import get_transform, load_class_names

WEIGHTS_PATH = "weights/Uzb_Food_Classifier.02.pth"
CLASS_NAMES_PATH = "app/class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = load_class_names(CLASS_NAMES_PATH)
model = load_model(WEIGHTS_PATH, len(class_names), device)
transform = get_transform()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        result = class_names[predicted.item()]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "prob": round(confidence.item() * 100, 2)
    })
