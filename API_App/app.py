"""FastAPI service for rare neurological disease classification from 2D MRI slices

Single model: the fine-tuned ConvNeXt-Tiny
It was preferred over the ResNet-50 + SVM baseline not on accuracy (3 images out of 300, not statistically significant)
but mostly because it is the only one that supports Grad-CAM
Plusits embedding separates the classes far better, and it deploys as a
single state_dict instead of a scikit-learn pickle tied to a library version (as it would have been with the SVM)

-> Research demonstrator: not a medical device, not for diagnostic use
"""

import io
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision.models import convnext_tiny

from gradcam import GradCAM, overlay_cam_on_image

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
ENABLE_REJECTION = os.getenv("ENABLE_REJECTION", "true").lower() != "false"

STATIC_DIR = Path(__file__).parent / "static"
WEIGHTS_PATH = MODEL_DIR / "convnext_tiny_finetuned_weights.pth"
DEPLOY_PATH = MODEL_DIR / "convnext_tiny_deploy.pkl"

ORPHA_CODES = {
    "walker_warburg_syndrome": "ORPHA:899",
    "pachygyria_cerebellar_hypoplasia": "ORPHA:2524",
    "moyamoya_disease": "ORPHA:2573",
    "hallervorden_spatz_disease": "ORPHA:157850",
    "fukuyama_muscular_dystrophy": "ORPHA:272",
}
DISCLAIMER = ("Research demonstrator trained on a small curated dataset "
              "Not a medical device; not for diagnostic use")

# must match dataset.py, or every prediction silently degrades
preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ClassProbability(BaseModel):
    class_name: str
    orpha_code: Optional[str] = None
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    orpha_code: Optional[str] = None
    confidence: float
    probabilities: List[ClassProbability]
    in_distribution: bool = Field(
        ..., description="False when the image does not look like any of the five classes; "
                         "the prediction is then MEANINGLESS and must be ignored")
    confidence_threshold: Optional[float] = None
    warning: Optional[str] = None
    disclaimer: str = DISCLAIMER


class ModelInfoResponse(BaseModel):
    model_architecture: str
    device: str
    img_size: int
    classes: List[str]
    rejection_enabled: bool
    confidence_threshold: Optional[float] = None
    test_accuracy: Optional[float] = None
    ece: Optional[float] = None


class Classifier:
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.threshold: Optional[float] = None
        self.deploy: dict = {}

    def load(self):
        if not DEPLOY_PATH.exists():
            raise FileNotFoundError(
                f"{DEPLOY_PATH} not found. Run the deployment-check cell (section 10) of notebook 02 and copy "
                f"convnext_tiny_deploy.pkl into {MODEL_DIR}.")
        with open(DEPLOY_PATH, 'rb') as f:
            self.deploy = pickle.load(f)

        # the model outputs integer indices; a wrong mapping gives confident, mislabelled answers
        self.class_names = list(self.deploy.get('class_names') or [])
        if not self.class_names:
            raise KeyError(f"'class_names' missing from {DEPLOY_PATH.name}; refusing to start.")

        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(self.class_names))
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        self.model = model.to(DEVICE).eval()

        thr = self.deploy.get('confidence_threshold')
        self.threshold = float(thr) if thr is not None else None
        print(f"[startup] ConvNeXt-Tiny on {DEVICE} | classes={self.class_names} | "
              f"rejection={'on' if self.rejection_active else 'off'}")

    @property
    def rejection_active(self) -> bool:
        return ENABLE_REJECTION and self.threshold is not None

    def predict(self, tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()


clf = Classifier()


@asynccontextmanager
async def lifespan(_: FastAPI):
    clf.load()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Rare Neurological Diseases MRI Classification",
    description=("Classifies a 2D axial brain MRI slice into one of five rare neurological conditions"
                 "-> Research demonstrator only — not a medical device."),
    version="2.1.0",
)

# The browser interface is served by the API itself: one static page calling the same public endpoints any other client would use
# StaticFiles ships with Starlette, so this adds no dependency to the image
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui():
    """The web interface. Falls back to the API docs if the page is not bundled"""
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return RedirectResponse("/docs")


async def _read_image(file: UploadFile) -> Image.Image:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not decode the file. Send a JPG or PNG image")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return preprocess(image).unsqueeze(0).to(DEVICE)

# msg on http://localhost:8000/health is "ok" when the model is loaded, "loading" while it is being loaded
@app.get("/health")
def health():
    return {"status": "ok" if clf.model is not None else "loading",
            "model": "convnext_tiny_finetuned"}

# msg on http://localhost:8000/model-info is a JSON with the model architecture, device, image size, class names, rejection status, 
# confidence threshold, test accuracy and ECE
@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_architecture="convnext_tiny (last stage + head fine-tuned)",
        device=str(DEVICE), img_size=IMG_SIZE, classes=clf.class_names,
        rejection_enabled=clf.rejection_active, confidence_threshold=clf.threshold,
        test_accuracy=clf.deploy.get('test_accuracy'), ece=clf.deploy.get('ece'))

# msg on http://localhost:8000/classes is a JSON with the class names and their Orpha codes
@app.get("/classes")
def classes():
    return {"classes": [{"index": i, "class_name": c, "orpha_code": ORPHA_CODES.get(c)}
                        for i, c in enumerate(clf.class_names)]}

# msg on http://localhost:8000/predict is a JSON with the predicted class, its Orpha code, confidence, 
# probabilities for all classes, in_distribution flag, confidence threshold and warning message if any
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Axial MRI slice, JPG or PNG")):
    probs = clf.predict(_to_tensor(await _read_image(file)))
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    confidence = float(probs[top])

    in_dist, warning = True, None
    if clf.rejection_active and confidence < clf.threshold:
        in_dist = False
        warning = (
            f"Confidence {confidence:.3f} is below the threshold {clf.threshold:.3f}. This image "
            "does not resemble the training data: the model has only five outputs and always "
            "returns one of them, so the prediction below is not meaningful. Likely a healthy "
            "scan, a different pathology, another modality, or a non-medical image.")

    return PredictionResponse(
        predicted_class=clf.class_names[top],
        orpha_code=ORPHA_CODES.get(clf.class_names[top]),
        confidence=confidence,
        probabilities=[ClassProbability(class_name=clf.class_names[i],
                                        orpha_code=ORPHA_CODES.get(clf.class_names[i]),
                                        probability=float(probs[i])) for i in order],
        in_distribution=in_dist,
        confidence_threshold=clf.threshold,
        warning=warning)

# msg on http://localhost:8000/gradcam is a PNG image with the Grad-CAM heatmap blended over the input slice,
# and the response headers contain the explained class, predicted class, confidence, in_distribution flag and disclaimer
@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(..., description="Axial MRI slice, JPG or PNG"),
    target_class: Optional[str] = Query(None, description="Explain this class instead of the predicted one"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
):
    """PNG with the Grad-CAM heatmap blended over the input image"""
    image = await _read_image(file)
    tensor = _to_tensor(image)

    class_idx = None
    if target_class is not None:
        if target_class not in clf.class_names:
            raise HTTPException(status_code=400,
                                detail=f"Unknown class '{target_class}'. Available: {clf.class_names}")
        class_idx = clf.class_names.index(target_class)

    # features[-1] emits [B, 768, 7, 7]: ConvNeXt keeps the spatial layout, so no reshape needed
    with GradCAM(clf.model, clf.model.features[-1]) as engine:
        cam, used_idx = engine(tensor, class_idx=class_idx)

    buf = io.BytesIO()
    overlay_cam_on_image(image, cam, alpha=alpha, size=IMG_SIZE).save(buf, format="PNG")
    buf.seek(0)

    probs = clf.predict(tensor)
    return StreamingResponse(buf, media_type="image/png", headers={
        "X-Explained-Class": clf.class_names[used_idx],
        "X-Predicted-Class": clf.class_names[int(np.argmax(probs))],
        "X-Confidence": f"{float(probs.max()):.4f}",
        "X-In-Distribution": str(not (clf.rejection_active and probs.max() < clf.threshold)),
        "X-Disclaimer": DISCLAIMER,
    })
