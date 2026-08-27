"""FastAPI service for rare neurological disease classification from 2D MRI slices.

Backends
--------
convnext    (default) fine-tuned ConvNeXt-Tiny, softmax scores
resnet_svm            frozen ResNet-50 features + calibrated linear SVM

Both expose the same contract: predicted disease, confidence, and the full probability
distribution over the five classes. `calibrated_probabilities` in the response says whether
those numbers went through a real calibration (Platt scaling, SVM) or are raw softmax
scores (ConvNeXt) — reporting the difference matters more in a clinical framing than the
numbers themselves.

This is a demonstrator built on a small curated research dataset. It is not a medical
device and its output must not be used for diagnosis.
"""

import io
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision.models import convnext_tiny, resnet50

from gradcam import GradCAM, overlay_cam_on_image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "convnext").strip().lower()
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))

# Must match dataset.py exactly: a mismatch here silently degrades every prediction
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONVNEXT_WEIGHTS = MODEL_DIR / "convnext_tiny_finetuned_weights.pth"
CONVNEXT_ARTIFACTS = MODEL_DIR / "convnext_tiny_finetuned_artifacts.pkl"
RESNET_WEIGHTS = MODEL_DIR / "resnet50_svm_weights.pth"
RESNET_ARTIFACTS = MODEL_DIR / "resnet50_svm_artifacts.pkl"

# ORPHA codes, for a more informative response
ORPHA_CODES = {
    "walker_warburg_syndrome": "ORPHA:899",
    "pachygyria_cerebellar_hypoplasia": "ORPHA:2524",
    "moyamoya_disease": "ORPHA:2573",
    "hallervorden_spatz_disease": "ORPHA:157850",
    "fukuyama_muscular_dystrophy": "ORPHA:272",
}

preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class ClassProbability(BaseModel):
    class_name: str
    orpha_code: Optional[str] = None
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    orpha_code: Optional[str] = None
    confidence: float = Field(..., description="Probability assigned to the predicted class")
    probabilities: List[ClassProbability]
    model_backend: str
    calibrated_probabilities: bool = Field(
        ..., description="False means the scores are raw softmax outputs, not calibrated"
    )
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_backend: str
    device: str
    img_size: int
    classes: List[str]
    calibrated_probabilities: bool
    test_accuracy: Optional[float] = None
    test_macro_f1: Optional[float] = None
    gradcam_available: bool


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Loads exactly one backend at startup and keeps it warm."""

    def __init__(self):
        self.backend = MODEL_BACKEND
        self.class_names: List[str] = []
        self.artifacts: Dict = {}
        self.torch_model: Optional[nn.Module] = None
        self.svm = None
        self.scaler = None
        self.calibrated = False

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _load_artifacts(path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Artifacts not found: {path}\n"
                f"Copy the .pkl/.pth produced by the notebooks into {MODEL_DIR}."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _require_class_names(artifacts: Dict, path: Path) -> List[str]:
        """
        Refuse to start without the label mapping.

        The models predict integer indices only. Falling back to a hard-coded list would
        produce an API that looks healthy and returns permuted disease names — the worst
        possible failure mode here, because nothing about the output looks wrong.
        """
        class_names = artifacts.get('class_names')
        if not class_names:
            raise KeyError(
                f"'class_names' missing from {path}. Refusing to start: predictions would "
                f"be integer indices with no reliable mapping to disease names."
            )
        return list(class_names)

    # -- loaders ------------------------------------------------------------
    def load(self):
        if self.backend == "convnext":
            self._load_convnext()
        elif self.backend == "resnet_svm":
            self._load_resnet_svm()
        else:
            raise ValueError(
                f"Unknown MODEL_BACKEND '{self.backend}'. Use 'convnext' or 'resnet_svm'."
            )
        print(f"[startup] backend={self.backend} device={DEVICE} "
              f"classes={self.class_names}")

    def _load_convnext(self):
        self.artifacts = self._load_artifacts(CONVNEXT_ARTIFACTS)
        self.class_names = self._require_class_names(self.artifacts, CONVNEXT_ARTIFACTS)

        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(self.class_names))

        if not CONVNEXT_WEIGHTS.exists():
            raise FileNotFoundError(f"Weights not found: {CONVNEXT_WEIGHTS}")
        state = torch.load(CONVNEXT_WEIGHTS, map_location='cpu')
        model.load_state_dict(state)

        self.torch_model = model.to(DEVICE).eval()
        # Softmax over logits is not a calibrated probability, and the response says so
        self.calibrated = bool(self.artifacts.get('calibrated_probabilities', False))

    def _load_resnet_svm(self):
        self.artifacts = self._load_artifacts(RESNET_ARTIFACTS)
        self.class_names = self._require_class_names(self.artifacts, RESNET_ARTIFACTS)

        self.svm = self.artifacts.get('svm')
        self.scaler = self.artifacts.get('scaler')
        if self.svm is None or self.scaler is None:
            raise KeyError(f"'svm' and/or 'scaler' missing from {RESNET_ARTIFACTS}.")
        if not hasattr(self.svm, "predict_proba"):
            raise RuntimeError(
                "The stored SVM has no predict_proba: it was fitted without "
                "probability=True. Re-run notebook 01 (train_svm_classifier handles this)."
            )

        backbone = resnet50(weights=None)
        backbone.fc = nn.Identity()
        if RESNET_WEIGHTS.exists():
            backbone.load_state_dict(torch.load(RESNET_WEIGHTS, map_location='cpu'))
        else:
            # Fall back to the ImageNet weights: the backbone was never fine-tuned, so this
            # is equivalent, but it requires network access at startup
            from torchvision.models import ResNet50_Weights
            print("[startup] resnet50_svm_weights.pth not found, downloading ImageNet weights")
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            backbone.fc = nn.Identity()

        self.torch_model = backbone.to(DEVICE).eval()
        self.calibrated = bool(self.artifacts.get('calibrated_probabilities', True))

    # -- inference ----------------------------------------------------------
    def predict_proba(self, tensor: torch.Tensor) -> np.ndarray:
        if self.backend == "convnext":
            with torch.no_grad():
                logits = self.torch_model(tensor)
                return torch.softmax(logits, dim=1)[0].cpu().numpy()

        with torch.no_grad():
            feats = self.torch_model(tensor).cpu().numpy()
        return self.svm.predict_proba(self.scaler.transform(feats))[0]

    @property
    def gradcam_available(self) -> bool:
        # Grad-CAM needs gradients flowing to a class logit; the SVM head sits outside the
        # graph, so the explanation is only defined for the end-to-end ConvNeXt
        return self.backend == "convnext"

    def gradcam_target_layer(self) -> nn.Module:
        return self.torch_model.features[-1]


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the backend once, at startup, so no request pays the loading cost."""
    registry.load()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Rare Neurological Diseases MRI Classification",
    description=(
        "Classifies a 2D axial brain MRI slice into one of five rare neurological "
        "conditions. Research demonstrator only — not a medical device."
    ),
    version="1.0.0",
)

DISCLAIMER = ("Research demonstrator trained on a small curated dataset. "
              "Not a medical device; not for diagnostic use.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _read_image(file: UploadFile) -> Image.Image:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File larger than the {MAX_UPLOAD_MB} MB limit."
        )
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not decode the file as an image. Send a JPG or PNG slice."
        )


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return preprocess(image).unsqueeze(0).to(DEVICE)


def _build_response(probs: np.ndarray) -> PredictionResponse:
    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])
    pred_name = registry.class_names[pred_idx]

    return PredictionResponse(
        predicted_class=pred_name,
        orpha_code=ORPHA_CODES.get(pred_name),
        confidence=float(probs[pred_idx]),
        probabilities=[
            ClassProbability(
                class_name=registry.class_names[i],
                orpha_code=ORPHA_CODES.get(registry.class_names[i]),
                probability=float(probs[i]),
            )
            for i in order
        ],
        model_backend=registry.backend,
        calibrated_probabilities=registry.calibrated,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    ready = registry.torch_model is not None
    return {"status": "ok" if ready else "loading", "backend": registry.backend}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_backend=registry.backend,
        device=str(DEVICE),
        img_size=IMG_SIZE,
        classes=registry.class_names,
        calibrated_probabilities=registry.calibrated,
        test_accuracy=registry.artifacts.get('test_accuracy'),
        test_macro_f1=registry.artifacts.get('test_macro_f1'),
        gradcam_available=registry.gradcam_available,
    )


@app.get("/classes")
def classes():
    return {
        "classes": [
            {"index": i, "class_name": c, "orpha_code": ORPHA_CODES.get(c)}
            for i, c in enumerate(registry.class_names)
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Axial MRI slice, JPG or PNG")):
    image = await _read_image(file)
    probs = registry.predict_proba(_to_tensor(image))
    return _build_response(probs)


@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(..., description="Axial MRI slice, JPG or PNG"),
    target_class: Optional[str] = Query(
        None, description="Explain this class instead of the predicted one."
    ),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Heatmap blending weight."),
):
    """Returns a PNG with the Grad-CAM heatmap blended over the input slice."""
    if not registry.gradcam_available:
        raise HTTPException(
            status_code=501,
            detail=(
                "Grad-CAM is only defined for the end-to-end ConvNeXt backend. The "
                "resnet_svm pipeline classifies with an SVM outside the autograd graph, "
                "so no class logit gradient exists. Set MODEL_BACKEND=convnext."
            ),
        )

    image = await _read_image(file)
    tensor = _to_tensor(image)

    class_idx = None
    if target_class is not None:
        if target_class not in registry.class_names:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown class '{target_class}'. Available: {registry.class_names}",
            )
        class_idx = registry.class_names.index(target_class)

    with GradCAM(registry.torch_model, registry.gradcam_target_layer()) as cam_engine:
        cam, used_idx = cam_engine(tensor, class_idx=class_idx)

    overlay = overlay_cam_on_image(image, cam, alpha=alpha, size=IMG_SIZE)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)

    probs = registry.predict_proba(tensor)
    headers = {
        "X-Explained-Class": registry.class_names[used_idx],
        "X-Predicted-Class": registry.class_names[int(np.argmax(probs))],
        "X-Confidence": f"{float(probs.max()):.4f}",
        "X-Disclaimer": DISCLAIMER,
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)
