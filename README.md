# Rare Neurological Diseases MRI Classification
_Feature Extraction vs Transfer Learning under Limited Data Regime_

## Project Description

Rare neurological diseases often suffer from extremely long diagnostic delays (the so-called “diagnostic odyssey”), averaging 5–7 years. One of the main causes is the scarcity of high-quality annotated imaging data. In this project we investigate which deep-learning strategy is more robust when only a limited number of MRI examples per class is available.

Two strategies are compared on identical data, identical splits and an identical random seed:

1. **Classical Feature Extraction** – _ResNet-50_ frozen as a feature extractor (2048-D embeddings, zero trainable parameters) followed by a linear **SVM** with Platt-calibrated probabilities
2. **Transfer Learning** – _ConvNeXt-Tiny_, a newest model, with lightweight fine-tuning: the ImageNet backbone is frozen except for its last stage, and the classification head is replaced with a 5-way linear layer

Both are evaluated on the full balanced dataset and on progressively reduced training sets, to quantify robustness under data scarcity

Augmentation is medical-domain specific (TorchIO) and deliberately excludes horizontal flipping, which would destroy diagnostically meaningful lesion laterality \
The best-performing model is then served through a FastAPI application, containerised with Docker, which returns the predicted disease, a confidence score, the full probability distribution over the five classes, a rejection flag for out-of-distribution inputs, and an optional Grad-CAM explanation \
The same service also ships a small web interface, with views to classify a slice, generate a Grad-CAM heatmap, and inspect the deployed model

## Dataset
**Rare Neurological Diseases MRI – Curated Edition**
[Kaggle link](<https://www.kaggle.com/datasets/ahsanneural/rare-neurological-diseases-mri-curated-edition>)

The linked dataset has the following main charactertistics:
- 2 000 high-quality axial MRI slices (JPG)
- 5 perfectly balanced classes (400 images each)
- Official ORPHA codes included
- Pre-split into train / validation / test (70 % / 15 % / 15 %)

| Disease | ORPHA Code | Images |
|---------|------------|--------|
| Walker-Warburg Syndrome | ORPHA:899 | 400 |
| Pachygyria with Cerebellar Hypoplasia | ORPHA:2524 | 400 |
| Moyamoya Disease with IVH | ORPHA:2573 | 400 |
| Pantothenate Kinase-Associated Neurodegeneration (PKAN) | ORPHA:157850 | 400 |
| Fukuyama Muscular Dystrophy | ORPHA:272 | 400 |

All thos diseases are part of the rarest ones, less analyzed in other datasets.

Perfect balance makes plain accuracy a fair metric, and macro-F1 and weighted-F1 coincide throughout

> Note that _Walker-Warburg_ and _Fukuyama_ belong to the same category, so confusion between them is clinically plausible rather than a sign of a broken model

---
## Project Results

**On the full dataset the two strategies are statistically indistinguishable** 

ConvNeXt-Tiny reaches **99.67 %** test accuracy (299/300) against **98.67 %** (296/300) for ResNet-50 + SVM, on the same 300 images. \
The difference is three images: _McNemar's_ exact test on the paired disagreements gives **p ≥ 0.25** under every possible split of the errors, so the full-data comparison is a tie.

**Under data scarcity they diverge sharply, and in favour of the simpler method**

With only **10 labelled examples per class** the frozen backbone reaches **71.6 % ± 1.1** against **56.0 % ± 7.3** for fine-tuning — a gap of 15.6 points, and roughly **7× more stable** across random subsets. \
The two curves cross over at about 50–100 examples per class. For rare diseases, where a handful of annotated cases is the realistic scenario, classical feature extraction is the better engineering choice.

**Fine-tuning nonetheless restructures the feature space substantially.** Measured with an LDA
projection on held-out data, class separation rises from **3.94** (frozen ImageNet features) to
**56.95** (fine-tuned embedding), roughly a fourteenfold improvement — which is why it wins once data is
sufficient.

**ConvNeXt-Tiny is the deployed model**, chosen not on the statistically insignificant accuracy difference but because Grad-CAM is impossible with the SVM (its classifier sits outside the autograd graph), because its embedding is far better separated, and because it deploys as a single `state_dict` rather than a version-fragile scikit-learn pickle. 
> A confidence threshold at **0.9990** — the 5th percentile of validation confidences — lets the service refuse inputs that are not one of the five conditions.

**Limitations** 

The five classes differ in preprocessing style as well as in anatomy, so these figures measure separability on this dataset as distributed rather than diagnostic performance. \
The test set (300 images, 1 and 4 errors) cannot resolve differences of a few images. `metadata.csv` carries no
patient identifier and the split is random at image level, so same-patient leakage cannot be excluded.

_Full analysis, figures and discussion: **[`summary.ipynb`](summary.ipynb)**_

---

## Repository structure

In this section there is a visualization of how the folders are organized:

```bash

rare-neuro-mri-classification/
├── API_App/                          # Web frontend and FastAPI REST backend service
│   └── models/                                  # Serialized model artifacts
│   │   ├── .gitkeep
│   │   ├── convnext_tiny_deploy.pkl             # Production deployment bundle (model + metadata)
│   │   └── convnext_tiny_finetuned_weights.pth  # PyTorch state dict for fine-tuned ConvNeXt-Tiny
|   └── static/                       # Static web assets 
│   │   └── index.html                      # Single-page frontend application
│   ├── app.py                            # FastAPI entry point defining REST endpoints
│   ├── docker-compose.yml                # Docker Compose service orchestration config
│   ├── Dockerfile                        # Container build configuration for API deployment
│   ├── gradcam.py                        # Grad-CAM algorithm implementation for visual explainability
│   └── requirements.txt                  # Python dependencies for the REST API container         
├── data/                             # Dataset root directory (Git-ignored)
│   └── rare_neuro_mri_curated/           # Curated MRI dataset (downloaded via Kaggle/script)
├── notebooks/                        # Jupyter notebooks for model development and evaluation
│   ├── 01_resnet50_svm.ipynb             # Baseline pipeline (ResNet-50 feature extraction + SVM)
│   └── 02_convnext-tiny_ft.ipynb         # Deep learning pipeline (ConvNeXt-Tiny fine-tuning)
├── src/                              # Core Python modules and data pipelines
│   ├── __init__.py
│   ├── dataset.py                        # Dataset handling, preprocessing, and augmentations
│   ├── feature_extractor.py              # Feature extraction routines for classical ML baseline
│   ├── metrics.py                        # Evaluation metrics 
│   └── transfer_learning.py              # ConvNeXt-Tiny fine-tuning routines and training loops                   
├── utils/                             # Project artifacts, visualizations, and assets
│   ├── api_test_images-5296/              # Generated test batch of augmented images (seed: 5296)
│   ├── convnext/                          # Output plots and performance figures for ConvNeXt-Tiny
│   ├── gui_screen/                        # UI screenshots for documentation
│   ├── resnet/                            # Output plots and performance figures for ResNet-50 + SVM
│   └── augmented-samples-test-5296.png  # Visualization grid of generated data augmentations
├── .gitattributes
├── .gitignore                            # Excludes datasets, checkpoints, and local environments
├── README.md                             # Main project documentation
├── requirements.txt                      # Core Python dependencies for model training & notebooks
└── summary.ipynb                         # Master notebook consolidating overall project evaluation

```
---

# How to run the application

### Option A — pull the published image (nothing to build)

```bash
docker run -d -p 8000:8000 --name rare-neuro-mri-api ghcr.io/rayoyo/rare-neuro-mri-api:latest
```

The image is self-contained: the trained model is baked in.

### Option B — build locally

The trained weights are **not** in the repository (110 MB). Download them from the
[latest release](https://github.com/Rayoyo/rare-neuro-mri/releases/latest) into `API_App/models/`
first, otherwise the build fails at `COPY models/`:

```bash
mkdir -p API_App/models
curl -L -o API_App/models/convnext_tiny_finetuned_weights.pth \https://github.com/Rayoyo/rare-neuro-mri/releases/latest/download/convnext_tiny_finetuned_weights.pth
curl -L -o API_App/models/convnext_tiny_deploy.pkl \https://github.com/Rayoyo/rare-neuro-mri/releases/latest/download/convnext_tiny_deploy.pkl

cd API_App
docker compose up -d --build
```

Stop with `docker compose down`. The first build takes 5–10 minutes (CPU-only PyTorch wheels).

### Using the service

| Service | URL |
|---|---|
| **Web interface** | **http://localhost:8000/** |
| Interactive API documentation (Swagger) | http://localhost:8000/docs |
| OpenAPI schema | http://localhost:8000/openapi.json |

The interface has a sidebar with five views: classify a slice, explain a prediction with Grad-CAM,
inspect the deployed model, browse the five conditions, and read the disclaimer. It calls the same
public endpoints as any other client.

| Method | Endpoint | Returns |
|---|---|---|
| `GET` | `/health` | liveness probe |
| `GET` | `/model-info` | architecture, classes, threshold, test accuracy, ECE |
| `GET` | `/classes` | the five classes with ORPHA codes |
| `POST` | `/predict` | predicted disease, confidence, full distribution, in/out-of-distribution flag |
| `POST` | `/gradcam` | PNG with the Grad-CAM heatmap over the slice |

```bash
curl -F "file=@slice.jpg" http://localhost:8000/predict
curl -F "file=@slice.jpg" "http://localhost:8000/gradcam?target_class=moyamoya_disease" -o cam.png
```

Inputs that do not resemble the training data are flagged with `"in_distribution": false` and a warning:
the model has five outputs and always returns one of them, so the prediction is not meaningful in that
case. The rule is a threshold on the softmax confidence, calibrated to refuse ~5 % of genuine inputs.

## Docker image

Published to the GitHub Container Registry and linked to this repository:

```bash
docker pull ghcr.io/rayoyo/rare-neuro-mri-api:2.1.0    # immutable, reproducible
docker pull ghcr.io/rayoyo/rare-neuro-mri-api:latest   # moving tag
```

---

## How to Run the Code (notebook)

Requires Python 3.11 and a Kaggle API token (in Colab, as the secret `KAGGLE_API_TOKEN`)

The notebooks create the folders they need and download the dataset on first run

```bash
git clone https://github.com/Rayoyo/rare-neuro-mri.git
cd rare-neuro-mri
pip install -r requirements.txt
```

Run the notebooks **in order** — the second reuses the splits and the seed of the first:

```bash
jupyter notebook notebooks/01_resnet50_svm.ipynb     # ~15 min on GPU
jupyter notebook notebooks/02_convnext-tiny_ft.ipynb # ~1 h on GPU (few-shot study included)
```

Set `BASE_DIR` in the first cell of each notebook to the folder where the project should live.
Notebook 02 produces the two artifacts the API needs: `convnext_tiny_finetuned_weights.pth` and
`convnext_tiny_deploy.pkl`.

To read the analysis without re-running anything, open `summary.ipynb`: it renders the saved figures  from `utils/`.

---

> **Disclaimer**
> This is a research demonstrator trained on a small curated dataset. It is not a medical device and its output must not be used for diagnosis






