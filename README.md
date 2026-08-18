# Rare Neurological Diseases MRI Classification
_Feature Extraction vs Transfer Learning under Limited Data Regime_

## Project Description

Rare neurological diseases often suffer from extremely long diagnostic delays (the so-called “diagnostic odyssey”), averaging 5–7 years. One of the main causes is the scarcity of high-quality annotated imaging data. In this project we investigate which deep-learning strategy is more robust when only a limited number of MRI examples per class is available.

Were compared two complementary approaches, both based on strong ImageNet-pretrained backbones:

1. **Classical Feature Extraction** – _ResNet-50_ used for feature extraction + _linear SVM_ as a classifier
2. **Transfer Learning** – _ConvNeXt-Tiny_, a newest model, fine-tuned

The evaluation is performed both on the full balanced dataset and on progressively reduced training sets (few-shot scenarios) in order to quantify robustness under data scarcity. The best-performing model is then used into a production-ready FastAPI service that:
- returns the predicted rare disease,
- the associated confidence score, and
- the full probability distribution over the five classes
An optional Grad-CAM explanation endpoint is also provided.

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

---

## How to Run the Code
