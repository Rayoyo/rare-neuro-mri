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
## Folders structure - - - work in progress

In this section there is a visualization of how the folders are organized:

```bash

rare-neuro-mri-classification/
├── README.md                             # this file with info about the project
├── .gitignore
├── requirements.txt
├── summary.ipynb                         # summary notebook - to add
│
├── data/                                 # to create if you can run the code on your pc (.gitignore) -> must change paths
│   └── rare_neuro_mri_curated/           # download from Kaggle (or by the code itself)
│
├── notebooks/
│   ├── 01_resnet50_svm.ipynb             # ResNet50 + SVM notebook
│   └── 02_convnext-tiny_ft.ipynb         # ConvNext-Tiny + fine tuning notebook
|
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── feature_extractor.py              # ResNet-50 + SVM
│   ├── metrics.py
│   └── transfer_learning.py              # ConvNeXt-Tiny
│                    
├── API_App/              # api folder
│   └── models/                           # copy of the best trained model
│   │   ├──
│   │   ├──
│   │   └── 
│   ├── app.py                            # FastAPI application
│   ├── docker-compose.yml               
│   └── Dockerfile    
|
└── 

```
---

## How to Run the Code

The great part about the notebook's code, is that those folder will be automatically created if there are not present. It is sufficient to set the right folder in whitch you want to keep the project.
- After choosing the right folder, you can firstly run the code in the notebook called 01_resnet50_svm.ipynb
  
- Then you can proceed with the notebook 02_convnext-tiny_ft.ipynb

Final folder structure in drive:
From the folder University -> rare-neuro-mri

```
rare-neuro-mri/
├── data/                                          # dataset downloaded from kaggle
│   └── rare_neuro_mri_curated            
│   │   ├── test                                   # 280 images each
│   │   |   ├── fukuyama_muscular_dystrophy        # contains the images relative to that disease
│   │   |   ├── hallervorden_spatz_disease 
│   │   |   ├── moyamoya_disease 
│   │   |   ├── pachygyria_cerebellar_hypoplasia 
│   │   |   └── walker_warburg_syndrome 
│   │   ├── train                                  # folder similar to test (with 60 images) 
│   │   ├── val                                    # folder similar to test (with 60 images) 
│   │   ├── disease_summary.csv                    # Disease information with clinical details and Orphadata references (5 rows)
│   │   └── metadata.csv                           # complete information for each image (2000 rows)
│
├── resnet models/                                 # resnet model
│   ├── resnet50_svm_artifacts.pkl             
│   ├── resnet50_svm_results.json
│   ├── resnet50_svm_weights.pth
│   ├── x_test.npy
│   ├── x_train.npy
│   ├── x_val.npy
│   ├── y_test.npy
│   ├── y_train.npy
│   └── y_val.npy
|
├── resnet utils/                                   # png generated from resnet code
│   ├── confmat_resnet50_svm.png
│   ├── fewshot_resnet50_svm.png
│   ├── perclass_metrics_resnet50_svm.png              
│   └── sample_grid.png
│                    
├── convnext models/                                # conv next model
│   ├── convnext_tiny_best.pth            
│   ├── convnext_tiny_finetuned_artifacts.pkl
│   ├── convnext_tiny_finetuned_results.json
│   └── convnext_tiny_finetuned_weights.pth
|
├── convnext utils/                                 # png generated from convnext code
│   ├── confmat_convnext_tiny_finetuned.png
│   ├── fewshot_convnext_tiny_finetuned.png
│   ├── perclass_metrics_convnext_tiny_finetuned.png         
│   └── sample_grid.png
└── 

```


---

## How to run the application



