# Automating Diabetic Retinopathy Screening with Deep Learning

This project compares two deep learning approaches for automated diabetic retinopathy (DR) severity grading from color fundus images: a **ResNet-50 multi-class classification baseline** and a **DenseNet-121 ordinal regression model**.


## Problem Statement

Diabetic retinopathy is a leading cause of preventable blindness, affecting roughly a third of diabetic patients worldwide. Manual screening is time-consuming and access to specialists is limited, especially in underserved regions. This project builds an automated pipeline to grade DR severity (0–4) from retinal fundus images, with a particular focus on minimizing dangerous *under-grading* — cases where a severe stage is mistakenly predicted as healthy.

Rather than treating DR grading as a flat 5-class classification problem, we explicitly model it as an **ordinal progression**, since some errors (e.g., predicting grade 0 when the true grade is 4) are far more clinically dangerous than others (e.g., grade 2 vs. 3).

## Dataset

- **EyePACS 2015** and **APTOS 2019** (combined), sourced from Kaggle
- ~91,898 labeled color fundus images across DR grades 0–4
- Patient-level stratified split: 70% train / 15% validation / 15% test, to prevent subject leakage
- Class distribution is highly imbalanced (far more no/mild DR images than severe/proliferative cases)

## Preprocessing

- Circular cropping to isolate the retinal field and remove black borders
- Ben Graham-style preprocessing (green-channel enhancement with Gaussian blur subtraction) to reduce illumination variation and enhance lesion visibility
- Resizing to 224×224 (ResNet-50) or 512×512 (DenseNet-121)
- Online augmentation: horizontal/vertical flips, small rotations (±15°), zoom, brightness/contrast jitter

## Models

### ResNet-50 (Baseline — 5-class classification)
- Input: 224×224 RGB, ImageNet-pretrained backbone
- Head: global average pooling → dropout (0.4) → 5-unit softmax
- Loss: categorical cross-entropy
- Two-phase training: backbone frozen initially, then top residual blocks unfrozen and fine-tuned at a lower learning rate
- ~23.6M parameters

### DenseNet-121 (Proposed — ordinal regression)
- Input: 512×512 RGB, to preserve small lesions
- Head: global average pooling → dropout (0.5) → single linear neuron outputting a continuous severity score
- Loss: MSE on the continuous score (MAE tracked as an auxiliary metric)
- Decision thresholds optimized post-training on the validation set to convert continuous scores into discrete grades (0–4)
- Mixed-precision training on an NVIDIA A100 GPU
- ~7.9M parameters (~66% fewer than ResNet-50)

Both models were trained with Adam, `ReduceLROnPlateau`, early stopping, and checkpointing.

## Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)** — primary metric, penalizes large ordinal disagreements more heavily than small ones
- Overall accuracy and full confusion matrix
- Class-wise recall and F1-score, with emphasis on referable DR (grades 2–4)

## Results

| Aspect | ResNet-50 | DenseNet-121 |
|---|---|---|
| Parameters | ~23.6M | ~7.9M |
| Formulation | 5-class softmax | Ordinal regression |
| Input size | 224×224 | 512×512 |
| QWK (test) | 0.62–0.68 | 0.78–0.79 |
| Referable DR focus | Indirect | Explicit thresholds |
| Test accuracy | ~0.79 | — |

DenseNet-121 achieved substantially higher QWK and better recall on referable DR cases, while using roughly a third of ResNet-50's parameters — making it a strong candidate for scalable or edge deployment in resource-limited screening settings.



## Technologies Used

- **Framework:** TensorFlow / Keras
- **Language:** Python
- **Libraries:** NumPy, scikit-learn, Matplotlib, Seaborn
- **Hardware:** NVIDIA A100 (40GB), mixed-precision training

## Repository Structure

```
├── README.md
├── requirements.txt
├── RetinAI_Resnet_50.ipynb              # ResNet-50 baseline pipeline
├── RetinAI_Densenet_121.ipynb        # DenseNet-121 ordinal regression pipeline
```

## Setup

```bash
pip install -r requirements.txt
```

GPU training requires a CUDA-compatible TensorFlow build matching your CUDA/cuDNN version. Training was performed on an NVIDIA A100 (40GB); CPU-only execution will be significantly slower given the dataset size.

## Limitations & Future Work

As noted in our report: training data is drawn from a limited set of public datasets, class imbalance persists despite mitigation efforts, and residual errors continue to occur on borderline or low-quality images. This system is intended as an **assistive triage tool** to support ophthalmologists, not a replacement for clinical judgment. Future work includes more advanced imbalance-handling techniques (e.g., focal loss, dynamic sampling), calibration methods, and broader real-world evaluation across diverse populations.

## Author

**Althaf Rasheed Abubakkar**
M.Eng. Artificial Intelligence for Smart Sensors/Actuators, B.E. Mechatronics
[GitHub](https://github.com/Althaf568)
