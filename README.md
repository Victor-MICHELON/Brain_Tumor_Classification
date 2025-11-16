# 🧠 Brain Tumor MRI Classification with Explainable AI (XAI)

This repository presents an end-to-end deep learning workflow for **multiclass classification of brain tumors from MRI scans**.  
The project compares a **custom CNN baseline** with a **fine-tuned EfficientNetB0 Transfer Learning model**, and emphasizes **interpretability using Grad-CAM** to ensure medically meaningful predictions.

---

## 📖 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Baseline: Custom CNN](#baseline-custom-cnn)
  - [Advanced: Transfer Learning](#advanced-transfer-learning)
- [Results & Performance](#results--performance)
- [Explainability with Grad-CAM](#explainability-with-grad-cam)
- [Limitations & Future Work](#limitations--future-work)
- [Installation](#installation)
- [Technologies](#technologies)
- [License](#license)

---

## 🔭 Overview

In medical imaging, accuracy alone is not sufficient — model decisions must be **transparent and trustworthy**.  
This project achieves **98% test accuracy** on brain tumor MRI classification while providing **explainability guarantees** through XAI techniques.

Grad-CAM heatmaps demonstrate that the model relies on **clinically relevant tumor regions**, not background artifacts.

---

## 🎯 Problem Statement

Build a computer vision model capable of **4-class classification**:

- **glioma**
- **meningioma**
- **pituitary**
- **no tumor**

---

## 🗂️ Dataset

The project uses a public Brain Tumor MRI dataset (e.g., Kaggle), containing thousands of MRI slices organized into four classes.

- **Total images:** ~7,000  
- **Classes:** 4  
- **Splits:** train / validation / test  
- **Preprocessing:** resizing, normalization, data augmentation

---

## 🔬 Methodology

Two modeling strategies were implemented to compare feature learning capabilities:

### **1. Baseline: Custom CNN**

A convolutional neural network built from scratch:

- stacked `Conv2D` + `MaxPooling2D`
- `Dropout` regularization
- `Flatten` + dense classification head
- trained from scratch with Adam optimizer

This serves as a performance benchmark.

---

### **2. Advanced: Transfer Learning (EfficientNetB0)**

A state-of-the-art model using:

- Pretrained **EfficientNetB0** (ImageNet)
- Frozen + partially fine-tuned feature extractor
- Custom classifier:
  - `GlobalAveragePooling2D`
  - Dense + ReLU
  - Dropout
  - Softmax output

Transfer Learning delivers superior generalization and faster convergence.

---

## 📊 Results & Performance

### **Custom CNN**  
**Accuracy:** 91.61%

### **EfficientNetB0 Transfer Learning**  
**Accuracy:** **98.17%**

The Transfer Learning model significantly outperforms the baseline across all metrics.

---

## 💡 Explainability with Grad-CAM

Grad-CAM visualizations validate the model's reasoning by highlighting the regions influencing its predictions.

| Glioma | Meningioma | Pituitary |
|:------:|:----------:|:---------:|
| ![Glioma](YOUR_GLIOMA_GRADCAM_URL) | ![Meningioma](YOUR_MENINGIOMA_GRADCAM_URL) | ![Pituitary](YOUR_PITUITARY_GRADCAM_URL) |
| *Diffuse activation over glioma mass* | *Focus on meningioma contours* | *Localized activation on pituitary tumor* |


These results confirm that the model bases decisions on **meaningful tumor structures**, not dataset biases.

---

## ⚠️ Limitations & Future Work

1. **Classification, not segmentation:**  
   The model predicts tumor categories but does not localize tumors pixel-wise.

2. **Generalization to real-world MRI:**  
   Domain shift from clinical scans may reduce performance. Further work may include domain adaptation or self-supervised learning.

3. **Early-stage tumor detection:**  
   Dataset mostly includes well-defined tumors; performance on subtle cases is unverified.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook "Brain Tumor Classification.ipynb"
