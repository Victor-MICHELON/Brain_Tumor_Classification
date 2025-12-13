# 🧠 Brain Tumor MRI Classification with Explainable AI (XAI)

This repository presents an end-to-end deep learning workflow for **multiclass classification of brain tumors from MRI scans**.  
The project compares a **custom CNN baseline** with a **fine-tuned EfficientNetB0 Transfer Learning model**, and emphasizes **interpretability using Grad-CAM** to ensure medically meaningful predictions.

---

## 🔭 Overview

In medical imaging, accuracy alone is not sufficient — model decisions must be **transparent and trustworthy**.  
This project achieves **98% test accuracy** on brain tumor MRI classification while providing **explainability guarantees** through XAI techniques.

---

## Problem Statement 

Brain tumors present differently across MRI scans, and early detection significantly improves patient outcomes. This project aims to build an intelligent system capable of recognizing four major diagnostic categories—glioma, meningioma, pituitary tumors, and healthy (no tumor).
By leveraging end-to-end data preprocessing, optimized data augmentation strategies, and the development of both a custom CNN and a transfer-learning model based on EfficientNetB0, the system learns to identify subtle patterns in MRI images that may not be visible to the naked eye. Grad-CAM heatmaps highlight the regions that most influence the prediction, making the model’s decisions more interpretable. The goal is to bring machine learning closer to real-world medical decision-support systems.

---

## Dataset

[Brain Tumor Dataset](https://www.kaggle.com/datasets/indk214/brain-tumor-dataset-segmentation-and-classification) containing labeled brain MRI images grouped into four main categories. Images present variations in acquisition quality, brightness, orientation, and anatomical details

- **Total MRI scans:** 5712 (Train set) & 1311 (Test set)
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor
- Data type: Grayscale or RGB MRI axial slices (JPG/PNG)
- **Splits:** train / test
  
---

## Methodology

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

## Results & Performance

### **Custom CNN**  
**Accuracy:** 91.61%

### **EfficientNetB0 Transfer Learning**  
**Accuracy:** **98.17%**

The Transfer Learning model significantly outperforms the baseline across all metrics.

---

## Explainability with Grad-CAM

Grad-CAM visualizations validate the model's reasoning by highlighting the regions influencing its predictions.

<table>
  <tr>
    <th>Glioma</th>
    <th>Meningioma</th>
    <th>Pituitary</th>
    <th>No Tumor</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Victor-MICHELON/Brain_Tumor_Classification/blob/main/images/gioma.png" width="200"></td>
    <td><img src="https://github.com/Victor-MICHELON/Brain_Tumor_Classification/blob/main/images/meningiomia.png" width="200"></td>
    <td><img src="https://github.com/Victor-MICHELON/Brain_Tumor_Classification/blob/main/images/pituitary.png" width="200"></td>
    <td><img src="https://github.com/Victor-MICHELON/Brain_Tumor_Classification/blob/main/images/notumor.png" width="200"></td>
  </tr>
</table>



These results confirm that the model bases decisions on **meaningful tumor structures**, not dataset biases.

---

## Limitations & Future Work

1. **Classification, not segmentation:**  
   The model predicts tumor categories but does not localize tumors pixel-wise.

2. **Generalization to real-world MRI:**  
   Domain shift from clinical scans may reduce performance. Further work may include domain adaptation or self-supervised learning.

3. **Early-stage tumor detection:**  
   Dataset mostly includes well-defined tumors; performance on subtle cases is unverified.

---

## Technologies

Python, NumPy, Pandas

TensorFlow / Keras

EfficientNetB0

Scikit-learn

Matplotlib, Seaborn

Grad-CAM (custom implementation)

---

## 👤 Author

**Victor MICHELON**  
Data Science & Engineering Student  
🔗 [Portfolio](https://victor-michelon.github.io/Victor_MICHELON_Portfolio.github.io/)

