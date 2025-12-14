# Flood Detection Using Deep Learning & Satellite Imagery
**IndabaX DRC 2025 Workshop**

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8+-green.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

A comprehensive workshop series designed to take you from machine learning fundamentals to deploying AI-powered disaster response systems using Sentinel-1 & 2 satellite imagery.

---

## 📋 Overview

This repository offers a two-step learning path:
1.  **Foundations:** An introduction to deep learning using the PyTorch framework and the FashionMNIST dataset.
2.  **Application:** A deep dive into building a **95%+ accurate flood detection system** using multi-modal satellite data and ensemble learning.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended)
* 15GB+ Free Disk Space

### Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/ganji759/Flood-Prediction-Using-Machine-Learning.git](https://github.com/ganji759/Flood-Prediction-Using-Machine-Learning.git)
    cd Flood-Prediction-Using-Machine-Learning
    ```

2.  **Create environment**:
    ```bash
    python -m venv flood_env
    source flood_env/bin/activate  # Windows: flood_env\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    # Install PyTorch (CUDA 11.8)
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    
    # Install workshop libraries
    pip install pytorch-lightning transformers kagglehub rasterio opencv-python pandas matplotlib seaborn scikit-learn xgboost thop
    ```

4.  **Configure Data Access**:
    * Download `kaggle.json` from your [Kaggle Account Settings](https://www.kaggle.com/settings/account).
    * Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<User>\.kaggle\` (Windows).

---

## 📦 The Notebooks

Run `jupyter notebook` and select the file matching your goal:

| Notebook File | Focus | Description |
| :--- | :--- | :--- |
| **`Workshop Session 1.ipynb`** | **Foundations** | **Start Here.** Based on official PyTorch tutorials. Introduces ML concepts, tensor operations, and image classification using the **FashionMNIST** dataset. |
| **`Workshop DAY 1.ipynb`** | **Application** | **Deep Dive.** Focuses on disaster risk monitoring. Covers processing **Sentinel-1 (SAR) & Sentinel-2 (Optical)** data, training ensembles (ResNet, EfficientNet), and evaluating the **SEN12FLOOD** dataset. |

---

## 🔬 Methodology (Flood Prediction)

In `Workshop DAY 1.ipynb`, we develop a robust classifier using the following techniques:

* **Dataset:** [SEN12FLOOD](https://www.kaggle.com/datasets/rhythmroy/sen12flood-flood-detection-dataset) (~10,000 satellite chips).
* **Preprocessing:** Speckle noise filtering (SAR), CLAHE, and Percentile stretching.
* **Models:** ResNet-50, DenseNet-121, EfficientNet-B0, Vision Transformer (ViT).
* **Ensembling:** Stacking (Logistic Regression meta-learner), Hard Voting, and Soft Voting.
* **Performance:** Achieved **95.6% accuracy** (Stacking Ensemble).

---

## 📚 References & Resources

### Frameworks & Tools
* [PyTorch](https://pytorch.org/) - Official documentation and tutorials.
* [Copernicus Open Data Hub](https://scihub.copernicus.eu/) - Access point for Sentinel satellite data.

### Datasets
* [Kaggle SEN12FLOOD](https://www.kaggle.com/datasets/rhythmroy/sen12flood-flood-detection-dataset) - Flood detection dataset (SAR + Optical).
* [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) - Benchmarking dataset used in the intro session.

### Satellite Missions
* [ESA Sentinel-1 Mission](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1) - Synthetic Aperture Radar (SAR) imagery specifics.
* [ESA Sentinel-2 Mission](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2) - Optical multispectral imagery specifics.

### Key Papers
* [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
* [SEN12FLOOD: A SAR and Multispectral Dataset for Flood Detection](https://arxiv.org/abs/2104.03704)
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

---

## 📝 License

This project is licensed under the **MIT License**.

Happy Learning! 🚀