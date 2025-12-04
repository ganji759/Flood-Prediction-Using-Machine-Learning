# Flood Detection Using Deep Learning & Satellite Imagery

**IndabaX DRC 2025 Workshop** - A comprehensive guide to building and deploying AI-powered flood detection systems using Sentinel-1 satellite imagery.

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8+-green.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

---

## 📋 Overview

This workshop provides a hands-on introduction to disaster risk monitoring through deep learning. You'll learn to:

- **Process satellite data** from Copernicus Sentinel-1 (SAR) and Sentinel-2 (optical)
- **Build & train models** using transfer learning (ResNet-50, DenseNet-121, EfficientNet-B0, Vision Transformer)
- **Ensemble predictions** with hard voting, soft voting, and stacking techniques
- **Deploy models** efficiently on GPU with mixed precision training
- **Evaluate performance** using bootstrap confidence intervals and ablation studies

**Key Achievement**: Develop a **95%+ accurate flood detection classifier** combining multiple deep learning architectures.

---

## 🎯 Learning Objectives

By completing this workshop, you will understand:

1. **Disaster Risk Monitoring**: How satellite imagery enables rapid disaster response
2. **Earth Observation**: Differences between SAR (Sentinel-1) and optical (Sentinel-2) data
3. **Computer Vision**: Image preprocessing, enhancement, and deep learning classification
4. **Model Development**: Transfer learning, hyperparameter tuning, and evaluation
5. **Ensemble Methods**: Combining multiple models to improve robustness
6. **Production Deployment**: Model profiling, optimization, and real-world considerations

---

## 📦 What's Included

### Notebooks

- **`Workshop DAY 1.ipynb`** - Complete end-to-end pipeline including:
  - Dataset download and exploration (SEN12FLOOD from Kaggle)
  - Satellite image preprocessing (SAR speckle reduction, CLAHE enhancement)
  - Multiple model architectures (CNN, ViT)
  - Ensemble techniques (voting, stacking, CNN aggregators)
  - Comprehensive evaluation and ablation studies
---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (CPU mode available but slower)
- 15GB+ free disk space (for models and datasets)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ganji759/Flood-Prediction-Using-Machine-Learning.git
cd Flood-Prediction-Using-Machine-Learning
```

2. **Create virtual environment** (recommended):
```bash
python -m venv flood_env
source flood_env/bin/activate  # Linux/Mac
# or
flood_env\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning transformers kagglehub rasterio opencv-python pandas matplotlib seaborn scikit-learn xgboost thop
```

4. **Configure Kaggle API**:
   - Download `kaggle.json` from [Kaggle Account Settings → API](https://www.kaggle.com/settings/account)
   - Place in `~/.kaggle/kaggle.json` (or follow `KAGGLE_SETUP_GUIDE.md` for Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

### Running the Workshop

```bash
jupyter notebook "Workshop DAY 1.ipynb"
```

Execute cells sequentially from top to bottom. Each section includes educational markdown explaining theory and implementation.

---

## 📊 Dataset

**SEN12FLOOD** - Publicly available flood detection dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/rhythmroy/sen12flood-flood-detection-dataset)
- **Size**: ~10,000+ satellite scenes
- **Sensors**: 
  - Sentinel-1: C-band SAR (VV, VH polarizations)
  - Sentinel-2: 11 multispectral bands (RGB, NIR, SWIR)
- **Labels**: Binary (Flood / Non-Flood)
- **Resolution**: 10m per pixel
- **Coverage**: Global flood events

**Data Split**: 70% training, 15% validation, 15% testing

---

## 🔬 Methodology

### Architecture Comparison

| Model | Parameters | FLOPs | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| **ResNet-50** | 25M | 8G | 94.6% | Baseline CNN |
| **DenseNet-121** | 7M | 3G | 94.9% | Efficient CNN |
| **EfficientNet-B0** | 5M | 0.4G | 95.1% | Mobile deployment |
| **ViT-B/16** | 86M | 17G | 93.5% | Transformer-based |
| **Soft Voting** | - | - | 95.5% | Best ensemble |
| **Stacking (LR)** | - | - | 95.6% | Meta-learning |

### Key Techniques

1. **Data Preprocessing**
   - Speckle noise reduction (median filtering for SAR)
   - Percentile stretching (2-98%)
   - CLAHE for local contrast enhancement
   - Resizing to 224×224

2. **Model Training**
   - Transfer learning from ImageNet
   - Mixed precision training (FP16/FP32)
   - Early stopping with validation monitoring
   - AdamW optimizer with weight decay

3. **Ensemble Methods**
   - Hard voting (majority vote)
   - Soft voting (probability averaging)
   - Stacking with meta-models (Logistic Regression, XGBoost, SVM)
   - CNN-based aggregators on intermediate features

4. **Evaluation**
   - Bootstrap confidence intervals (95% CI)
   - Sequential ablation studies
   - Model profiling (parameters, FLOPs)
   - Latency analysis (GPU vs CPU)

---

## 📈 Results Summary

### Individual Models
- ResNet-50: **94.6%** accuracy
- DenseNet-121: **94.9%** accuracy
- EfficientNet-B0: **95.1%** accuracy (best single model)
- Vision Transformer: **93.5%** accuracy

### Ensemble Methods
- Hard Voting: **95.4%** accuracy
- Soft Voting: **95.5%** accuracy
- Stacking (Logistic Regression): **95.6%** accuracy **Best overall**

### Performance by Metric
- Accuracy: 95.6% (overall correctness)
- Precision: 95.8% (false positive rate)
- Recall: 95.6% (false negative rate - critical for disaster response)
- F1-Score: 95.5% (harmonic mean)

**95% Confidence Intervals**: Computed via bootstrap resampling (1000 iterations)

---

## 🛠 Usage Guide

### For Beginners
1. Start with "GPU Verification" to check your setup
2. Run dataset download cells
3. Execute visualization cells to understand satellite data
4. Train a single model (e.g., ResNet-50)
5. Review metrics and confusion matrices

### For Intermediate Users
1. Compare multiple model architectures
2. Experiment with hyperparameters
3. Implement custom preprocessing
4. Modify data splits

### For Advanced Users
1. Implement custom ensemble methods
2. Add new model architectures from `timm`
3. Optimize for edge deployment (quantization, pruning)
4. Extend to multi-label or segmentation tasks

---

## 🔍 Notebook Structure

```
Workshop DAY 1.ipynb
├── 00 - Introduction
│   └── GPU verification & library imports
├── 01 - Data Preprocessing
│   ├── Dataset download (Kaggle)
│   ├── Metadata parsing
│   ├── Sensor classification
│   ├── Image enhancement
│   └── PyTorch dataset pipeline
├── 02 - Model Training
│   ├── ResNet-50
│   ├── DenseNet-121
│   ├── EfficientNet-B0
│   └── Vision Transformer
├── 03 - Ensemble Methods
│   ├── Hard/Soft voting
│   ├── Stacking (LR, XGBoost, SVM)
│   └── CNN aggregators
└── 04 - Evaluation & Analysis
    ├── Model profiling
    ├── Latency analysis
    ├── Bootstrap confidence intervals
    └── Ablation studies
```

---

## 📚 Theoretical Background

### Satellite Data Types

**Sentinel-1 (SAR)**
- All-weather capability (works day/night, through clouds)
- Water appears dark (specular reflection)
- Ideal for flood detection
- Reference: [ESA Sentinel-1 Mission](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1)

**Sentinel-2 (Optical)**
- 11 multispectral bands
- Excellent for vegetation and land use mapping
- Limited by cloud cover
- Reference: [ESA Sentinel-2 Mission](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)

### Deep Learning Concepts

- **Transfer Learning**: Leverage ImageNet pretraining to reduce data requirements
- **Mixed Precision**: Train with FP16 for 2-4× speedup while maintaining FP32 numerical stability
- **Ensemble Methods**: Combine predictions from multiple models for improved robustness
- **Stacking**: Train meta-model to learn optimal combination of base models

---

## 🚨 Real-World Applications

This system is designed for:

1. **Rapid Disaster Response**: Automated flood detection for emergency alerts
2. **Impact Assessment**: Quantify flooded areas for humanitarian response
3. **Climate Monitoring**: Track flood patterns over time
4. **Insurance & Risk**: Assess flood risk for property valuation
5. **Urban Planning**: Identify flood-prone regions

**Example**: During the 2025 Kinshasa floods, similar systems provided real-time impact maps to UNOSAT for emergency coordination.

---

## ⚙️ Advanced Topics

### Model Optimization
- Quantization (INT8) for 2× speedup
- Knowledge distillation to smaller models
- Pruning to reduce parameters
- Batch normalization folding

### Deployment
- TensorRT optimization
- ONNX model export
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)

### Extensions
- Multi-label classification (flood type)
- Semantic segmentation (pixel-level masks)
- Temporal modeling (flood progression)
- Uncertainty quantification

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

---

## 📖 References

### Papers
- [ResNet (2015)](https://arxiv.org/abs/1512.03385) - Deep Residual Learning
- [DenseNet (2017)](https://arxiv.org/abs/1608.06993) - Dense Connections
- [EfficientNet (2019)](https://arxiv.org/abs/1905.11946) - Compound Scaling
- [Vision Transformer (2021)](https://arxiv.org/abs/2010.11929) - Attention for Vision
- [Ensemble Methods Survey](https://ieeexplore.ieee.org/document/5128974)

### Resources
- [Copernicus Open Data Hub](https://scihub.copernicus.eu/) - Free satellite data
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework
- [QGIS](https://www.qgis.org/) - Open-source GIS for spatial analysis
- [SEN12FLOOD Dataset Paper](https://arxiv.org/abs/2104.03704)

### Organizations
- [UNOSAT](https://www.unitar.org/unosat/) - UN Satellite Centre
- [ESA](https://www.esa.int/) - European Space Agency
- [UNDRR](https://www.undrr.org/) - UN Disaster Risk Reduction

---

## 📝 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

**Happy Learning!** 🚀 Use this knowledge to tackle climate challenges and disaster risk reduction in your region.
