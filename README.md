
# EpiCare - Skin Lesion Classification (Kaggle Notebook)

This repository contains the Kaggle-based development and experimentation notebook for the **EpiCare mobile skin cancer screening project**.

The notebook demonstrates how a **hybrid CNN-ViT model** is trained and optimized for real-time, on-device lesion classification. The final model is exported to **TensorFlow Lite (TFLite)** for integration into the mobile app.

Models link : https://drive.google.com/drive/folders/1gbP5lf_kbB309HzMhxegYBKdf_kPtKeu?usp=sharing
---

## Notebook Objectives

- Load and preprocess the ISIC skin lesion dataset
- Build a hybrid model using:
  - VGG16 (CNN) for feature extraction
  - Transformer encoder layers for global context
- Train and evaluate the model (accuracy, AUC)
- Convert to `.tflite` format using TensorFlow Lite
- Save key artifacts: model, confusion matrix, plots

---

## Files in This Repository

| File | Description |
|------|-------------|
| `epicare_cnnvit_notebook.ipynb` | Main Kaggle notebook with model training, plots, and conversion |
| `cnn_vit_model.tflite` | Final mobile-ready quantized model |
| `*.png` | Visuals such as loss curves, and confusion matrices |

---

## 🚀 Getting Started

### Run in Kaggle or Colab

You can open and run the notebook in:

- **Kaggle**: https://www.kaggle.com/code/teddynoughton/vgg16-vit-for-skin-cancer-classification-binary

---

## 🧪 Results Summary

- **Final Accuracy**: ~81.1%
- **AUC Score**: ~0.901
- **Model Format**: TensorFlow Lite (FP16)
- **On-device Inference**: < 3 seconds

---

## 📄 License

This repository is open-source under the **MIT License**. You may use, modify, or distribute the notebook with credit to the author.

---
  
