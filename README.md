# CNN-Based Image Classifier for Malaysian Food Recognition 🍛

**Course:** WID3011 - Deep Learning
**Author:** Vanness Liu Chuen Wei  <br>
**Matric Number:** 23005021 <br>
**Occ:** 1

## 📌 Project Overview
This project implements a Convolutional Neural Network (CNN) pipeline to classify 6 categories of traditional Malaysian food/products. The objective is to evaluate the feature extraction capabilities of a scratch-built 4-layer CNN against a fine-tuned, pre-trained **ResNet-50** architecture, simulating a commercial deployment scenario for Malaysian SMEs (e.g., Automated Smart Billing Kiosks).

## 📊 Dataset
Due to GitHub storage limits, the dataset is not included in this repository. 
* **Source:**[Malaysia Food 11 Dataset (Kaggle)](https://www.kaggle.com/datasets/karkengchan/malaysia-food-11)
* **Categories Used:** Nasi Lemak, Popiah, Laksa, Satay, Roti Canai, Kaya Toast.
* **Pre-processing:** The dataset underwent a programmatic sanitization pass using PIL to remove corrupted images, followed by Majority Undersampling to achieve a perfectly balanced parity of 999 images per class.

## 🧠 Model Architectures & Weights
1. **Custom CNN (Baseline):** A VGG-style 4-block CNN built from scratch. (Receptive Field: 46x46).
2. **ResNet-50 (Champion):** Transfer learning approach utilizing ImageNet priors, followed by fine-tuning of the final residual block (`layer4`).

**Download Pre-trained Weights:**
Because `.pth` files exceed GitHub's 100MB limit, you can download the trained model weights here:[models](https://drive.google.com/drive/folders/1_Qqs3UeqxRPTQY7L0RCLYoXaVqnyKTfX?usp=share_link)

## 📈 Key Results
| Model | Val Acc | Test Acc | Train Time |
|---|---|---|---|
| Custom CNN (Scratch) | 73.23% | 73.76% | 23.37 min |
| ResNet-50 (Transfer Learning) | **93.10%** | **92.90%** | 47.09 min |

## Setup

```bash
pip install torch torchvision torchsummary pandas numpy matplotlib seaborn scikit-learn split-folders pillow
```

## Usage

Open `jupyter_notebook_code.ipynb` and run cells in order:

- **Part A** — Dataset sanitization, perfect-parity undersampling, split, and optimal food-specific data augmentation.
- **Part B** — Custom CNN architecture design, receptive field analysis, and model training.
- **Part C** — ResNet-50 two-stage fine-tuning and comparative evaluation.
- **Part D** — Confusion matrix generation and qualitative analysis of misclassified samples.

- ## Custom CNN Architecture

Four sequential convolutional blocks (32→64→128→256 channels). Each block features a `3x3` Conv2d layer (stride 1, padding 1), ReLU activation, and a `2x2` MaxPool2d (stride 2). The spatial features are flattened into a Fully Connected head (50,176→512→6) with a 50% Dropout layer.

- **Total parameters:** 26,082,118
- **Final receptive field:** 46×46px (Optimal for recognizing regional food textures like flakiness or skewers).

---

## ResNet-50 Fine-Tuning

**Stage 1 (Feature Extraction):** Freeze the backbone, replace the FC head with `Dropout(0.5) → Linear(6)`, and train using `lr=1e-3`.
**Stage 2 (Fine-Tuning):** Unfreeze the final residual block (`layer4`) to adapt to specific Malaysian food textures, continuing training with a lower `lr=1e-5`.

---

## Training Config

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 (Custom CNN & Stage 1), 1e-5 (Stage 2) |
| Regularization | Early Stopping (Patience=4), Dropout (p=0.5) |
| Batch size | 32 |
| Loss Function | CrossEntropyLoss |
| Hardware Accelerator| Apple Silicon (MPS) |
```
