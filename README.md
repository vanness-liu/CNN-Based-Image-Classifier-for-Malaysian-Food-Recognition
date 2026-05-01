# CNN-Based Image Classifier for Malaysian Product Recognition 🍛

**Course:** WID3011 - Deep Learning (Universiti Malaya)  
**Author:** Vanness Liu Chuen Wei  

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

## 📈 Key Results
| Metric | Custom CNN | ResNet-50 |
| :--- | :--- | :--- |
| **Final Test Accuracy** | 73.76% | **92.90%** |
| **Convergence Efficiency**| 25 epochs | **High accuracy within 1st epoch** |