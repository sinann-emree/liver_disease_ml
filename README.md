# 🩺 Liver Disease Prediction: Comprehensive ML Pipeline & UI

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-ANN-FF6F00?style=for-the-badge)
![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare_AI-00A651?style=for-the-badge)

An end-to-end Machine Learning and Artificial Neural Network (ANN) project aimed at predicting the presence of liver disease using patient medical data. This project showcases a full data science lifecycle, from Exploratory Data Analysis (EDA) and feature engineering to rigorous model evaluation and a user-friendly frontend interface.

## ✨ Key Features & Methodology

* **Robust Data Engineering:** Includes dedicated scripts for data exploration (`veri_inceleme.py`) and preprocessing/cleaning (`veri_onisleme.py`) to handle missing values and scale features (`scaler.joblib`).
* **Multi-Model Evaluation:** Trains and evaluates multiple algorithms, including traditional ML models and Artificial Neural Networks (ANN). Model performance is visualized through training graphs (`ysa_egitim_grafigi.png`).
* **Advanced Validation Metrics:** * Implements both **Holdout** and **Cross-Validation (CV)** methodologies to ensure model generalization.
  * Evaluates performance using statistical tools like **ROC Curves** and **Confusion Matrices**.
* **Explainable AI (XAI):** Analyzes and visualizes **Feature Importance** (`ozellik_onemi.py`) to interpret which medical metrics contribute most to the liver disease prediction, providing transparency to the AI's decisions.
* **Interactive Interface:** Deploys the best-performing model (`en_iyi_model.joblib`) into a functional user interface (`karaciger_arayuz.py`), allowing real-time clinical predictions.

## 📂 Project Structure

* **Data:** `Liver_disease_data.csv` (Raw) and `Liver_disease_data_processed.csv` (Cleaned).
* **Core Scripts:**
  * `veri_inceleme.py` & `veri_onisleme.py`: EDA and preprocessing.
  * `egitim_analiz.py`: Model training, validation, and performance analysis.
  * `scaler_kayit.py` & `ozellik_onemi.py`: Feature scaling and importance extraction.
  * `karaciger_arayuz.py`: The frontend user interface for the model.
* **Outputs & Metrics:**
  * `.png` files containing ROC curves, confusion matrices, feature importance, and ANN training histories.
  * `.csv` files detailing holdout and cross-validation performance metrics.
* **Saved Models:** `.joblib` files for the trained scaler and the optimized prediction model.

## 🚀 Getting Started

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/liver_disease_ml.git](https://github.com/YOUR_USERNAME/liver_disease_ml.git)
   cd liver_disease_ml
