# NeuroNexus
TASK 2- CREDIT CARD FRAUD DETECTION

# 💳 NeuroNexus: Credit Card Fraud Detection

A machine learning project designed to detect fraudulent credit card transactions using classification models. This project tackles real-world issues such as class imbalance and transaction data preprocessing, and it is containerized for reproducibility in GitHub Codespaces.

---

## 📁 Project Structure
---
```bash
.
NeuroNexus/
│
├── data/
│   └── creditcard.csv          # Your dataset
│
├── src/
│   └── preprocess.py           # Preprocessing functions
│   └── model.py                # Model training and evaluation
│
├── main.py                     # Main script to run the pipeline
├── requirements.txt            # Required libraries
├── .gitignore
├── README.md
```

---

## 🔍 Problem Statement

Credit card fraud is a major concern for both consumers and financial institutions. This project aims to:
- Preprocess and normalize transaction data
- Handle class imbalance using techniques like SMOTE
- Build a classification model (Random Forest)
- Evaluate using precision, recall, and F1-score

---

## 📊 Dataset

The dataset used is `creditcard.csv`, which contains transactions made by European cardholders in September 2013. Features include anonymized variables (`V1-V28`), `Time`, `Amount`, and a binary `Class` label indicating fraud (`1`) or not (`0`).

---

## 🚀 Features

- **Data Normalization**: Scales the `Amount` feature and drops the `Time` feature.
- **Imbalance Handling**: Uses `SMOTE` for synthetic oversampling of fraud cases.
- **Model**: Random Forest Classifier for high accuracy and interpretability.
- **Evaluation**: Uses metrics like precision, recall, and F1-score for performance.

---

## 🧪 How to Run

> **Step 1:** Clone the repository  
```bash
git clone https://github.com/yourusername/NeuroNexus.git
cd NeuroNexus
