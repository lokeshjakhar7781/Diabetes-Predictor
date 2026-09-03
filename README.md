# Diabetes Prediction | Machine Learning

<p align="center">
  <b>End-to-End Machine Learning Pipeline for Diabetes Prediction</b>
</p>


---

## Overview

This project implements an **end-to-end machine learning solution for diabetes prediction** using a structured diabetes dataset.

The system automatically preprocesses the data, trains a **Random Forest Classification model**, saves the trained model and preprocessing pipeline, and generates predictions for unseen data.

---

## Key Features

* Automated data preprocessing
* Missing-value handling
* Numerical feature scaling
* Categorical feature encoding
* Stratified train/test splitting
* Random Forest classification
* Model and pipeline persistence
* Automated inference
* CSV-based prediction output

---

## Machine Learning Workflow

```text
                 ┌──────────────────────────┐
                 │  Diabetes Dataset        │
                 │  diabetes_prediction_    │
                 │  dataset.csv             │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Stratified Split       │
                 │     80% Train / 20% Test │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Data Preprocessing   │
                 │                          │
                 │ Numerical → Imputation   │
                 │            → Scaling     │
                 │                          │
                 │ Categorical → Imputation │
                 │            → One-Hot     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Random Forest          │
                 │   Classifier             │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Saved Model & Pipeline │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      input.csv           │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │       Prediction         │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      output.csv          │
                 └──────────────────────────┘
```

---

## Technologies Used

| Technology       | Purpose                      |
| ---------------- | ---------------------------- |
| Random Forest    | Classification model         |
| Pipeline         | Automated preprocessing      |
| Standard Scaler  | Numerical feature scaling    |
| One-Hot Encoding | Categorical feature encoding |
| Joblib           | Model persistence            |
| CSV              | Data input and output        |

---

## Project Structure

```text
Diabetes-Prediction/
│
├── main.py
├── diabetes_prediction_dataset.csv
├── input.csv
├── output.csv
├── model.pkl
├── pipeline.pkl
└── README.md
```

### File Description

**`main.py`**
Contains the complete training and inference workflow.

**`diabetes_prediction_dataset.csv`**
Main dataset used to train the classification model.

**`input.csv`**
Test dataset generated through stratified sampling.

**`output.csv`**
Contains the predicted diabetes results.

**`model.pkl`**
Saved Random Forest classification model.

**`pipeline.pkl`**
Saved preprocessing pipeline.

---

## Preprocessing

### Numerical Features

```text
Missing Values
      ↓
Median Imputation
      ↓
Standard Scaling
```

### Categorical Features

```text
Missing Values
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
```

The preprocessing pipeline is applied consistently during both training and inference.

---

## Model

The project uses a:

```text
Random Forest Classifier
```

The target variable is:

```text
diabetes
```

The model is trained on the preprocessed training data and then saved for future predictions.

---

## Data Splitting

The dataset uses an **80/20 stratified split** with a fixed random state for reproducibility.

The split is stratified using the `HbA1c_level` feature to maintain a representative distribution between training and test data.

---

## Getting Started

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### Run the Project

```bash
python main.py
```

---

## How It Works

### First Run

If `model.pkl` does not exist:

1. Load the diabetes dataset.
2. Create a stratified train/test split.
3. Separate features and target.
4. Build the preprocessing pipeline.
5. Transform the training data.
6. Train the Random Forest Classifier.
7. Save the model and pipeline.

### Subsequent Runs

If the model already exists:

1. Load the saved model.
2. Load the preprocessing pipeline.
3. Read `input.csv`.
4. Transform the input data.
5. Generate diabetes predictions.
6. Save predictions to `output.csv`.

---

## Prediction Output

The generated `output.csv` contains:

```text
diabetes
```

Each row represents the predicted diabetes classification for a corresponding record from the input dataset.

---

## Project Goal

The goal of this project is to demonstrate a complete **classification workflow** that transforms structured health-related data into an automated diabetes prediction system.

```text
Raw Data
   ↓
Stratified Split
   ↓
Preprocessing
   ↓
Feature Transformation
   ↓
Random Forest Classifier
   ↓
Diabetes Prediction
```

---

<p align="center">
  <b>From Data to Intelligent Diabetes Prediction</b>
</p>