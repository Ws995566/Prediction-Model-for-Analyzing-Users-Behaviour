# 🛒 Prediction Model for Analyzing Users Behaviour in E-Commerce

## Overview

This project was developed to predict whether an online shopper is likely to make a purchase based on their browsing behavior during a session. The prediction is generated using machine learning models trained on the Online Shoppers Purchasing Intention dataset.

In addition to model development, this project also includes a Streamlit web application that allows users to explore the dataset, perform preprocessing, train models, evaluate performance, and generate purchase intention predictions through an interactive interface.

## Team Members

**Group 1 - LC01**

* Edwin Antonie (2802397306)
* Maximilianus Ronald (2802391006)
* Wesley Sumedha Deano (2802401846)

---

## Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

The dataset contains information about user browsing sessions on an e-commerce website, including page visits, session duration, exit rates, page values, visitor type, and several other behavioral attributes.

**Target Variable:** `Revenue`

* TRUE → User completed a purchase
* FALSE → User did not complete a purchase

The dataset is imbalanced, with approximately 84.5% non-purchasing sessions and 15.5% purchasing sessions. SMOTE was applied during preprocessing to reduce this imbalance.

---

## Application Features

The Streamlit application provides several functionalities:

### 1. Dataset Overview

* Display dataset information
* View feature descriptions
* Inspect data samples

### 2. Exploratory Data Analysis (EDA)

* Univariate analysis
* Bivariate analysis
* Correlation analysis
* Outlier visualization

### 3. Feature Selection

* Select numerical features
* Select categorical features

### 4. Data Preprocessing

* Categorical encoding
* Outlier capping
* Yeo-Johnson transformation
* Train-test split
* Feature scaling
* SMOTE oversampling

### 5. Model Training

Supported models:

* Logistic Regression
* Random Forest
* XGBoost

Users can configure several model parameters directly from the application.

### 6. Model Evaluation

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

### 7. Purchase Intention Prediction

Users can input session information and obtain real-time purchase intention predictions from the trained model.

---

## Project Structure

```text
├── app.py
├── online_shoppers_intention.csv
├── requirements.txt
├── preprocessing.ipynb
├── training.ipynb
├── evaluation.ipynb
├── model/
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   └── model_xgboost.pkl
└── README.md
```

### Notebook Description

| File                | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| preprocessing.ipynb | Data cleaning, EDA, feature engineering, and preprocessing |
| training.ipynb      | Model training and model export                            |
| evaluation.ipynb    | Model evaluation and performance analysis                  |

---

## Model Performance

Model performance was evaluated using Precision, Recall, F1-Score, and ROC-AUC.

| Model               | ROC-AUC Score |
| ------------------- | ------------- |
| Logistic Regression | 0.9134        |
| Random Forest       | 0.8949        |
| XGBoost             | 0.8824        |

Among the tested models, Logistic Regression achieved the highest ROC-AUC score and provided the most balanced performance.

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd Prediction-Model-for-Analyzing-Users-Behaviour
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Matplotlib
* Seaborn