# Shell.ai Hackathon 2025  
## Fuel Blend Properties Prediction Challenge

---

## 🚀 Overview
This repository contains our solution developed for the **Shell.ai Hackathon for Sustainable and Affordable Energy 2025**.  
The challenge focused on predicting the final properties of complex fuel blends using machine learning techniques to enable faster, safer, and more sustainable fuel formulation for next-generation low-carbon fuels such as Sustainable Aviation Fuels (SAFs).

Our team ranked **among the Top 65 teams** in the hackathon.

**Team Members**
- Nikhil Papnai  
- Priyam Sharma  
- Soham Gupta  
- Vishnu Roshan  

---

## 🧩 Problem Statement
Fuel blending is a complex, high-dimensional process where the final properties of a blend depend on non-linear interactions between multiple fuel components and their proportions. Accurate prediction of these properties is essential to ensure safety, performance, regulatory compliance, and sustainability.

The objective of this challenge was to develop machine learning models capable of predicting **10 final blend properties** based on:
- Volume fractions of 5 fuel components, and  
- 50 component-level physical, chemical, and environmental properties.

These predictive models enable rapid evaluation of blend combinations, reduce experimental effort, and accelerate the development of sustainable fuels.

---

## 📊 Dataset Description
The dataset provided for the hackathon consisted of:

### `train.csv`
- 65 columns per sample:
  - 5 columns: Blend composition (volume percentages)
  - 50 columns: Component properties (5 components × 10 properties each)
  - 10 columns: Target blend properties

### `test.csv`
- Contains the same 55 input feature columns as `train.csv`
- Target blend properties are not included and must be predicted

### `sample_submission.csv`
- Defines the required submission format
- Contains 10 columns corresponding to predicted blend properties

> ⚠️ Due to confidentiality and competition rules, raw datasets are not included in this repository.

---

## 🛠️ Methodology
The solution followed a structured machine learning workflow:

1. **Exploratory Data Analysis (EDA)**  
   - Feature distribution analysis  
   - Correlation and interaction analysis  

2. **Preprocessing & Feature Handling**  
   - Numerical feature handling and scaling  
   - Train–validation split (70% / 30%)  

3. **Model Development & Experimentation**  
   Multiple models were implemented and compared to capture both linear and non-linear relationships:
   - Artificial Neural Network (ANN)
   - CatBoost Regressor
   - Random Forest Regressor
   - Decision Tree Regressor
   - LightGBM Regressor

4. **Evaluation & Model Selection**  
   - Models evaluated using Mean Absolute Percentage Error (MAPE)  
   - Tree-based and boosting models demonstrated strong performance on complex feature interactions  

---

## 📈 Models Implemented
- Artificial Neural Network (ANN)
- CatBoost
- Random Forest
- Decision Tree
- LightGBM

---

## 📏 Evaluation Metric
Model performance was evaluated using **Mean Absolute Percentage Error (MAPE)**:

\[
MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
\]

Lower MAPE indicates better predictive accuracy.

---

## 🧪 Results
- Multiple machine learning models were trained and validated
- Ensemble and boosting-based models effectively captured non-linear relationships in fuel blend data
- ANN models were explored to model higher-order feature interactions
- The final selected model was used for leaderboard submission

> Final leaderboard rankings were determined using a private evaluation dataset.

---

## 📁 Repository Structure
```text
├── data/              # Dataset descriptions (no raw data included)
├── notebooks/         # EDA and model experimentation
├── src/               # Training and evaluation scripts
├── models/            # Saved trained models
├── results/           # Plots and metrics
├── requirements.txt   # Python dependencies
└── README.md
