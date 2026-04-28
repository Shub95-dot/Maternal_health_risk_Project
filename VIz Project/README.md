# Maternal Health Risk Prediction & Analytics

**VitaMaternal Analytics - Project Pipeline**

This repository contains an end-to-end data analytics and machine learning pipeline developed to predict maternal health risks based on clinical and physiological data. It is designed to take raw maternal health readings, engineer predictive clinical metrics, handle class imbalances, and automatically train and evaluate a suite of machine learning classifiers.

---

## 🎯 Project Overview

Healthcare professionals need reliable methods to categorize expectant mothers into low, mid, or high-risk categories to provide timely and personalized interventions. This project implements predictive modeling to achieve this by:
1. **Clinical Feature Engineering:** Calculating established medical indicators like Mean Arterial Pressure (MAP), Pulse Pressure, and Shock Index to provide the models with multivariate clinical context.
2. **Handling Class Imbalance:** Using Synthetic Minority Over-sampling Technique (SMOTE) to ensure the models do not become biased toward the majority class.
3. **Automated Pipeline Execution:** Centralizing preprocessing and modeling code into scalable, modular Python scripts to ensure reproducibility and stability.

## 📂 Repository Structure

The project has been refactored from experimental notebooks into a professional, modular architecture:

```text
├── data/                               # Contains raw and processed datasets
│   └── Maternal Health Risk Data Set.csv
├── models/                             # Serialized champion models (.pkl)
├── notebooks/                          # Jupyter Notebooks for exploration
│   └── eda_main_data.ipynb             # Exploratory Data Analysis (EDA) and Visualization
├── src/                                # Core pipeline modules
│   ├── data_preprocessing.py           # Feature engineering and SMOTE integration
│   └── model_training.py               # Model definitions (Tree-based, Ensembles)
├── run_pipeline.py                     # Master execution script for the pipeline
├── requirements.txt                    # Project dependencies
└── README.md                           # This documentation file
```

## 📊 Exploratory Data Analysis (EDA)

The `notebooks/eda_main_data.ipynb` notebook contains premium Seaborn visualizations that explore the dataset's characteristics and the impact of our feature engineering:
* **Class Distribution:** Visualizing the imbalance in the original risk categories.
* **Comparative Boxplots:** Analyzing the physiological trends (e.g., Blood Sugar, Systolic BP, Age) across different risk levels.
* **Correlation Analysis:** Heatmaps demonstrating how our engineered clinical features (like the `CombinedRiskScore`) improve the predictive signal compared to raw, isolated metrics.

## 🚀 How to Run the Pipeline

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the master script to load the data, engineer features, apply SMOTE, and train the suite of machine learning models.
```bash
python run_pipeline.py
```
This script will output the training process to the console and save the trained models (`Random_Forest.pkl`, `XGBoost.pkl`, etc.) into the `models/` directory. It will also output a final evaluation table sorted by Macro Recall.

## 🏆 Model Performance

The pipeline trains several models, focusing on robust tree-based algorithms to bypass local environment restrictions with linear solvers. On the real-world dataset (balanced via SMOTE), the models achieve the following performance (Macro Recall):

1. **Random Forest:** 0.8660 *(Champion Model)*
2. **XGBoost:** 0.8599
3. **Voting Ensemble:** 0.8437
4. **Decision Tree:** 0.8253
5. **Gradient Boosting:** 0.7981

> **Note:** Random Forest was selected as the champion model as it demonstrated the highest recall, which is the most critical metric in medical diagnostics (minimizing false negatives for high-risk patients).

---
*Developed for coursework assessment in Data Analytics and Visualisation (COM725).*
