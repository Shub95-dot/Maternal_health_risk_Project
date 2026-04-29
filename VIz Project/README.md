# Maternal Health Risk Analysis & Prediction Project

This project provides a comprehensive end-to-end pipeline for analyzing maternal health risk data and training predictive models to classify risk levels (Low, Mid, High).

## Project Structure

- `data/`: Contains the raw and engineered datasets.
- `notebooks/`: 
    - `eda_main_data.ipynb`: Exploratory Data Analysis and Feature Engineering.
    - `modeling.ipynb`: Extensive model experimentation and evaluation.
- `src/`: Core Python modules for modularity.
    - `data_preprocessing.py`: Feature engineering and data preparation logic.
    - `model_training.py`: Model definitions and ensemble logic.
- `models/`: Saved `.pkl` files for the trained models.
- `figures/`: Automatically generated high-resolution plots for reports.
- `tests/`: Unit tests for ensuring pipeline reliability.
- `main.py`: Central entry point to run everything (tests, pipeline, and notebook execution).
- `run_pipeline.py`: Independent script to train all models and save results.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Entire Project**:
   Execute the master script to run tests, train models, and generate all visualizations:
   ```bash
   python main.py
   ```

3. **Check Results**:
   - Training logs are saved in `run_log.txt`.
   - All report visualizations are saved in the `figures/` directory.
   - Saved models are available in the `models/` directory.

## Key Features
- **SMOTE Balancing**: Handles class imbalance in maternal health risk levels.
- **Advanced Engineering**: Includes clinical heuristics like Mean Arterial Pressure (MAP) and Shock Index.
- **Ensemble Learning**: Features XGBoost, Random Forest, and a Voting Ensemble.
- **Automated Reporting**: Jupyter notebooks are executed programmatically to ensure figures are always up-to-date with the latest data.
