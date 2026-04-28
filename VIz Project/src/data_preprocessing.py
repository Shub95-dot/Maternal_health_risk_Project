import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_engineer_features(filepath):
    """
    Loads the dataset and applies centralized feature engineering.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    df["MAP"] = (df["SystolicBP"] + 2 * df["DiastolicBP"]) / 3
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    df["ShockIndex"] = df["HeartRate"] / df["SystolicBP"]
    df["BPRatio"] = df["SystolicBP"] / df["DiastolicBP"]
    
    # Clinical Heuristics (Using thresholds to create a composite risk score)
    temp_dev = df["BodyTemp"] - 98.2
    df["CombinedRiskScore"] = (
        (df["MAP"] > 105).astype(int) +
        (df["BS"] > 10).astype(int) +
        (df["HeartRate"] > 90).astype(int) +
        (temp_dev > 1).astype(int)
    )
    
    return df

def prepare_data_for_modeling(df, target_col="RiskLevel", test_size=0.2, random_state=42, use_smote=False):
    """
    Splits data into X and y, encodes target, performs train/test split,
    and optionally applies SMOTE to the training set.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, le
