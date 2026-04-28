import pytest
import pandas as pd
import os
import sys
import numpy as np

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import load_and_engineer_features, prepare_data_for_modeling

@pytest.fixture
def sample_data_path(tmp_path):
    """Creates a dummy CSV file for testing."""
    data = {
        "Age": [25, 35, 45],
        "SystolicBP": [120, 140, 160],
        "DiastolicBP": [80, 90, 100],
        "BS": [7.0, 8.0, 15.0],
        "BodyTemp": [98.0, 99.0, 101.0],
        "HeartRate": [70, 80, 95],
        "RiskLevel": ["low risk", "mid risk", "high risk"]
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_maternal_data.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

def test_load_and_engineer_features(sample_data_path):
    """Checks if engineered columns are created correctly."""
    df = load_and_engineer_features(sample_data_path)
    
    expected_columns = [
        "Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel",
        "MAP", "PulsePressure", "ShockIndex", "BPRatio", "CombinedRiskScore"
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Column {col} missing from engineered dataframe"
        
    # Check a specific calculation (e.g., MAP)
    # MAP = (120 + 2*80) / 3 = 280 / 3 = 93.33
    assert df.loc[0, "MAP"] == pytest.approx(93.33, rel=1e-2)

def test_prepare_data_for_modeling(sample_data_path):
    """Checks shapes, label encoding, and SMOTE integration."""
    df = load_and_engineer_features(sample_data_path)
    
    # We need more data for SMOTE (at least 6 samples usually for default k_neighbors)
    # So we'll use a larger dummy set for this specific test
    data_large = {
        "Age": [25]*10 + [35]*10 + [45]*10,
        "SystolicBP": [120]*10 + [140]*10 + [160]*10,
        "DiastolicBP": [80]*10 + [90]*10 + [100]*10,
        "BS": [7.0]*10 + [8.0]*10 + [15.0]*10,
        "BodyTemp": [98.0]*10 + [99.0]*10 + [101.0]*10,
        "HeartRate": [70]*10 + [80]*10 + [95]*10,
        "RiskLevel": ["low risk"]*10 + ["mid risk"]*10 + ["high risk"]*10
    }
    df_large = pd.DataFrame(data_large)
    df_large = load_and_engineer_features_from_df(df_large) # Helper or just copy logic
    
    # Let's just mock the prepare_data_for_modeling input
    X_train, X_test, y_train, y_test, le = prepare_data_for_modeling(df_large, use_smote=True)
    
    # Check shapes
    assert len(X_train) > len(X_test)
    assert X_train.shape[1] == 11 # 6 raw + 5 engineered
    
    # Check label encoding
    assert set(y_train) == {0, 1, 2}
    assert "low risk" in le.classes_
    
    # Check SMOTE (classes should be perfectly balanced in training set)
    unique, counts = np.unique(y_train, return_counts=True)
    assert len(set(counts)) == 1, "SMOTE failed to balance classes in training set"

def load_and_engineer_features_from_df(df):
    """Helper for testing logic directly on a dataframe."""
    df["MAP"] = (df["SystolicBP"] + 2 * df["DiastolicBP"]) / 3
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    df["ShockIndex"] = df["HeartRate"] / df["SystolicBP"]
    df["BPRatio"] = df["SystolicBP"] / df["DiastolicBP"]
    temp_dev = df["BodyTemp"] - 98.2
    df["CombinedRiskScore"] = (
        (df["MAP"] > 105).astype(int) +
        (df["BS"] > 10).astype(int) +
        (df["HeartRate"] > 90).astype(int) +
        (temp_dev > 1).astype(int)
    )
    return df
