import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_models(num_classes):
    """
    Returns a dictionary of properly scaled models and meta-learners.
    Each model is wrapped in a Pipeline with a StandardScaler.
    """
    
    # Define individual models
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="auto",
        n_jobs=1,
        random_state=42
    )

    # Wrap models in Pipelines with StandardScaler
    models = {
        "Decision Tree": Pipeline([("scaler", StandardScaler()), ("model", dt)]),
        "Random Forest": Pipeline([("scaler", StandardScaler()), ("model", rf)]),
        "Gradient Boosting": Pipeline([("scaler", StandardScaler()), ("model", gb)]),
        "XGBoost": Pipeline([("scaler", StandardScaler()), ("model", xgb)])
    }

    # Define estimators for the Voting Ensemble
    # We use the scaled versions to ensure consistency
    estimators = [
        ("dt", models["Decision Tree"]),
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"])
    ]

    models["Voting Ensemble"] = VotingClassifier(
        estimators=estimators,
        voting="soft"
    )

    return models
