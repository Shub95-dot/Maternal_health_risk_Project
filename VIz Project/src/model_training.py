import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_models(num_classes):
    """
    Returns a dictionary of properly scaled models and meta-learners.
    """
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

    # Dictionary of standalone models
    models = {
        "Decision Tree": dt,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "XGBoost": xgb
    }

    estimators = [
        ("dt", dt),
        ("rf", rf),
        ("gb", gb)
    ]

    models["Voting Ensemble"] = VotingClassifier(
        estimators=estimators,
        voting="soft"
    )

    return models
