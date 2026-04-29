import sys
import os
import pandas as pd
from sklearn.metrics import recall_score, classification_report, confusion_matrix
import joblib

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_preprocessing import load_and_engineer_features, prepare_data_for_modeling
from model_training import get_models

def main():
    log = open("run_log.txt", "w")
    def p(text):
        print(text)
        log.write(text + "\n")
        log.flush()

    p("=" * 50)
    p("Maternal Health Risk - Master Pipeline")
    p("=" * 50)
    
    data_path = os.path.join("data", "Maternal Health Risk Data Set.csv")
    p(f"Loading REAL data from: {data_path}")
    
    # Load and preprocess
    df = load_and_engineer_features(data_path)
    X_train, X_test, y_train, y_test, le = prepare_data_for_modeling(df, use_smote=True)
    
    p("Applied SMOTE to the training set to handle class imbalance.")
    
    # Initialize models
    models = get_models(num_classes=len(le.classes_))
    
    results = {}
    
    # Training and evaluation
    for name, model in models.items():
        p("-" * 30)
        p(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        preds = model.predict(X_test)
        
        # Metrics
        macro_recall = recall_score(y_test, preds, average="macro")
        results[name] = macro_recall
        
        p(f"Macro Recall: {macro_recall:.4f}")
        
        # Save model
        model_filename = os.path.join("models", f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_filename)
        p(f"Saved {name} to {model_filename}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        p(f"Confusion Matrix for {name}:\n{cm}\n")

    # Comparison Table
    p("\n" + "=" * 50)
    p("FINAL EVALUATION RESULTS")
    p("=" * 50)
    comparison_df = pd.DataFrame.from_dict(results, orient="index", columns=["Macro Recall"])
    comparison_df = comparison_df.sort_values(by="Macro Recall", ascending=False)
    p(comparison_df.to_string())
    log.close()

if __name__ == "__main__":
    main()
