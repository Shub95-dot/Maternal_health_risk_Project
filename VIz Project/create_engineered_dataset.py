import pandas as pd
import os
import sys

# Add src to path to import the engineering logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_preprocessing import load_and_engineer_features

def main():
    # Define paths
    input_path = os.path.join("data", "Maternal Health Risk Data Set.csv")
    output_path = os.path.join("data", "feature_engineered_main_dataset.csv")
    
    print(f"Loading data from {input_path}...")
    
    # Load and apply feature engineering
    df_engineered = load_and_engineer_features(input_path)
    
    # Save the new dataset
    print(f"Saving feature engineered dataset to {output_path}...")
    df_engineered.to_csv(output_path, index=False)
    
    print("Dataset creation complete!")
    print(f"New columns added: {list(df_engineered.columns)}")

if __name__ == "__main__":
    main()
