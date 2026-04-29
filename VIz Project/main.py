import subprocess
import os
import sys

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Use shell=True for Windows compatibility with module calls
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR during {description}:")
        print(e.stderr)
        return False
    return True

def main():
    print("=" * 60)
    print("MATERNAL HEALTH RISK PROJECT - MASTER EXECUTION")
    print("=" * 60)

    # 1. Run Unit Tests
    if not run_command("python -m pytest tests/test_preprocessing.py", "Running Unit Tests"):
        print("Tests failed. Aborting pipeline.")
        return

    # 2. Run Training Pipeline
    if not run_command("python run_pipeline.py", "Executing Model Training Pipeline"):
        print("Pipeline failed.")
        return

    # 3. Generate Visualizations from Notebooks
    print("\n>>> Generating Visualizations from Notebooks...")
    
    # Run EDA Notebook
    eda_cmd = "python -m nbconvert --to notebook --execute --inplace notebooks/eda_main_data.ipynb"
    if run_command(eda_cmd, "Executing EDA Notebook"):
        print("SUCCESS: EDA Visualizations saved to figures/")
    
    # Run Modeling Notebook
    mod_cmd = "python -m nbconvert --to notebook --execute --inplace notebooks/modeling.ipynb"
    if run_command(mod_cmd, "Executing Modeling Notebook"):
        print("SUCCESS: Modeling Evaluation Visualizations saved to figures/")

    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETE!")
    print("Check the 'figures/' folder for all exported plots.")
    print("Check 'run_log.txt' for training history.")
    print("=" * 60)

if __name__ == "__main__":
    main()
