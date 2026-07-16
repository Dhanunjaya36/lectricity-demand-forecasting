# scripts/evaluate_models.py

import pandas as pd
from src.electricity_demand.evaluation import ModelComparison

def main():
    try:
        df = pd.read_csv("model_comparison.csv")
        print("Model comparison results:")
        print(df)
    except FileNotFoundError:
        print("Run the pipeline first to generate model_comparison.csv")

if __name__ == "__main__":
    main()
