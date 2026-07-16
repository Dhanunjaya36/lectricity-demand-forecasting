# scripts/run_pipeline.py

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.electricity_demand.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
