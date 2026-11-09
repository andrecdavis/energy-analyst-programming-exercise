# Energy Analyst Programming Exercise — main script
# Runs end-to-end: python main.py

# ---- System-specific parameters (update these to run on another machine) ----
# Required packages: pandas, numpy, matplotlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "historicalPriceData"   # historical power price data
OUTPUT_DIR = BASE_DIR / "output"              # all generated CSVs and images
OUTPUT_DIR.mkdir(exist_ok=True)
# -----------------------------------------------------------------------------
