# Energy Analyst Programming Exercise — main script
# Runs end-to-end: python main.py


# ======================================================================
# Parameters
# ======================================================================
# Required packages: pandas, numpy, matplotlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent     # repo root
DATA_DIR = BASE_DIR / "historicalPriceData"   # historical power price data
OUTPUT_DIR = BASE_DIR / "output"              # all generated CSVs and images
OUTPUT_DIR.mkdir(exist_ok=True)


# ======================================================================
# Task 1 — Read and combine all historical data files
# ======================================================================
# Read every yearly price file and stack them into one dataframe
files = sorted(DATA_DIR.glob("*.csv"))
prices = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Parse timestamps so we can group by year/month and sort by hour later
prices["Date"] = pd.to_datetime(prices["Date"])

# Quick checks
print("Files read:", [f.name for f in files])
print("Rows:", len(prices))
print("Date range:", prices["Date"].min(), "to", prices["Date"].max())
print("Settlement points:", prices["SettlementPoint"].nunique())
print(sorted(prices["SettlementPoint"].unique()))
print("Missing prices:", prices["Price"].isna().sum())
print(prices.head())
