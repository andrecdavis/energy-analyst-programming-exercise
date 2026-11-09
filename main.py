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


# ======================================================================
# Task 2 — Monthly average price per settlement point (Jan 2016–Dec 2019, keep prices ≤ 0)
# ======================================================================
# Add Year and Month columns from the hourly timestamp
prices["Year"] = prices["Date"].dt.year
prices["Month"] = prices["Date"].dt.month

# Average price per settlement point per year-month.
# Prices <= 0 are kept on purpose: negative prices are real in deregulated markets.
# Result: 681 rows, not 15 x 48 = 720. HB_PAN only has data from 2019-04-06,
# so it has 9 year-months (Apr-Dec 2019); April 2019 is a partial month (600 hours).
monthly_avg = (
    prices.groupby(["SettlementPoint", "Year", "Month"], as_index=False)["Price"]
    .mean()
    .rename(columns={"Price": "AveragePrice"})
)

# Second average: one price per year-month across ALL settlement points (48 rows).
# Uses every hourly price from every point in that month, so it weights by hours of data.
# Note: HB_PAN joins from April 2019, so from then on 15 points feed the average instead of 14.
monthly_avg_all_points = (
    prices.groupby(["Year", "Month"], as_index=False)["Price"]
    .mean()
    .rename(columns={"Price": "AveragePrice"})
)
