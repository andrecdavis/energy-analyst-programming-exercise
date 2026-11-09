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


# ======================================================================
# Task 3 — Write AveragePriceByMonth.csv (SettlementPoint, Year, Month, AveragePrice)
# ======================================================================
# Write the per-settlement-point monthly averages (exactly the 4 required columns, no index)
monthly_avg[["SettlementPoint", "Year", "Month", "AveragePrice"]].to_csv(
    OUTPUT_DIR / "AveragePriceByMonth.csv", index=False
)


# ======================================================================
# Task 4 — Hourly volatility per year, hubs (HB_) only, set prices ≤ 0 to NaN before log returns
# ======================================================================
# Hubs only (prefix HB_); load zones are excluded by the task
hubs = prices[prices["SettlementPoint"].str.startswith("HB_")].copy()

# Sort each hub's prices in time order before taking returns
hubs = hubs.sort_values(["SettlementPoint", "Date"])

# Filter out prices <= 0 (log is only defined for positive values).
# Set them to NaN instead of dropping the rows, so no return is computed across
# the gap (a multi-hour change would otherwise be reported as a 1-hour return).
hubs["Price"] = hubs["Price"].where(hubs["Price"] > 0)

# Hourly log return = ln(P_t) - ln(P_t-1), computed within each hub and year
# (groupby keeps returns from crossing hubs or year boundaries).
# Returns into or out of a NaN hour are NaN and are skipped by std().
hubs["LogReturn"] = hubs.groupby(["SettlementPoint", "Year"])["Price"].transform(
    lambda p: np.log(p).diff()
)

# Hourly volatility = standard deviation of log returns (sample std, ddof=1)
hourly_vol = (
    hubs.groupby(["SettlementPoint", "Year"], as_index=False)["LogReturn"]
    .std()
    .rename(columns={"LogReturn": "HourlyVolatility"})
)


# ======================================================================
# Task 5 — Write HourlyVolatilityByYear.csv (SettlementPoint, Year, HourlyVolatility)
# ======================================================================
# Write hourly volatilities (exact, case-sensitive column names, no index)
hourly_vol[["SettlementPoint", "Year", "HourlyVolatility"]].to_csv(
    OUTPUT_DIR / "HourlyVolatilityByYear.csv", index=False
)


# ======================================================================
# Task 6 — Highest-volatility hub per year → MaxVolatilityByYear.csv
# ======================================================================
# For each year, keep the hub row with the highest hourly volatility
# (compared at full precision, so close calls like 2016 are not rounded into a tie)
max_vol = hourly_vol.loc[hourly_vol.groupby("Year")["HourlyVolatility"].idxmax()]

# Same columns as Task 5
max_vol[["SettlementPoint", "Year", "HourlyVolatility"]].to_csv(
    OUTPUT_DIR / "MaxVolatilityByYear.csv", index=False
)


# ======================================================================
# Task 7 — Model-ready files: output/formattedSpotHistory/spot_<SettlementPoint>.csv (15 files)
# ======================================================================
# Model-ready format (see supplementalMaterials): one row per day,
# columns Variable, Date, X1..X24. Data is hour-beginning, so hour 00:00 -> X1, 23:00 -> X24.
SPOT_DIR = OUTPUT_DIR / "formattedSpotHistory"
SPOT_DIR.mkdir(exist_ok=True)

spot = prices[["SettlementPoint", "Date", "Price"]].copy()
spot["Day"] = spot["Date"].dt.strftime("%Y-%m-%d")
spot["HourCol"] = "X" + (spot["Date"].dt.hour + 1).astype(str)
hour_cols = [f"X{h}" for h in range(1, 25)]

for point, data in spot.groupby("SettlementPoint"):
    # One row per day, one column per hour; reindex keeps all 24 columns in order
    wide = data.pivot(index="Day", columns="HourCol", values="Price").reindex(columns=hour_cols)
    wide.insert(0, "Variable", point)
    wide = wide.reset_index().rename(columns={"Day": "Date"})
    wide = wide[["Variable", "Date"] + hour_cols]
    # Missing hours (DST spring-forward 02:00) are written as "NA", like the examples
    wide.to_csv(SPOT_DIR / f"spot_{point}.csv", index=False, na_rep="NA")
