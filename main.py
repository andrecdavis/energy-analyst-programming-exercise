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


# ======================================================================
# Bonus — Mean plots: SettlementHubAveragePriceByMonth.png, LoadZoneAveragePriceByMonth.png
# ======================================================================
import matplotlib.dates as mdates

# Date column for the x-axis: first day of each month keeps months in chronological order
monthly_avg["MonthStart"] = pd.to_datetime(dict(year=monthly_avg["Year"], month=monthly_avg["Month"], day=1))

# Eight distinct colors, assigned in a fixed order (one per settlement point)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot_monthly_avg(prefix, title, filename):
    """Overlay one line per settlement point with the given prefix and save as PNG."""
    data = monthly_avg[monthly_avg["SettlementPoint"].str.startswith(prefix)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for color, (point, d) in zip(COLORS, data.groupby("SettlementPoint")):
        ax.plot(d["MonthStart"], d["AveragePrice"], color=color, linewidth=2, label=point)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average price ($/MWh)")
    # Tick every 3 months, labelled as YYYY-MM-DD (first day of the month)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, color="#e0e0e0", linewidth=0.8)
    ax.legend(title="Settlement point", loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


plot_monthly_avg("HB_", "Monthly average price by settlement hub", "SettlementHubAveragePriceByMonth.png")
plot_monthly_avg("LZ_", "Monthly average price by load zone", "LoadZoneAveragePriceByMonth.png")


# ======================================================================
# Bonus — Volatility plots
# ======================================================================
# Hub -> color, same assignment as the hub mean plot (sorted hub names, fixed color order)
hub_names = sorted(hourly_vol["SettlementPoint"].unique())
HUB_COLORS = dict(zip(hub_names, COLORS))
vol_wide = hourly_vol.pivot(index="Year", columns="SettlementPoint", values="HourlyVolatility")
years = vol_wide.index.tolist()

# Grouped bars - compare hubs within each year (tallest bar = Task 6 winner)
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.8 / len(hub_names)
for k, hub in enumerate(hub_names):
    x = [y + (k - (len(hub_names) - 1) / 2) * bar_width for y in years]
    ax.bar(x, vol_wide[hub], width=bar_width, color=HUB_COLORS[hub], label=hub)
ax.set_xticks(years)
ax.set_title("Hourly price volatility by settlement hub and year")
ax.set_xlabel("Year")
ax.set_ylabel("Hourly volatility (std. dev. of log returns)")
ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(title="Settlement hub", loc="upper left")
ax.text(0.5, 0.97, "HB_PAN has data from Apr 2019 only", transform=ax.transAxes, ha="center", va="top", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "HourlyVolatilityByHubBars.png", dpi=150)
plt.close(fig)


# ======================================================================
# Bonus — Hourly shape profiles: output/hourlyShapeProfiles/profile_<SettlementPoint>.csv (15 files)
# ======================================================================
# Hourly shape profile: the 24 average hourly prices for one month-of-year and
# day-of-week, divided by their own mean so the 24 values average exactly 1.
PROFILE_DIR = OUTPUT_DIR / "hourlyShapeProfiles"
PROFILE_DIR.mkdir(exist_ok=True)

shape = prices[["SettlementPoint", "Date", "Price"]].copy()
shape["Month"] = shape["Date"].dt.month
shape["DayOfWeek"] = shape["Date"].dt.dayofweek + 1   # 1 = Monday ... 7 = Sunday
shape["Hour"] = shape["Date"].dt.hour + 1             # hour-beginning: 00:00 -> hour 1

# Average price for each point, month, day of week and hour (all years pooled)
avg = shape.groupby(["SettlementPoint", "Month", "DayOfWeek", "Hour"])["Price"].mean()

# Wide layout: one row per (point, month, day of week), columns X1..X24
profiles = avg.unstack("Hour")
profiles.columns = [f"X{h}" for h in profiles.columns]

# Normalize: divide each row by its own 24-hour mean
# Caveat: HB_PAN (data from Apr 2019 only, so 63 profiles instead of 84) has frequent
# negative prices. Some of its 24-hour means are near zero or negative (Dec-Thursday = -0.75),
# so dividing flips signs and gives extreme values (-11 to +16). These profiles still average
# exactly 1 but are not reliable shapes; all other points range about 0.22-5.9.
profiles = profiles.div(profiles.mean(axis=1), axis=0)

for point, p in profiles.groupby(level="SettlementPoint"):
    p.reset_index().to_csv(PROFILE_DIR / f"profile_{point}.csv", index=False)


# ======================================================================
# Bonus — Open-ended analysis
# ======================================================================
# Open-ended analysis: one extreme morning reshapes a whole hourly shape profile.
#
# Each shape profile averages only ~18 days (one month-of-year x one day-of-week over
# 4 years), so a single scarcity event can dominate it. January Wednesday, all points:
#   - On Wed 2018-01-17, prices across the 14 settlement points rose from $72-79 at 03:00
#     to $1,543-1,617 at 06:00 and $1,637-1,710 at 07:00, then fell to $102-144 by 09:00
#     (median January hourly price, all points, 2016-2019: ~$22).
#   - It was system-wide: all points moved together within a narrow band, so this was a
#     market-wide supply shortage, not local congestion.
#   - It was a multi-day event: the all-point daily average was ~$26-33 on 01-12..01-15,
#     $64.0 on 01-16, $229.7 on 01-17, $75.4 on 01-18, then ~$18-29 on 01-19..01-21.
#   - Timing (early-morning heating peak in mid-January) is consistent with a cold
#     snap; weather/load data would be needed to confirm the cause.
# Effect: using all points' prices, the January-Wednesday shape peaks at ~3.5x the daily
# mean (every individual point: 3.35-3.55x), versus at most ~2.0x on other January days.
# Without 2018-01-17 (dashed line) the Wednesday peak is ~1.7x, like the other weekdays.
# Mean-based shapes from short histories mix "typical shape" with rare events; a model
# using them should consider more years, medians, or spike handling.
# A second chart shows hourly prices at every settlement point around the event.
SPIKE_MONTH = 1
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
month_rows = shape[shape["Month"] == SPIKE_MONTH]   # all settlement points


def normalized_profile(rows):
    """24 average hourly prices divided by their mean (same method as the shape profiles)."""
    hourly = rows.groupby("Hour")["Price"].mean()
    return hourly / hourly.mean()


# The single highest hourly price in January (any point) identifies the spike day
spike = month_rows.loc[month_rows["Price"].idxmax()]
spike_day, spike_dow = spike["Date"].normalize(), spike["DayOfWeek"]

# Chart 1: market-wide January shape profile for each day of week
fig, ax = plt.subplots(figsize=(12, 6))
for dow, color in zip(range(1, 8), COLORS):
    width = 3 if dow == spike_dow else 1.2
    ax.plot(range(1, 25), normalized_profile(month_rows[month_rows["DayOfWeek"] == dow]),
            color=color, linewidth=width, label=DAY_NAMES[dow - 1])

# Same day-of-week profile recomputed without the spike day
same_dow = month_rows[month_rows["DayOfWeek"] == spike_dow]
ax.plot(range(1, 25), normalized_profile(same_dow[same_dow["Date"].dt.normalize() != spike_day]),
        color=COLORS[spike_dow - 1], linewidth=2, linestyle="--",
        label=f"{DAY_NAMES[spike_dow - 1]} without {spike_day:%Y-%m-%d}")

ax.axhline(1, color="#999999", linewidth=0.8)
ax.set_title(f"January shape profiles, all settlement points: one morning ({spike_day:%Y-%m-%d}) "
             f"drives the {DAY_NAMES[spike_dow - 1]} spike")
ax.set_xlabel("Hour (hour-beginning, 1 = 00:00)")
ax.set_ylabel("Shape factor (mean = 1)")
ax.set_xticks([1, 6, 12, 18, 24])
ax.grid(True, color="#e0e0e0", linewidth=0.8)
ax.legend(title="Day of week", loc="upper right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ShapeProfileSpike_January.png", dpi=150)
plt.close(fig)

# Chart 2: hourly prices at every settlement point for the days around the spike.
# The lines lie almost on top of each other, showing the price jump was system-wide.
window = prices[(prices["Date"] >= spike_day - pd.Timedelta(days=2))
                & (prices["Date"] < spike_day + pd.Timedelta(days=3))]
fig, ax = plt.subplots(figsize=(12, 6))
for point, d in window.groupby("SettlementPoint"):
    ax.plot(d["Date"], d["Price"], color="#5b7fa8", linewidth=1, alpha=0.7)
ax.plot([], [], color="#5b7fa8", linewidth=1,
        label=f"Each settlement point ({window['SettlementPoint'].nunique()})")
peak = window[window["Date"] == spike["Date"]]["Price"]
ax.set_title(f"Hourly prices at all settlement points, {spike_day - pd.Timedelta(days=2):%Y-%m-%d} to "
             f"{spike_day + pd.Timedelta(days=2):%Y-%m-%d} (at {spike['Date']:%H:%M} on {spike_day:%Y-%m-%d}: "
             f"\\${peak.min():,.0f}-\\${peak.max():,.0f}/MWh)")  # \\$ = literal dollar sign
ax.set_xlabel("Date (hour-beginning)")
ax.set_ylabel("Price ($/MWh)")
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.grid(True, color="#e0e0e0", linewidth=0.8)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "SystemWidePrices_Jan2018.png", dpi=150)
plt.close(fig)
