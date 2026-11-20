"""
Step 4: Build the final, frozen per-catchment dataset used in all experiments.

Combines
  - unrounded daily discharge (mm/day) from Step 1   data/discharge_mm_per_day.csv
  - USGS quality information from the raw downloads  data/raw/USGS-<id>_daily_raw.csv
  - selected ERA5-Land variables from Caravan part 2 data/era5land_daily/timeseries/csv/camels/

Discharge is taken from Step 1 rather than from the Caravan files, because Caravan
rounds to two decimals, which turns very low flows (e.g. Lamine River) into 0.00.

Output: data/final/<gauge_id>.csv, one file per catchment, daily, local standard time.

Requirements: pip install pandas
"""

from ast import literal_eval
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DISCHARGE_FILE = DATA / "discharge_mm_per_day.csv"
RAW_DIR = DATA / "raw"
ERA5L_DIR = DATA / "era5land_daily" / "timeseries" / "csv" / "camels"
OUT_DIR = DATA / "final"

GAUGES = ["11264500", "14305500", "01047000", "04127997", "06906800"]

PERIOD = ("1981-01-01", "2025-09-30")  # ERA5-Land starts 1981; WY2025 ends 30 Sep 2025
TEST = ("2024-10-01", "2025-09-30")

# Caravan column -> final column name (units: mm/day, degrees Celsius, mm)
WEATHER_COLUMNS = {
    "total_precipitation_sum": "precip_mm_d",
    "temperature_2m_mean": "temp_mean_c",
    "temperature_2m_min": "temp_min_c",
    "temperature_2m_max": "temp_max_c",
    "snow_depth_water_equivalent_mean": "swe_mm",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
dates = pd.date_range(*PERIOD, freq="D")

discharge = pd.read_csv(DISCHARGE_FILE, index_col="date", parse_dates=True)
summary = []

for gauge in GAUGES:
    # --- Discharge (unrounded) -------------------------------------------- #
    q = discharge[gauge].reindex(dates)

    # --- USGS quality information ----------------------------------------- #
    raw = pd.read_csv(RAW_DIR / f"USGS-{gauge}_daily_raw.csv", dtype={"qualifier": str})
    raw["date"] = pd.to_datetime(raw["time"]).dt.tz_localize(None).dt.normalize()
    if raw["date"].duplicated().any():
        raise ValueError(f"{gauge}: duplicate dates in raw USGS file")
    raw = raw.set_index("date").reindex(dates)
    qualifier = raw["qualifier"].fillna("")
    approval = raw["approval_status"]

    # --- Weather (ERA5-Land via Caravan) ---------------------------------- #
    era5 = pd.read_csv(ERA5L_DIR / f"camels_{gauge}.csv", index_col="date", parse_dates=True)
    missing_cols = [c for c in WEATHER_COLUMNS if c not in era5.columns]
    if missing_cols:
        raise ValueError(f"{gauge}: missing columns in Caravan file: {missing_cols}")
    weather = era5[list(WEATHER_COLUMNS)].rename(columns=WEATHER_COLUMNS).reindex(dates)
    if weather.isna().any().any():
        raise ValueError(f"{gauge}: missing weather values inside {PERIOD}")

    # --- Combine ----------------------------------------------------------- #
    final = pd.DataFrame(index=dates)
    final.index.name = "date"
    final["streamflow_mm_d"] = q
    # Flag values USGS marks as estimated or ice-affected (less reliable than measured values).
    # All raw codes stay available in q_qualifier.
    final["q_estimated"] = qualifier.str.contains("ESTIMATED|ICE")
    final["q_qualifier"] = qualifier
    final["q_approval"] = approval
    final = final.join(weather)

    # --- Consistency checks ------------------------------------------------ #
    test = final.loc[TEST[0]:TEST[1]]
    if test["streamflow_mm_d"].isna().any():
        raise ValueError(f"{gauge}: missing discharge in the test period")
    if (test["q_approval"] != "Approved").any():
        print(f"  NOTE {gauge}: test period contains non-approved values")
    has_q = final["streamflow_mm_d"].notna()
    if (final.loc[~has_q, "q_qualifier"] != "").any():
        raise ValueError(f"{gauge}: qualifier present on a day without discharge")

    final.to_csv(OUT_DIR / f"{gauge}.csv")

    codes = sorted({c for s in qualifier[qualifier != ""] for c in literal_eval(s)}) if (qualifier != "").any() else []
    summary.append({
        "gauge": gauge,
        "days": len(final),
        "first_discharge": final.index[has_q][0].date(),
        "pct_estimated_all": round(100 * final.loc[has_q, "q_estimated"].mean(), 1),
        "pct_estimated_test": round(100 * test["q_estimated"].mean(), 1),
        "qualifier_codes": ", ".join(codes),
    })

pd.set_option("display.width", 200)
print(f"Final dataset written to {OUT_DIR}")
print(pd.DataFrame(summary).to_string(index=False))
