"""
Step 1: Download daily mean discharge for the five study catchments from the
USGS Water Data API and produce a data-quality report.

Requirements:  pip install dataretrieval pandas
Recommended:   a free USGS API key (https://api.waterdata.usgs.gov/signup/),
               set as environment variable API_USGS_PAT before running.

Outputs (in ./data/):
  raw/USGS-<id>_daily_raw.csv     untouched API response (keep for reproducibility)
  discharge_mm_per_day.csv        one column per gauge, daily values in mm/day
  quality_report.csv              completeness and approval status per gauge/period
"""

from pathlib import Path

import pandas as pd
from dataretrieval import waterdata

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Drainage areas (km^2) from the CAMELS-US attribute file camels_topo.txt
# (column area_gages2; Addor et al., 2017). Needed to convert ft^3/s to mm/day.
GAUGES = {
    "11264500": {"name": "Merced R. at Happy Isles Bridge nr Yosemite, CA", "area_km2": 467.98},
    "14305500": {"name": "Siletz River at Siletz, OR", "area_km2": 526.33},
    "01047000": {"name": "Carrabassett River nr North Anson, ME", "area_km2": 909.10},
    "04127997": {"name": "Sturgeon River at Wolverine, MI", "area_km2": 487.16},
    "06906800": {"name": "Lamine River nr Otterville, MO", "area_km2": 1414.75},
}

HISTORY = ("1980-10-01", "2024-09-30")  # water years 1981-2024: baselines, thresholds
TEST = ("2024-10-01", "2025-09-30")     # water year 2025: the test period

PARAMETER_CODE = "00060"  # discharge
STATISTIC_ID = "00003"    # daily mean

CFS_TO_M3S = 0.0283168466  # 1 ft^3/s in m^3/s

OUT = Path("data")
(OUT / "raw").mkdir(parents=True, exist_ok=True)


def cfs_to_mm_per_day(q_cfs: pd.Series, area_km2: float) -> pd.Series:
    """Convert discharge (ft^3/s) to runoff depth (mm/day) over the catchment."""
    m3_per_day = q_cfs * CFS_TO_M3S * 86400.0
    return m3_per_day / (area_km2 * 1e6) * 1000.0


def quality_row(df: pd.DataFrame, gauge: str, start: str, end: str, label: str) -> dict:
    """Completeness and approval statistics for one gauge and one period."""
    record_start = df["date"].min()
    # Days before the station's first record are not "missing", the gauge did not exist yet.
    expected = pd.date_range(max(pd.Timestamp(start), record_start), end, freq="D")
    sub = df[(df["date"] >= expected[0]) & (df["date"] <= expected[-1])]
    present = sub["value"].notna().sum()
    provisional = (sub["approval_status"] == "Provisional").sum()
    qualified = sub["qualifier"].notna() & (sub["qualifier"].astype(str).str.strip() != "")
    return {
        "gauge": gauge,
        "period": label,
        "record_start": record_start.date(),
        "days_expected": len(expected),
        "days_with_value": int(present),
        "pct_missing": round(100 * (1 - present / len(expected)), 2),
        "pct_provisional": round(100 * provisional / max(present, 1), 2),
        "pct_with_qualifier": round(100 * qualified.sum() / max(present, 1), 2),
        "qualifier_values": "; ".join(sorted(sub.loc[qualified, "qualifier"].astype(str).unique())),
    }


series_mm = {}
report = []

for gauge, meta in GAUGES.items():
    site = f"USGS-{gauge}"
    print(f"Downloading {site} ({meta['name']}) ...")
    df, _ = waterdata.get_daily(
        monitoring_location_id=site,
        parameter_code=PARAMETER_CODE,
        statistic_id=STATISTIC_ID,
        time=f"{HISTORY[0]}/{TEST[1]}",
    )
    if df.empty:
        print(f"  WARNING: no data returned for {site}")
        continue

    df.to_csv(OUT / "raw" / f"{site}_daily_raw.csv", index=False)

    units = df["unit_of_measure"].dropna().unique()
    print(f"  {len(df)} records, units: {units}")
    if len(units) != 1 or "ft" not in str(units[0]).lower():
        raise ValueError(f"Unexpected units for {site}: {units}. Check before converting.")

    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date", keep="last")

    report.append(quality_row(df, gauge, *HISTORY, "history WY1981-2024"))
    report.append(quality_row(df, gauge, *TEST, "test WY2025"))

    full_index = pd.date_range(HISTORY[0], TEST[1], freq="D")
    q = df.set_index("date")["value"].reindex(full_index)  # missing days become NaN
    series_mm[gauge] = cfs_to_mm_per_day(q, meta["area_km2"])

pd.DataFrame(series_mm).rename_axis("date").to_csv(OUT / "discharge_mm_per_day.csv")
report_df = pd.DataFrame(report)
report_df.to_csv(OUT / "quality_report.csv", index=False)

pd.set_option("display.width", 200)
print("\nData-quality report:")
print(report_df.drop(columns="qualifier_values").to_string(index=False))
print("\nQualifier codes found (look these up in the USGS documentation):")
for _, row in report_df.iterrows():
    if row["qualifier_values"]:
        print(f"  {row['gauge']} {row['period']}: {row['qualifier_values']}")
