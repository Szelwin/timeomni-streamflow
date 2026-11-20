"""
Event descriptions and forecasting prompts for the TimeOmni-1 / Qwen2.5 experiments.

Every event is defined with an established threshold from the hydrological literature:

  Heavy precipitation day   P >= 5 x mean daily precipitation          Addor et al. (2017), CAMELS
  Dry day                   P < 1 mm/day                               Addor et al. (2017), CAMELS
  Precipitation phase       snow if daily mean air temperature < 0 °C  Addor et al. (2017), CAMELS
  Rain-on-snow              rain >= 10 mm on a pre-existing snowpack   Musselman et al. (2018)
                            of SWE >= 10 mm
  Relevant snowpack         SWE >= 10 mm                               Musselman et al. (2018);
                                                                       Freudiger et al. (2014)

Documented deviation: Musselman et al. (2018) additionally require snowmelt to contribute
>= 20 % of the water released. This criterion is not applied, because (a) daily SWE changes
mix melt with rain retained in the snowpack, and (b) the melt caused by rain on the origin
day appears in the following days, i.e. evaluating it would require future data.

Prompt format follows TimeOmni-1's event-aware forecasting task (Guan et al., 2026,
Appendix D.2 and Table 7): numeric history, dated event lines, then the forecast window.

LEAKAGE RULE: every function receives the full time series but uses only data up to and
including the forecast origin. check_no_leakage() verifies this by perturbing all values
after the origin and asserting that the prompt is unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "data" / "final"

LOOKBACK_DAYS = 7      # event window (decided)
INPUT_DAYS = 90        # streamflow history shown to the model
HORIZON_DAYS = 7       # forecast horizon
P_MEAN_END = "2024-09-30"  # mean daily precipitation from data before the test year only

HEAVY_FACTOR = 5.0     # Addor et al. (2017)
DRY_DAY_MM = 1.0       # Addor et al. (2017)
SNOW_TEMP_C = 0.0      # Addor et al. (2017)
ROS_RAIN_MM = 10.0     # Musselman et al. (2018)
SNOWPACK_SWE_MM = 10.0 # Musselman et al. (2018); Freudiger et al. (2014)

RIVER_NAMES = {
    "11264500": "Merced River at Happy Isles Bridge near Yosemite, California",
    "14305500": "Siletz River at Siletz, Oregon",
    "01047000": "Carrabassett River near North Anson, Maine",
    "04127997": "Sturgeon River at Wolverine, Michigan",
    "06906800": "Lamine River near Otterville, Missouri",
}

# TimeOmni-1 system prompt for Task 3 (CoT variant), Guan et al. (2026), Appendix D.2.
SYSTEM_PROMPT = (
    "You should think the impact of the event first, then output the predicted sequence. "
    "Output Format:\n"
    "<think>Your step-by-step reasoning process</think>\n"
    "<answer>[Your predicted sequence]</answer>"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_catchment(gauge: str) -> pd.DataFrame:
    return pd.read_csv(FINAL_DIR / f"{gauge}.csv", index_col="date", parse_dates=True)


def compute_p_mean(df: pd.DataFrame, end: str = P_MEAN_END) -> float:
    """Mean daily precipitation from the period before the test year only."""
    return float(df.loc[:end, "precip_mm_d"].mean())


def fmt_value(v: float) -> str:
    """Three significant digits, never scientific notation (keeps very low flows visible)."""
    return np.format_float_positional(v, precision=3, unique=False, fractional=False, trim="-")


def fmt_temp(t: float) -> str:
    return f"{t:+.1f} °C"


# --------------------------------------------------------------------------- #
# Event descriptions
# --------------------------------------------------------------------------- #
def describe_events(df: pd.DataFrame, origin: pd.Timestamp, p_mean: float,
                    lookback: int = LOOKBACK_DAYS) -> list[str]:
    """Dated event lines for the `lookback` days ending at (and including) `origin`."""
    origin = pd.Timestamp(origin)
    past = df.loc[:origin]                      # <-- nothing after the origin is ever used
    if past.index[-1] != origin:
        raise ValueError(f"origin {origin.date()} not in data")
    window = past.iloc[-lookback:]
    swe_prev = past["swe_mm"].shift(1)          # SWE at the end of the previous day
    lines = []

    # Day-by-day events: heavy precipitation and rain-on-snow (merged if on the same day)
    for day, row in window.iterrows():
        p, t = row["precip_mm_d"], row["temp_mean_c"]
        is_rain = t >= SNOW_TEMP_C
        heavy = p >= HEAVY_FACTOR * p_mean
        ros = is_rain and p >= ROS_RAIN_MM and swe_prev.loc[day] >= SNOWPACK_SWE_MM
        if not (heavy or ros):
            continue
        label = " and ".join(x for x in ["Heavy precipitation" if heavy else "",
                                         "rain-on-snow" if ros else ""] if x)
        label = label[0].upper() + label[1:]
        text = f"{day.date()} {label}: {p:.1f} mm"
        if heavy:
            text += f" ({p / p_mean:.0f} times this catchment's mean daily precipitation)"
        text += f", falling as {'rain' if is_rain else 'snow'} (mean air temperature {fmt_temp(t)})"
        if ros:
            text += f", on an existing snowpack of {swe_prev.loc[day]:.0f} mm water equivalent"
        lines.append(text + ".")

    # Dry period: every day in the window is a dry day
    if (window["precip_mm_d"] < DRY_DAY_MM).all():
        lines.append(f"{window.index[0].date()} to {origin.date()} Dry period: "
                     f"no day with at least {DRY_DAY_MM:.0f} mm of precipitation.")

    # Snowpack state at the origin, if a relevant snowpack existed at any point in the window
    swe_start = past["swe_mm"].iloc[-lookback - 1]   # end of the day before the window
    if max(window["swe_mm"].max(), swe_start) >= SNOWPACK_SWE_MM:
        swe_now = window["swe_mm"].iloc[-1]
        lines.append(f"{origin.date()} Snowpack: {swe_now:.0f} mm water equivalent "
                     f"({swe_now - swe_start:+.0f} mm over the past {lookback} days).")
    return lines


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def build_prompt(df: pd.DataFrame, gauge: str, origin: pd.Timestamp, with_context: bool,
                 p_mean: float | None = None) -> tuple[str, str]:
    """(system prompt, user prompt). The two conditions differ only in the event block."""
    origin = pd.Timestamp(origin)
    past = df.loc[:origin]
    history = past["streamflow_mm_d"].iloc[-INPUT_DAYS:]
    if len(history) < INPUT_DAYS or history.isna().any():
        raise ValueError(f"{gauge} {origin.date()}: incomplete streamflow history")

    values = ", ".join(fmt_value(v) for v in history)
    user = (f"Over the past {INPUT_DAYS} days (from {history.index[0].date()} to {origin.date()}), "
            f"daily mean streamflow of the {RIVER_NAMES[gauge]}, in mm/day, was:\n[{values}].")

    if with_context:
        if p_mean is None:
            raise ValueError("p_mean is required for the context condition")
        first = (origin - pd.Timedelta(days=LOOKBACK_DAYS - 1)).date()
        events = describe_events(df, origin, p_mean)
        user += (f" Significant events in the past {LOOKBACK_DAYS} days "
                 f"(from {first} to {origin.date()}):\n")
        user += "\n".join(events) if events else "None."

    start = (origin + pd.Timedelta(days=1)).date()
    end = (origin + pd.Timedelta(days=HORIZON_DAYS)).date()
    user += (f"\n\nForecast the daily mean streamflow, in mm/day, for each of the next "
             f"{HORIZON_DAYS} days (from {start} to {end}).")
    return SYSTEM_PROMPT, user


# --------------------------------------------------------------------------- #
# Leakage test
# --------------------------------------------------------------------------- #
def check_no_leakage(df: pd.DataFrame, gauge: str, origin: pd.Timestamp, p_mean: float,
                     seed: int = 0) -> None:
    """Randomise every value after the origin; the prompt must not change at all."""
    if pd.Timestamp(P_MEAN_END) > pd.Timestamp(origin):
        raise AssertionError("p_mean period extends beyond the origin")
    rng = np.random.default_rng(seed)
    perturbed = df.copy()
    after = perturbed.index > pd.Timestamp(origin)
    for col in ["streamflow_mm_d", "precip_mm_d", "temp_mean_c", "temp_min_c", "temp_max_c", "swe_mm"]:
        perturbed.loc[after, col] = rng.uniform(0, 200, after.sum())
    for ctx in (False, True):
        if build_prompt(df, gauge, origin, ctx, p_mean) != build_prompt(perturbed, gauge, origin, ctx, p_mean):
            raise AssertionError(f"LEAKAGE: {gauge} {pd.Timestamp(origin).date()} context={ctx}")


# --------------------------------------------------------------------------- #
# Demonstration on the real data
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    test_days = pd.date_range("2024-10-01", "2025-09-23", freq="D")  # forecast window stays in WY2025
    summary = []
    for gauge in RIVER_NAMES:
        df = load_catchment(gauge)
        p_mean = compute_p_mean(df)
        counts = {"heavy": 0, "rain-on-snow": 0, "dry": 0, "snowpack": 0, "none": 0}
        for origin in test_days:
            check_no_leakage(df, gauge, origin, p_mean)
            events = describe_events(df, origin, p_mean)
            text = " ".join(events)
            counts["heavy"] += "Heavy precipitation" in text
            counts["rain-on-snow"] += "ain-on-snow" in text
            counts["dry"] += "Dry period" in text
            counts["snowpack"] += "Snowpack" in text
            counts["none"] += not events
        summary.append({"gauge": gauge, "p_mean_mm_d": round(p_mean, 2),
                        "heavy_threshold_mm": round(HEAVY_FACTOR * p_mean, 1), **counts})

    print(f"Leakage test passed for all {len(test_days)} origins x {len(RIVER_NAMES)} catchments.\n")
    print("Number of origins (out of %d) whose event block mentions each event type:" % len(test_days))
    print(pd.DataFrame(summary).to_string(index=False))

    gauge = "01047000"
    origin = pd.Timestamp("2024-12-11")
    df = load_catchment(gauge)
    system, user = build_prompt(df, gauge, origin, with_context=True, p_mean=compute_p_mean(df))
    print("\n" + "=" * 80 + "\nSYSTEM PROMPT\n" + system + "\n\nUSER PROMPT (with context)\n" + user)
