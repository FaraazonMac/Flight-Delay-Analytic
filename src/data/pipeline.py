"""Shared data-loading pipeline used by both the Streamlit app and the
training script, so features are built in exactly one place."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Fallback names for the 14 carriers in the 2015 DOT dataset, used when
# data/raw/airlines.csv has not been downloaded.
AIRLINE_NAMES = {
    "UA": "United Air Lines Inc.", "AA": "American Airlines Inc.",
    "US": "US Airways Inc.", "F9": "Frontier Airlines Inc.",
    "B6": "JetBlue Airways", "OO": "Skywest Airlines Inc.",
    "AS": "Alaska Airlines Inc.", "NK": "Spirit Air Lines",
    "WN": "Southwest Airlines Co.", "DL": "Delta Air Lines Inc.",
    "EV": "Atlantic Southeast Airlines", "HA": "Hawaiian Airlines Inc.",
    "MQ": "American Eagle Airlines Inc.", "VX": "Virgin America",
}
DAY_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
             5: "Friday", 6: "Saturday", 7: "Sunday"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def time_of_day(scheduled_dep: pd.Series) -> pd.Series:
    """Bucket HHMM scheduled departure into four labelled windows."""
    hour = (scheduled_dep // 100).astype("int16")
    labels = np.select(
        [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 17),
         (hour >= 17) & (hour < 21)],
        ["Morning (5am-12pm)", "Afternoon (12pm-5pm)", "Evening (5pm-9pm)"],
        default="Night (9pm-5am)",
    )
    return pd.Series(labels, index=scheduled_dep.index)


def load_and_preprocess_data(config_path: Path = CONFIG_PATH):
    """Return (dataframe, config). Raises FileNotFoundError if data is missing."""
    config = load_config(config_path)
    csv_path = ROOT / config["data"]["sample_path"]
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ARRIVAL_DELAY", "SCHEDULED_DEPARTURE"]).copy()

    # NOTE: ~8% of 2015 rows use numeric airport IDs instead of IATA codes.
    # They are kept so system-wide metrics match the full-dataset analysis.
    df["ORIGIN_AIRPORT"] = df["ORIGIN_AIRPORT"].astype(str)
    df["DESTINATION_AIRPORT"] = df["DESTINATION_AIRPORT"].astype(str)

    airline_map = AIRLINE_NAMES
    airlines_csv = ROOT / config["data"]["airlines_path"]
    if airlines_csv.exists():
        a = pd.read_csv(airlines_csv)
        airline_map = dict(zip(a["IATA_CODE"].astype(str).str.strip(), a["AIRLINE"]))

    df["AIRLINE_NAME"] = df["AIRLINE"].map(airline_map).fillna(df["AIRLINE"])
    df["DAY_NAME"] = df["DAY_OF_WEEK"].map(DAY_NAMES)
    df["TIME_OF_DAY"] = time_of_day(df["SCHEDULED_DEPARTURE"])
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] > 15).astype("int8")

    # Memory: downcast numerics, categorise strings.
    for col in ("YEAR", "MONTH", "DAY", "DAY_OF_WEEK"):
        df[col] = df[col].astype("int16" if col == "YEAR" else "int8")
    for col in ("ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE", "AIR_SYSTEM_DELAY",
                "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"):
        df[col] = df[col].astype("float32")
    for col in ("AIRLINE", "AIRLINE_NAME", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
                "DAY_NAME", "TIME_OF_DAY"):
        df[col] = df[col].astype("category")
    return df.reset_index(drop=True), config
