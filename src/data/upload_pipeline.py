"""Upload pipeline: read -> map columns -> clean -> engineer features -> analyse.

Pure functions (no Streamlit) so every step can be tested on its own.
"""
from __future__ import annotations

import calendar
import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# canonical name -> (label shown in the UI, required?, accepted column names)
FIELDS = {
    "arrival_delay": ("Arrival delay (minutes)", True,
        ["arrival_delay", "arr_delay", "arrdelay", "arrival_delay_minutes",
         "arr_delay_minutes", "arrdelayminutes", "delay_minutes", "delay"]),
    "airline": ("Airline / carrier", True,
        ["airline", "carrier", "op_carrier", "op_unique_carrier", "unique_carrier",
         "uniquecarrier", "airline_code", "carrier_code", "airline_name",
         "operating_airline", "reporting_airline"]),
    "origin": ("Origin airport", True,
        ["origin_airport", "origin", "origin_airport_code", "dep_airport",
         "departure_airport", "from", "source", "origincode"]),
    "destination": ("Destination airport", False,
        ["destination_airport", "dest", "destination", "dest_airport",
         "arrival_airport", "arr_airport", "to", "destcode"]),
    "date": ("Flight date", False,
        ["fl_date", "flight_date", "date", "flightdate", "scheduled_date", "dep_date"]),
    "month": ("Month (1-12)", False, ["month"]),
    "day_of_week": ("Day of week", False,
        ["day_of_week", "dayofweek", "weekday", "dow", "day_name"]),
    "sched_dep": ("Scheduled departure time", False,
        ["scheduled_departure", "crs_dep_time", "crsdeptime", "sched_dep_time",
         "scheduled_dep_time", "std", "scheduled_departure_time"]),
    "departure_delay": ("Departure delay (minutes)", False,
        ["departure_delay", "dep_delay", "depdelay", "departure_delay_minutes",
         "dep_delay_minutes"]),
    "distance": ("Distance", False, ["distance", "distance_miles", "distance_km", "dist"]),
    "cancelled": ("Cancelled flag", False, ["cancelled", "canceled", "is_cancelled", "cancel"]),
    "late_aircraft": ("Late-aircraft delay (min)", False,
        ["late_aircraft_delay", "lateaircraftdelay", "late_aircraft"]),
    "air_system": ("Air-system / NAS delay (min)", False,
        ["air_system_delay", "nas_delay", "nasdelay", "air_traffic_delay"]),
    "carrier_delay": ("Carrier delay (min)", False,
        ["airline_delay", "carrier_delay", "carrierdelay"]),
    "weather": ("Weather delay (min)", False, ["weather_delay", "weatherdelay"]),
    "security": ("Security delay (min)", False, ["security_delay", "securitydelay"]),
}

CAUSE_COLUMNS = {
    "late_aircraft": ("LATE_AIRCRAFT_DELAY", "Late-arriving aircraft"),
    "air_system": ("AIR_SYSTEM_DELAY", "Air system congestion"),
    "carrier_delay": ("CARRIER_DELAY", "Carrier / airline"),
    "weather": ("WEATHER_DELAY", "Weather"),
    "security": ("SECURITY_DELAY", "Security"),
}

TIME_ORDER = ["Morning (5am-12pm)", "Afternoon (12pm-5pm)",
              "Evening (5pm-9pm)", "Night (9pm-5am)"]
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CATEGORY_ORDER = ["Early / on time", "Minor", "Moderate", "Severe"]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------------
# 1. Reading files
# --------------------------------------------------------------------------
def excel_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


def read_table(data: bytes, filename: str, sheet: str | None = None) -> pd.DataFrame:
    """Read a CSV or XLSX upload into a DataFrame."""
    if filename.lower().endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0)
    for encoding in ("utf-8", "latin-1"):
        try:
            text = data[:20000].decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding, low_memory=False)


# --------------------------------------------------------------------------
# 2. Column detection
# --------------------------------------------------------------------------
def detect_columns(columns: list[str]) -> dict[str, str | None]:
    """Match uploaded column names to canonical fields (exact, normalised match)."""
    lookup = {_norm(c): c for c in columns}
    used: set[str] = set()
    result: dict[str, str | None] = {}
    for key, (_, _, synonyms) in FIELDS.items():
        result[key] = None
        for syn in synonyms:
            col = lookup.get(_norm(syn))
            if col is not None and col not in used:
                result[key] = col
                used.add(col)
                break
    return result


# --------------------------------------------------------------------------
# 3. Cleaning
# --------------------------------------------------------------------------
@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)

    def add(self, step: str, affected: int, detail: str = "") -> None:
        self.steps.append({"Step": step, "Rows affected": int(affected), "What happened": detail})


def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.mask(s.isin(["", "nan", "NaN", "None", "NULL", "null", "N/A", "NA"]))
    if s.dropna().str.len().median() <= 4:  # looks like IATA / carrier codes
        s = s.str.upper()
    return s


def _to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    txt = s.astype("string").str.extract(r"(-?\d+\.?\d*)")[0]
    return pd.to_numeric(txt, errors="coerce")


def _parse_hour(s: pd.Series) -> pd.Series:
    """Hour 0-23 from HHMM numbers, 'HH:MM' text, Excel times or datetimes."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.hour.astype("float64")
    if pd.api.types.is_numeric_dtype(s):
        h = pd.to_numeric(s, errors="coerce") // 100
    else:
        txt = s.astype("string").str.strip()
        digits = txt.str.fullmatch(r"\d{1,4}").fillna(False).to_numpy(dtype=bool)
        h = pd.Series(np.nan, index=s.index, dtype="float64")
        if digits.any():
            h[digits] = pd.to_numeric(txt[digits]) // 100
        rest = (~digits) & txt.notna().to_numpy(dtype=bool)
        if rest.any():
            parsed = pd.to_datetime(txt[rest], errors="coerce", format="mixed")
            h[rest] = parsed.dt.hour
    h = h.where((h >= 0) & (h <= 24))
    return h % 24


def _time_bucket(hour: pd.Series) -> pd.Series:
    out = np.select(
        [(hour >= 5) & (hour < 12), (hour >= 12) & (hour < 17), (hour >= 17) & (hour < 21)],
        TIME_ORDER[:3], default=TIME_ORDER[3])
    out = pd.Series(out, index=hour.index, dtype="object")
    return out.where(hour.notna())


def _parse_dow(s: pd.Series) -> pd.Series:
    """Day names (Mon..Sun) from 1-7, 0-6 or day-name text."""
    if pd.api.types.is_numeric_dtype(s):
        n = pd.to_numeric(s, errors="coerce")
        if n.min() == 0:
            n = n + 1
    else:
        names = {d.lower(): i + 1 for i, d in enumerate(DAY_ORDER)}
        n = s.astype("string").str.strip().str[:3].str.lower().map(names)
        n = pd.to_numeric(n, errors="coerce")
    n = n.where((n >= 1) & (n <= 7))
    return n.map(lambda v: DAY_ORDER[int(v) - 1] if pd.notna(v) else np.nan)


def clean_data(raw: pd.DataFrame, mapping: dict, threshold: float = 15,
               drop_cancelled: bool = True, drop_duplicates: bool = True,
               max_abs_delay: float | None = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Return (clean_df, report). Raises ValueError if a required field is unmapped."""
    for key, (label, required, _) in FIELDS.items():
        if required and not mapping.get(key):
            raise ValueError(f"Required column not mapped: {label}")

    rep = CleaningReport(rows_in=len(raw), mapping={k: v for k, v in mapping.items() if v})
    df_raw = raw

    if drop_duplicates:
        before = len(df_raw)
        df_raw = df_raw.drop_duplicates()
        rep.add("Remove exact duplicate rows", before - len(df_raw),
                "Rows identical in every original column were kept once.")

    df = pd.DataFrame({k: df_raw[v].reset_index(drop=True)
                       for k, v in mapping.items() if v})

    # text columns
    for k in ("airline", "origin", "destination"):
        if k in df:
            df[k] = _clean_text(df[k])
    rep.add("Standardise text", 0, "Trimmed spaces, unified blanks/'N/A' as missing, upper-cased codes.")

    # numeric columns
    num_keys = ["arrival_delay", "departure_delay", "distance", *CAUSE_COLUMNS]
    bad_arrival = 0
    for k in num_keys:
        if k in df:
            before_valid = df[k].notna().sum()
            df[k] = _to_num(df[k])
            if k == "arrival_delay":
                bad_arrival = int(before_valid - df[k].notna().sum())
    rep.add("Convert numeric columns", bad_arrival,
            "Extracted numbers from text such as '12 min'; values that were not numbers became missing.")

    # cancelled flights
    if "cancelled" in df and drop_cancelled:
        flag = df["cancelled"].astype("string").str.strip().str.lower()
        is_cx = flag.isin(["1", "1.0", "y", "yes", "true", "t"]).to_numpy(dtype=bool)
        rep.add("Remove cancelled flights", int(is_cx.sum()),
                "Cancelled flights have no arrival delay, so they are excluded.")
        df = df[~is_cx]

    # missing key values
    before = len(df)
    df = df.dropna(subset=["arrival_delay"])
    rep.add("Drop rows with missing arrival delay", before - len(df),
            "Dropped, not filled with 0: a fake 0 would pull averages towards 'on time'.")
    before = len(df)
    df = df.dropna(subset=["airline", "origin"])
    rep.add("Drop rows with missing airline or origin", before - len(df), "")

    # extreme delays
    extreme = int((df["arrival_delay"].abs() > 1440).sum())
    if max_abs_delay:
        before = len(df)
        df = df[df["arrival_delay"].abs() <= max_abs_delay]
        rep.add(f"Exclude delays beyond +/-{int(max_abs_delay)} min", before - len(df),
                "Optional cap set by you.")
    elif extreme:
        rep.notes.append(f"{extreme:,} flights have a delay over 24 hours. They were kept; "
                         "use the cap option if you consider them errors.")

    df = df.reset_index(drop=True)

    # feature engineering
    out = pd.DataFrame({"AIRLINE": df["airline"], "ORIGIN": df["origin"]})
    if "destination" in df:
        out["DESTINATION"] = df["destination"]
    out["ARRIVAL_DELAY"] = df["arrival_delay"].astype("float32")
    if "departure_delay" in df:
        out["DEPARTURE_DELAY"] = df["departure_delay"].astype("float32")
    if "distance" in df:
        out["DISTANCE"] = df["distance"].astype("float32")

    month = dow = None
    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        if dates.notna().mean() > 0.5:
            month, dow = dates.dt.month, dates.dt.dayofweek.map(
                lambda v: DAY_ORDER[int(v)] if pd.notna(v) else np.nan)
        else:
            rep.notes.append("The date column could not be parsed, so it was ignored.")
    if month is None and "month" in df:
        month = pd.to_numeric(df["month"], errors="coerce")
    if dow is None and "day_of_week" in df:
        dow = _parse_dow(df["day_of_week"])
    if month is not None:
        out["MONTH"] = month.astype("float32")
    else:
        rep.notes.append("No date/month column: monthly trends are skipped.")
    if dow is not None:
        out["DAY_NAME"] = dow
    else:
        rep.notes.append("No date/weekday column: weekday analysis is skipped.")
    if "sched_dep" in df:
        hour = _parse_hour(df["sched_dep"])
        out["HOUR"] = hour.astype("float32")
        out["TIME_OF_DAY"] = _time_bucket(hour)
        if hour.isna().mean() > 0.5:
            rep.notes.append("Most scheduled-departure times could not be read; "
                             "time-of-day analysis may be unreliable.")
    else:
        rep.notes.append("No scheduled-departure time column: time-of-day analysis is skipped.")

    for key, (col, _) in CAUSE_COLUMNS.items():
        if key in df:
            out[col] = df[key].astype("float32")

    out["IS_DELAYED"] = (out["ARRIVAL_DELAY"] > threshold).astype("int8")
    severe_cut = max(60, threshold + 15)
    out["DELAY_CATEGORY"] = pd.cut(
        out["ARRIVAL_DELAY"], [-np.inf, 0, threshold, severe_cut, np.inf],
        labels=CATEGORY_ORDER).astype("object")

    for c in ("AIRLINE", "ORIGIN", "DESTINATION"):
        if c in out:
            out[c] = out[c].astype("category")

    for col, label in (("ORIGIN", "origin"), ("DESTINATION", "destination")):
        if col in out:
            numeric_share = out[col].astype("string").str.fullmatch(r"\d+").mean()
            if numeric_share > 0.01:
                rep.notes.append(
                    f"{numeric_share * 100:.0f}% of {label} values are numeric IDs (not airport "
                    "codes). Add a lookup table to convert them, or they will appear as numbers.")

    rep.rows_out = len(out)
    if rep.rows_in:
        kept = rep.rows_out / rep.rows_in * 100
        if kept < 80:
            rep.notes.append(f"Only {kept:.0f}% of rows survived cleaning. Check the column mapping.")
    if len(out) and (out["ARRIVAL_DELAY"] < 0).mean() > 0.9:
        rep.notes.append("Over 90% of delays are negative. Check the units and sign convention.")
    return out, rep


# --------------------------------------------------------------------------
# 4. Analysis
# --------------------------------------------------------------------------
def adaptive_min_flights(n_rows: int) -> int:
    """Minimum flights for a group to appear in rankings (scales with data size)."""
    return int(max(10, min(200, n_rows // 500)))


def kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    delayed = int(df["IS_DELAYED"].sum())
    return {
        "flights": n,
        "delayed": delayed,
        "delay_rate": delayed / n * 100 if n else 0.0,
        "mean_delay": float(df["ARRIVAL_DELAY"].mean()) if n else 0.0,
        "median_delay": float(df["ARRIVAL_DELAY"].median()) if n else 0.0,
        "p90_delay": float(df["ARRIVAL_DELAY"].quantile(0.9)) if n else 0.0,
        "avg_when_delayed": float(df.loc[df["IS_DELAYED"] == 1, "ARRIVAL_DELAY"].mean())
        if delayed else 0.0,
    }


def group_stats(df: pd.DataFrame, by, min_flights: int = 1) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(
        Flights=("ARRIVAL_DELAY", "size"),
        Delayed=("IS_DELAYED", "sum"),
        **{"Avg delay (min)": ("ARRIVAL_DELAY", "mean")},
    )
    g["Delay rate (%)"] = g["Delayed"] / g["Flights"] * 100
    return g[g["Flights"] >= min_flights].reset_index()


def cause_breakdown(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c: label for c, label in CAUSE_COLUMNS.values() if c in df.columns}
    delayed = df[df["IS_DELAYED"] == 1]
    if not cols or delayed.empty:
        return None
    totals = delayed[list(cols)].sum()
    if totals.sum() <= 0:
        return None
    res = pd.DataFrame({"Cause": [cols[c] for c in totals.index],
                        "Minutes": totals.values})
    res["Share (%)"] = res["Minutes"] / res["Minutes"].sum() * 100
    return res.sort_values("Minutes", ascending=False).reset_index(drop=True)


def top_routes(df: pd.DataFrame, min_flights: int = 1, n: int = 15) -> pd.DataFrame | None:
    if "DESTINATION" not in df.columns:
        return None
    g = group_stats(df, ["ORIGIN", "DESTINATION"], min_flights)
    if g.empty:
        return None
    g.insert(0, "Route", g["ORIGIN"].astype(str) + " -> " + g["DESTINATION"].astype(str))
    return g.sort_values("Delay rate (%)", ascending=False).head(n).drop(
        columns=["ORIGIN", "DESTINATION"]).reset_index(drop=True)


def make_insights(df: pd.DataFrame) -> list[str]:
    """Plain-English findings generated from the data."""
    if df.empty:
        return []
    k = kpis(df)
    min_n = adaptive_min_flights(len(df))
    out = [f"**{k['delay_rate']:.1f}%** of {k['flights']:,} flights arrived late; "
           f"a late flight averaged **{k['avg_when_delayed']:.0f} min**. "
           f"The median flight was {k['median_delay']:+.0f} min, the mean {k['mean_delay']:+.1f} min "
           f"(a gap that signals a long right tail of severe delays)."]

    def extremes(by, label, fmt=lambda v: str(v)):
        g = group_stats(df, by, min_n)
        if len(g) >= 2:
            g = g.sort_values("Delay rate (%)")
            lo, hi = g.iloc[0], g.iloc[-1]
            out.append(f"**{label}:** highest delay rate is **{fmt(hi[by])}** "
                       f"({hi['Delay rate (%)']:.1f}%), lowest is **{fmt(lo[by])}** "
                       f"({lo['Delay rate (%)']:.1f}%), among groups with {min_n}+ flights.")

    extremes("AIRLINE", "Airlines")
    extremes("ORIGIN", "Origin airports")
    if "TIME_OF_DAY" in df and df["TIME_OF_DAY"].notna().any():
        extremes("TIME_OF_DAY", "Time of day")
    if "DAY_NAME" in df and df["DAY_NAME"].notna().any():
        extremes("DAY_NAME", "Day of week")
    if "MONTH" in df and df["MONTH"].notna().any():
        extremes("MONTH", "Month", lambda v: calendar.month_abbr[int(v)])
    causes = cause_breakdown(df)
    if causes is not None:
        top = causes.iloc[0]
        out.append(f"**Root cause:** {top['Cause']} accounts for {top['Share (%)']:.0f}% "
                   "of all delay minutes on late flights.")
    return out


# --------------------------------------------------------------------------
# 5. What-if simulation (works for any airline / airport / time window)
# --------------------------------------------------------------------------
def simulate(df: pd.DataFrame, dimension: str, value, reduction_pct: float,
             threshold: float, passengers_per_flight: float = 100,
             scale_factor: float = 1.0) -> dict:
    """Cut positive arrival delays by `reduction_pct` for rows where dimension == value."""
    mask = (df[dimension].astype(str) == str(value)).to_numpy()
    delay = df["ARRIVAL_DELAY"].to_numpy(dtype="float64")
    new = delay.copy()
    hit = mask & (delay > 0)
    new[hit] = delay[hit] * (1 - reduction_pct / 100)
    minutes_saved = float((delay - new).sum())
    n = len(df)
    return {
        "affected_flights": int(mask.sum()),
        "rate_before": float((delay > threshold).sum() / n * 100) if n else 0.0,
        "rate_after": float((new > threshold).sum() / n * 100) if n else 0.0,
        "minutes_saved": minutes_saved * scale_factor,
        "passenger_hours_saved": minutes_saved * scale_factor * passengers_per_flight / 60,
    }


# --------------------------------------------------------------------------
# 6. Guidance for people who don't know what to upload
# --------------------------------------------------------------------------
FIELD_HELP = {
    "arrival_delay": ("Minutes the flight arrived after schedule (negative = early)", "12, -5, 0"),
    "airline": ("Airline name or code", "DL, United"),
    "origin": ("Departure airport code", "JFK"),
    "destination": ("Arrival airport code", "LAX"),
    "date": ("Flight date", "2024-01-15"),
    "month": ("Month number (use if you have no date)", "1 to 12"),
    "day_of_week": ("Weekday as 1-7 or a name (use if you have no date)", "1 = Mon, or Monday"),
    "sched_dep": ("Scheduled departure time", "0830 or 08:30"),
    "departure_delay": ("Minutes the flight left after schedule", "8"),
    "distance": ("Route distance", "2475"),
    "cancelled": ("1 or yes if the flight was cancelled", "0 or 1"),
    "late_aircraft": ("Delay minutes caused by a late incoming aircraft", "15"),
    "air_system": ("Delay minutes caused by air traffic or airport congestion", "10"),
    "carrier_delay": ("Delay minutes caused by the airline (crew, maintenance)", "5"),
    "weather": ("Delay minutes caused by weather", "0"),
    "security": ("Delay minutes caused by security", "0"),
}

TEMPLATE_COLUMNS = ["FL_DATE", "AIRLINE", "ORIGIN", "DEST", "SCHEDULED_DEPARTURE",
                    "DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE", "CANCELLED",
                    "LATE_AIRCRAFT_DELAY", "AIR_SYSTEM_DELAY", "AIRLINE_DELAY",
                    "WEATHER_DELAY", "SECURITY_DELAY"]
TEMPLATE_ROWS = [
    ["2024-01-15", "DL", "JFK", "LAX", "0830", 8, 12, 2475, 0, 0, 0, 0, 0, 0],
    ["2024-01-15", "AA", "ORD", "DFW", "1745", 40, 52, 802, 0, 30, 22, 0, 0, 0],
    ["2024-01-16", "UA", "SFO", "SEA", "0615", -3, -8, 679, 0, 0, 0, 0, 0, 0],
    ["2024-01-16", "WN", "DEN", "PHX", "2005", 25, 33, 602, 0, 20, 13, 0, 0, 0],
    ["2024-01-17", "DL", "ATL", "MCO", "1210", 0, -5, 404, 0, 0, 0, 0, 0, 0],
    ["2024-01-17", "AA", "MIA", "JFK", "1900", 70, 85, 1089, 0, 40, 0, 45, 0, 0],
    ["2024-01-18", "UA", "EWR", "ORD", "0950", None, None, 719, 1, None, None, None, None, None],
    ["2024-01-18", "B6", "BOS", "DCA", "1430", 5, 18, 399, 0, 0, 10, 0, 8, 0],
    ["2024-01-19", "WN", "LAS", "DEN", "1625", -2, -10, 628, 0, 0, 0, 0, 0, 0],
    ["2024-01-19", "DL", "LGA", "ATL", "1815", 55, 62, 762, 0, 35, 12, 15, 0, 0],
]


def template_csv() -> bytes:
    """A small example file with the recommended column names."""
    return pd.DataFrame(TEMPLATE_ROWS, columns=TEMPLATE_COLUMNS).to_csv(index=False).encode("utf-8")


def requirements_table() -> pd.DataFrame:
    rows = []
    for key, (label, required, synonyms) in FIELDS.items():
        what, example = FIELD_HELP[key]
        rows.append({"Column": label, "Required": "Yes" if required else "Optional",
                     "What it contains": what, "Example": example,
                     "Names we recognise": ", ".join(synonyms[:4])})
    return pd.DataFrame(rows)


# analysis name -> (groups of alternative fields; each group needs one), plain-English need
ANALYSES = [
    ("Delay rate, mean, median and severity split", [["arrival_delay"]], "arrival delay"),
    ("Airline and airport rankings", [["arrival_delay"], ["airline"], ["origin"]], "airline, origin airport"),
    ("Route ranking", [["arrival_delay"], ["destination"]], "destination airport"),
    ("Time-of-day analysis", [["arrival_delay"], ["sched_dep"]], "scheduled departure time"),
    ("Weekday analysis", [["arrival_delay"], ["date", "day_of_week"]], "date or day of week"),
    ("Monthly trend", [["arrival_delay"], ["date", "month"]], "date or month"),
    ("Root-cause breakdown", [["late_aircraft", "air_system", "carrier_delay", "weather", "security"]],
     "at least one delay-cause column"),
    ("What-if simulator", [["arrival_delay"], ["airline"], ["origin"]], "airline, origin airport"),
]


def analysis_availability(mapping: dict) -> pd.DataFrame:
    rows = []
    for name, groups, needs in ANALYSES:
        ok = all(any(mapping.get(f) for f in group) for group in groups)
        rows.append({"Analysis": name, "Status": "Available" if ok else "Missing: " + needs})
    return pd.DataFrame(rows)
    