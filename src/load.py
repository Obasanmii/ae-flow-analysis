"""Load and tidy NHS England monthly A&E CSVs (one file per month).

Each file has one row per provider and, in most months, a national TOTAL row.
We drop the TOTAL rows and rebuild national figures from the providers, so every
month is treated the same way, and keep the published totals to reconcile against.
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Short names for the 22 published columns, in file order.
COLUMNS = {
    "Period": "period_raw",
    "Org Code": "org_code",
    "Parent Org": "region",
    "Org name": "org_name",
    "A&E attendances Type 1": "att_t1",
    "A&E attendances Type 2": "att_t2",
    "A&E attendances Other A&E Department": "att_t3",
    "A&E attendances Booked Appointments Type 1": "att_booked_t1",
    "A&E attendances Booked Appointments Type 2": "att_booked_t2",
    "A&E attendances Booked Appointments Other Department": "att_booked_t3",
    "Attendances over 4hrs Type 1": "over4_t1",
    "Attendances over 4hrs Type 2": "over4_t2",
    "Attendances over 4hrs Other Department": "over4_t3",
    "Attendances over 4hrs Booked Appointments Type 1": "over4_booked_t1",
    "Attendances over 4hrs Booked Appointments Type 2": "over4_booked_t2",
    "Attendances over 4hrs Booked Appointments Other Department": "over4_booked_t3",
    "Patients who have waited 4-12 hs from DTA to admission": "dta_4_12h",
    "Patients who have waited 12+ hrs from DTA to admission": "dta_12h_plus",
    "Emergency admissions via A&E - Type 1": "adm_t1",
    "Emergency admissions via A&E - Type 2": "adm_t2",
    "Emergency admissions via A&E - Other A&E department": "adm_t3",
    "Other emergency admissions": "adm_other",
}
ATT_ALL = ["att_t1", "att_t2", "att_t3", "att_booked_t1", "att_booked_t2", "att_booked_t3"]
OVER4_ALL = ["over4_t1", "over4_t2", "over4_t3", "over4_booked_t1", "over4_booked_t2", "over4_booked_t3"]


def _month(period_raw: str) -> pd.Timestamp:
    m = re.match(r"MSITAE-([A-Z]+)-(\d{4})", str(period_raw).strip().upper())
    if not m:
        raise ValueError(f"Unrecognised period: {period_raw!r}")
    return pd.to_datetime(f"1 {m.group(1)} {m.group(2)}", format="%d %B %Y")


def read_month(path: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return (provider rows, national TOTAL row or None) for one monthly file."""
    df = pd.read_csv(path).rename(columns=COLUMNS)
    missing = set(COLUMNS.values()) - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {sorted(missing)}")
    # The national total row is usually labelled TOTAL in the Period column, but in
    # February 2026 the label is in Org Code instead. Missing it double-counts England.
    is_total = (df["period_raw"].astype(str).str.strip().str.upper().eq("TOTAL")
                | df["org_code"].astype(str).str.strip().str.upper().eq("TOTAL"))
    total = df[is_total].iloc[0] if is_total.any() else None
    df = df[~is_total].copy()
    df["month"] = df["period_raw"].map(_month)
    for c in ("region", "org_name", "org_code"):
        df[c] = df[c].astype(str).str.strip()
    return df, total


def load_all(raw_dir: str = RAW_DIR) -> tuple[pd.DataFrame, dict]:
    """All months stacked, plus the published TOTAL rows keyed by month (for checks)."""
    frames, totals = [], {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        df, total = read_month(path)
        frames.append(df)
        if total is not None:
            totals[df["month"].iloc[0]] = total
    data = pd.concat(frames, ignore_index=True)
    dupes = data.duplicated(["month", "org_code"])
    if dupes.any():
        raise ValueError(f"{dupes.sum()} duplicate provider-months; check for a month uploaded twice")
    return data, totals


def england_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """National figures per month, all A&E types (the basis of the official 4-hour measure)."""
    g = data.groupby("month")
    out = pd.DataFrame({
        "attendances": g[ATT_ALL].sum().sum(axis=1),
        "over_4h": g[OVER4_ALL].sum().sum(axis=1),
        "att_t1": g["att_t1"].sum(),
        "over4_t1": g["over4_t1"].sum(),
        "dta_12h_plus": g["dta_12h_plus"].sum(),
        "adm_t1": g["adm_t1"].sum(),
    })
    out["pct_4h_all"] = 100 * (1 - out["over_4h"] / out["attendances"])
    out["pct_4h_t1"] = 100 * (1 - out["over4_t1"] / out["att_t1"])
    return out
