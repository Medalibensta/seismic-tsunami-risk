"""
Cleaning and feature engineering for the USGS earthquake catalog.

The raw catalog mixes natural earthquakes with a few hundred anthropogenic
events (nuclear tests, explosions). Intensity / network-quality fields
(felt, cdi, mmi, alert, nst, gap, dmin, rms) are only populated for recent,
well-instrumented events, so they are handled with median imputation plus an
explicit "was-missing" indicator rather than being dropped.

Public helpers:
    load_raw()                 -> raw DataFrame
    clean(df)                  -> earthquake-only, typed, with datetime parts
    build_features(df)         -> adds physical / geographic features
    classification_matrix(df)  -> (X, y) for tsunami prediction
    regression_matrix(df)      -> (X, y) for magnitude prediction
    build_processed()          -> runs the full pipeline, writes processed CSV
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_CSV = DATA_DIR / "raw" / "usgs_earthquakes.csv"
PROCESSED_CSV = DATA_DIR / "processed" / "earthquakes_clean.csv"

RANDOM_STATE = 42

# Sparse quality/intensity fields imputed with a missingness flag.
SPARSE_QUALITY_COLS = ["felt", "cdi", "mmi", "nst", "gap", "dmin", "rms"]


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Load the raw USGS CSV produced by data_download.py."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to natural earthquakes and add parsed datetime components.

    - Drops non-earthquake event types (nuclear tests, explosions).
    - Drops rows missing any of the four core physical fields (mag, depth,
      latitude, longitude) — after the M>=5.5 pull these are always present,
      but the guard keeps the function safe on arbitrary inputs.
    - Parses the epoch-millisecond `time` into a UTC datetime and extracts
      calendar parts used by the time-series and modelling stages.
    """
    df = df.copy()
    df = df[df["type"] == "earthquake"].copy()

    core = ["mag", "depth", "latitude", "longitude"]
    df = df.dropna(subset=core).reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["decade"] = (df["year"] // 10) * 10

    # Physically implausible depths clipped to the sea-level datum.
    df["depth"] = df["depth"].clip(lower=0)

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physical and geographic features used by the models.

    - `log_energy`: seismic energy in log10 Joules via Gutenberg-Richter
      (log10 E = 1.5 * M + 4.8).
    - `depth_class`: shallow (<70 km) / intermediate (70-300) / deep (>300),
      the standard seismological banding.
    - `abs_lat`, `hemisphere`: coarse latitude descriptors.
    - one-hot of `magType` grouped into the common scales.
    """
    df = df.copy()

    df["log_energy"] = 1.5 * df["mag"] + 4.8
    df["is_shallow"] = (df["depth"] < 70).astype(int)
    df["depth_class"] = pd.cut(
        df["depth"], bins=[-1, 70, 300, 1000],
        labels=["shallow", "intermediate", "deep"],
    )
    df["abs_lat"] = df["latitude"].abs()
    df["hemisphere_n"] = (df["latitude"] >= 0).astype(int)

    # Group magnitude types into families (moment / body / surface / local).
    def _mag_family(mt: str) -> str:
        mt = str(mt).lower()
        if mt.startswith("mw"):
            return "moment"
        if mt.startswith("mb"):
            return "body"
        if mt.startswith("ms"):
            return "surface"
        if mt.startswith("ml") or mt.startswith("md"):
            return "local"
        return "other"

    df["mag_family"] = df["magType"].apply(_mag_family)
    return df


def _impute_with_flags(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Median-impute `cols` in place and append a `<col>_missing` indicator."""
    X = X.copy()
    for col in cols:
        if col not in X.columns:
            continue
        X[f"{col}_missing"] = X[col].isna().astype(int)
        X[col] = X[col].fillna(X[col].median())
    return X


def classification_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for tsunami classification.

    Uses only features known immediately from event detection — location,
    depth, magnitude, and their derived physical descriptors — to avoid
    leaking downstream impact assessments into the tsunami flag.
    """
    feats = [
        "mag", "depth", "latitude", "longitude", "abs_lat",
        "log_energy", "is_shallow", "hemisphere_n",
    ]
    X = df[feats].copy()
    X = pd.get_dummies(
        pd.concat([X, df[["mag_family", "depth_class"]]], axis=1),
        columns=["mag_family", "depth_class"], drop_first=True,
    )
    y = df["tsunami"].astype(int)
    return X, y


def regression_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for magnitude regression.

    Predicts magnitude from depth, geographic coordinates and the available
    intensity / network-quality measures (median-imputed with missingness
    flags). Energy-derived columns are excluded because they are a
    deterministic function of the magnitude target.
    """
    base_feats = ["depth", "latitude", "longitude", "abs_lat", "is_shallow"]
    X = df[base_feats].copy()
    X = _impute_with_flags(pd.concat([X, df[SPARSE_QUALITY_COLS]], axis=1),
                           SPARSE_QUALITY_COLS)
    X = pd.get_dummies(
        pd.concat([X, df[["depth_class"]]], axis=1),
        columns=["depth_class"], drop_first=True,
    )
    y = df["mag"].astype(float)
    return X, y


def build_processed() -> pd.DataFrame:
    """Run clean + build_features and persist to data/processed/."""
    df = build_features(clean(load_raw()))
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Processed {len(df):,} earthquakes -> {PROCESSED_CSV}")
    print(f"Tsunami rate: {100 * df['tsunami'].mean():.2f}%  "
          f"| years {df['year'].min()}-{df['year'].max()}")
    return df


if __name__ == "__main__":
    build_processed()
