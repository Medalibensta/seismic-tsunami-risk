"""
USGS Earthquake Catalog ingestion.

Pulls significant earthquakes (magnitude >= MIN_MAG) from the USGS FDSN event
web service, paginated year-by-year to stay under the 20,000-events-per-request
limit, and writes a single flat CSV to data/raw/.

Source: https://earthquake.usgs.gov/fdsnws/event/1/
Docs:   https://earthquake.usgs.gov/fdsnws/event/1/ (GeoJSON format)

The USGS `tsunami` property is a 0/1 flag set when the event occurs in an
oceanic region where a tsunami product may be issued. It is the target used by
the classic Kaggle "significant earthquakes" tsunami-risk task, so we reuse it
here for the binary classification stage.

Run:
    python src/data_download.py --start 1965 --end 2024 --min-mag 5.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch_year(year: int, min_mag: float, session: requests.Session,
               max_retries: int = 4) -> list[dict]:
    """Fetch one calendar year of events as a list of flat record dicts.

    Uses GeoJSON output; a single year of M>=5.5 events is comfortably under the
    20k cap, so no intra-year pagination is needed.
    """
    params = {
        "format": "geojson",
        "starttime": f"{year}-01-01",
        "endtime": f"{year + 1}-01-01",
        "minmagnitude": min_mag,
        "orderby": "time-asc",
    }
    for attempt in range(max_retries):
        try:
            resp = session.get(USGS_URL, params=params, timeout=60)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            return [_flatten(f) for f in features]
        except (requests.RequestException, ValueError) as exc:
            wait = 2 ** attempt
            print(f"  [year {year}] attempt {attempt + 1} failed ({exc}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch year {year} after {max_retries} tries")


def _flatten(feature: dict) -> dict:
    """Flatten one GeoJSON feature into a single-level record."""
    props = feature.get("properties", {})
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None, None]
    return {
        "id": feature.get("id"),
        "time": props.get("time"),          # epoch ms
        "longitude": coords[0],
        "latitude": coords[1],
        "depth": coords[2],                 # km
        "mag": props.get("mag"),
        "magType": props.get("magType"),
        "place": props.get("place"),
        "type": props.get("type"),
        "tsunami": props.get("tsunami"),    # 0/1 target
        "sig": props.get("sig"),            # USGS significance score
        "felt": props.get("felt"),          # # of felt reports
        "cdi": props.get("cdi"),            # community internet intensity
        "mmi": props.get("mmi"),            # instrumental intensity
        "alert": props.get("alert"),        # green/yellow/orange/red
        "nst": props.get("nst"),            # # of seismic stations
        "gap": props.get("gap"),            # azimuthal gap
        "dmin": props.get("dmin"),          # dist to nearest station
        "rms": props.get("rms"),            # travel-time residual
    }


def download(start: int, end: int, min_mag: float) -> pd.DataFrame:
    """Download [start, end] inclusive and return the concatenated DataFrame."""
    session = requests.Session()
    session.headers.update({"User-Agent": "portfolio-seismic-project/1.0"})
    records: list[dict] = []
    for year in range(start, end + 1):
        year_records = fetch_year(year, min_mag, session)
        records.extend(year_records)
        print(f"  {year}: {len(year_records):>5} events "
              f"(running total {len(records)})")
        time.sleep(0.3)  # be polite to the API
    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download USGS earthquake catalog")
    parser.add_argument("--start", type=int, default=1965)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--min-mag", type=float, default=5.5)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RAW_DIR / "usgs_earthquakes.csv"

    print(f"Downloading M>={args.min_mag} events {args.start}-{args.end} from USGS...")
    df = download(args.start, args.end, args.min_mag)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} unique events -> {out_path}")
    print(f"Tsunami-flagged events: {int(df['tsunami'].fillna(0).sum()):,} "
          f"({100 * df['tsunami'].fillna(0).mean():.2f}%)")


if __name__ == "__main__":
    main()
