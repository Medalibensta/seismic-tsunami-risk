"""
Geospatial visualisation of the earthquake catalogue.

Produces:
    * reports/figures/earthquake_map.html   — interactive Plotly world map,
      every M>=5.5 event, sized/coloured by magnitude, tsunami events flagged.
    * reports/figures/tsunami_hotspots.png   — static map of tsunami-flagged
      events for the README.

Run:
    python src/geoviz.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

from preprocessing import PROCESSED_CSV

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def interactive_map(df: pd.DataFrame) -> None:
    """Interactive world map of all events, coloured by magnitude."""
    plot_df = df[["latitude", "longitude", "mag", "depth", "year",
                  "tsunami", "place"]].copy()
    plot_df["Event"] = plot_df["tsunami"].map({0: "No tsunami flag",
                                               1: "Tsunami flag"})
    fig = px.scatter_geo(
        plot_df, lat="latitude", lon="longitude",
        color="mag", size="mag",
        hover_name="place",
        hover_data={"mag": ":.1f", "depth": ":.0f", "year": True,
                    "latitude": False, "longitude": False},
        color_continuous_scale="Inferno",
        projection="natural earth",
        title="Global M≥5.5 earthquakes 1965-2024 (USGS) — colour = magnitude",
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    out = FIG_DIR / "earthquake_map.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Saved interactive map -> {out}")


def tsunami_hotspots(df: pd.DataFrame) -> None:
    """Static map: tsunami-flagged events over the full catalogue."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.scatter(df["longitude"], df["latitude"], s=3, alpha=0.15,
               color="#bdc3c7", label="All events")
    ts = df[df["tsunami"] == 1]
    ax.scatter(ts["longitude"], ts["latitude"], s=18, alpha=0.7,
               color="#c0392b", edgecolor="k", linewidth=0.2,
               label="Tsunami-flagged")
    ax.set(title="Tsunami-flagged earthquakes cluster on subduction margins",
           xlabel="Longitude", ylabel="Latitude",
           xlim=(-180, 180), ylim=(-90, 90))
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "tsunami_hotspots.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved static hotspot map -> {out}")


def main() -> None:
    matplotlib.use("Agg")  # headless-safe when run as a script
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_CSV)
    interactive_map(df)
    tsunami_hotspots(df)


if __name__ == "__main__":
    main()
