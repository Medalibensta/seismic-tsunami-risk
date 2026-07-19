"""
Temporal analysis of global seismic activity with Prophet.

Builds a monthly count series of M>=5.5 earthquakes, fits a Prophet model to
expose long-run trend and any yearly seasonality, and exports figures used in
the notebook / report.

Tectonic seismicity is essentially non-seasonal, so the yearly component is
expected to be small; the dominant structure is a slow upward trend driven by
improving global network coverage (a detection artefact, discussed in the
README) plus sharp aftershock spikes after great earthquakes (2004, 2011).

Run:
    python src/timeseries.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

from preprocessing import PROCESSED_CSV

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Prophet-ready (ds, y) frame of events per calendar month."""
    dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    series = (
        dt.dt.to_period("M").value_counts().sort_index()
        .rename_axis("ds").reset_index(name="y")
    )
    series["ds"] = series["ds"].dt.to_timestamp()
    return series


def fit_prophet(series: pd.DataFrame, periods: int = 60) -> tuple[Prophet, pd.DataFrame]:
    """Fit Prophet on the monthly series and forecast `periods` months ahead."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,  # allow trend to bend around big sequences
    )
    model.fit(series)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return model, forecast


def export_figures(df: pd.DataFrame) -> None:
    """Generate and save all time-series figures to reports/figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    series = monthly_counts(df)

    # 1. Annual counts (raw activity + detection-era context).
    annual = df.groupby("year").size()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(annual.index, annual.values, color="#c0392b", alpha=0.85)
    ax.set(title="Global M≥5.5 earthquakes per year (USGS, 1965-2024)",
           xlabel="Year", ylabel="Number of events")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "annual_counts.png", dpi=130)
    plt.close(fig)

    # 2. Prophet forecast.
    model, forecast = fit_prophet(series)
    fig = model.plot(forecast)
    fig.gca().set(title="Prophet forecast — monthly M≥5.5 count",
                  xlabel="Date", ylabel="Events / month")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prophet_forecast.png", dpi=130)
    plt.close(fig)

    # 3. Prophet components (trend + yearly seasonality).
    fig = model.plot_components(forecast)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prophet_components.png", dpi=130)
    plt.close(fig)

    # 4. Monthly energy released (intensity view) — 12-month rolling mean.
    energy = (10 ** df.set_index("datetime")["log_energy"]).astype(float)
    energy.index = pd.to_datetime(energy.index, utc=True).tz_localize(None)
    monthly_energy = energy.resample("MS").sum()
    roll = monthly_energy.rolling(12, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(monthly_energy.index, monthly_energy.values, color="#95a5a6",
            alpha=0.4, label="Monthly")
    ax.plot(roll.index, roll.values, color="#2c3e50", lw=2,
            label="12-month rolling mean")
    ax.set(title="Seismic energy released per month (log-scale)",
           xlabel="Date", ylabel="Energy (Joules)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_energy.png", dpi=130)
    plt.close(fig)

    # Report the seasonality amplitude as a concrete number.
    yearly = forecast.set_index("ds")["yearly"]
    print(f"Yearly-seasonality peak-to-peak amplitude: "
          f"{yearly.max() - yearly.min():.2f} events/month "
          f"(mean level {series['y'].mean():.1f})")
    print(f"Saved 4 figures -> {FIG_DIR}")


def main() -> None:
    matplotlib.use("Agg")  # headless-safe when run as a script
    df = pd.read_csv(PROCESSED_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", utc=True)
    export_figures(df)


if __name__ == "__main__":
    main()
