"""Synthetic hourly inputs: temperature, per-bus load, wind/solar/hydro availability.

Design notes
------------
* Every stochastic process draws from its own child of one ``SeedSequence``,
  so changing (say) the wind parameters does not change the temperature path.
* The calendar is a naive hourly index (no DST). Real AESO data in Phase 5
  needs proper America/Edmonton / hour-ending handling; the synthetic data
  deliberately sidesteps that.
* Weather anomalies are AR(1) processes: persistent like real weather, but
  simple enough to explain in one line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

STREAMS = ("temperature", "bus_temp", "load_noise", "wind", "solar")


def rngs(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(s) for name, s in zip(STREAMS, children)}


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float, size: int = 1) -> np.ndarray:
    """AR(1) path(s) x_t = phi x_{t-1} + eps_t, started from the stationary distribution."""
    eps = rng.normal(0.0, sigma, size=(size, n))
    eps[:, 0] *= 1.0 / np.sqrt(1.0 - phi**2)  # stationary start
    return lfilter([1.0], [1.0, -phi], eps, axis=1)


def calendar(cfg: dict) -> pd.DataFrame:
    idx = pd.date_range(cfg["time"]["start"], periods=cfg["time"]["hours"], freq="h")
    idx.name = "timestamp"
    return pd.DataFrame(
        {
            "hour": idx.hour,
            "dow": idx.dayofweek,
            "month": idx.month,
            "doy": idx.dayofyear,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
        },
        index=idx,
    )


def temperature(cfg: dict, cal: pd.DataFrame, r: dict) -> pd.DataFrame:
    p = cfg["temperature"]
    n = len(cal)
    seasonal = p["annual_mean"] - p["annual_amplitude"] * np.cos(
        2 * np.pi * (cal["doy"].to_numpy() - p["coldest_day_of_year"]) / 365.25
    )
    daily = p["daily_amplitude"] * np.cos(2 * np.pi * (cal["hour"].to_numpy() - p["warmest_hour"]) / 24)
    anomaly = ar1(r["temperature"], n, p["ar_phi"], p["ar_sigma"])[0]
    system = seasonal + daily + anomaly
    out = {}
    for bus, b in cfg["buses"].items():
        if b["load_share"] == 0:
            continue
        out[f"temp_{bus}"] = system + b["temp_offset"] + r["bus_temp"].normal(0, p["bus_noise_sigma"], n)
    return pd.DataFrame(out, index=cal.index)


def load(cfg: dict, cal: pd.DataFrame, temps: pd.DataFrame, r: dict) -> pd.DataFrame:
    p = cfg["load"]
    shape = np.asarray(p["hourly_shape"], float)
    shape = shape / shape.mean()
    hour_f = shape[cal["hour"].to_numpy()]
    week_f = np.where(cal["is_weekend"].to_numpy() == 1, p["weekend_factor"], 1.0)
    out = {}
    for bus, b in cfg["buses"].items():
        if b["load_share"] == 0:
            continue
        t = temps[f"temp_{bus}"].to_numpy()
        temp_f = (
            1.0
            + p["heating_coef"] * np.maximum(0.0, p["heating_base_C"] - t)
            + p["cooling_coef"] * np.maximum(0.0, t - p["cooling_base_C"])
        )
        f = hour_f * week_f * temp_f
        if bus in p["industrial_buses"]:
            f = 1.0 + p["industrial_shape_weight"] * (f - 1.0)
        noise = 1.0 + r["load_noise"].normal(0, p["noise_sigma"], len(cal))
        out[f"load_{bus}"] = p["mean_total_mw"] * b["load_share"] * f * noise
    return pd.DataFrame(out, index=cal.index)


def wind(cfg: dict, cal: pd.DataFrame, r: dict) -> pd.DataFrame:
    p = cfg["wind"]
    n = len(cal)
    mu = p["mean_logit"] + p["seasonal_amplitude"] * np.cos(
        2 * np.pi * (cal["doy"].to_numpy() - p["windiest_day_of_year"]) / 365.25
    )
    z = ar1(r["wind"], n, p["ar_phi"], p["ar_sigma"], size=2)
    rho = p["site_correlation"]
    z_south = z[0]
    z_central = rho * z[0] + np.sqrt(1 - rho**2) * z[1]
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    # scale logit anomaly up so the CF spans ~0..1 (AR std is ~1.3 in logit space)
    return pd.DataFrame(
        {"cf_wind_south": sig(mu + 1.8 * z_south), "cf_wind_central": sig(mu - 0.1 + 1.8 * z_central)},
        index=cal.index,
    )


def solar(cfg: dict, cal: pd.DataFrame, r: dict) -> pd.DataFrame:
    p = cfg["solar"]
    lat = np.deg2rad(p["latitude_deg"])
    doy = cal["doy"].to_numpy()
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    # mid-hour local standard time; solar noon in Alberta is ~13:00 MST
    solar_hour = cal["hour"].to_numpy() + 0.5 - 1.0
    omega = np.deg2rad(15.0 * (solar_hour - 12.0))
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(omega)
    max_sin = np.sin(np.pi / 2 - lat + np.deg2rad(23.44))
    clear = p["peak_cf"] * np.clip(sin_el, 0, None) / max_sin
    cloud = np.clip(p["cloud_mean"] + ar1(r["solar"], len(cal), p["cloud_ar_phi"], p["cloud_ar_sigma"])[0], 0, 0.95)
    return pd.DataFrame({"cf_solar_south": clear * (1 - cloud)}, index=cal.index)


def hydro(cfg: dict, cal: pd.DataFrame) -> pd.DataFrame:
    monthly = np.asarray(cfg["hydro"]["monthly_pmax_pu"], float)
    return pd.DataFrame({"pu_hydro": monthly[cal["month"].to_numpy() - 1]}, index=cal.index)


def generate_inputs(cfg: dict) -> pd.DataFrame:
    """One DataFrame with calendar, temperature, per-bus load, and availability profiles."""
    r = rngs(cfg["seed"])
    cal = calendar(cfg)
    temps = temperature(cfg, cal, r)
    loads = load(cfg, cal, temps, r)
    return pd.concat([cal, temps, loads, wind(cfg, cal, r), solar(cfg, cal, r), hydro(cfg, cal)], axis=1)


PROFILE_COLUMNS = {
    "wind_south": "cf_wind_south",
    "wind_central": "cf_wind_central",
    "solar_south": "cf_solar_south",
    "hydro": "pu_hydro",
}
