"""
Synthetic dataset generator for power grid failure prediction.
Produces transformer/feeder-level records with load, weather, and fault history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_ASSETS = 4000

DISCOS = [
    ("Eko DisCo", 6.52, 3.38), ("Ikeja DisCo", 6.60, 3.35),
    ("Ibadan DisCo", 7.38, 3.95), ("Abuja DisCo", 9.07, 7.40),
    ("Enugu DisCo", 6.45, 7.51), ("Kano DisCo", 12.00, 8.52),
    ("Jos DisCo", 9.92, 8.89), ("Kaduna DisCo", 10.52, 7.44),
    ("Benin DisCo", 6.34, 5.63), ("Port Harcourt DisCo", 4.82, 7.05),
    ("Yola DisCo", 9.23, 12.46),
]

ASSET_TYPES = ["Distribution Transformer", "33kV Feeder", "11kV Feeder", "Injection Substation", "Ring Main Unit"]


def generate_grid_dataset(n_assets: int = N_ASSETS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic power grid asset dataset for failure prediction.

    Features include loading percentage, age, weather exposure,
    maintenance history, and fault record counts.
    """
    rng = np.random.default_rng(seed)
    records = []

    for asset_id in range(n_assets):
        disco, lat, lon = DISCOS[rng.integers(0, len(DISCOS))]
        asset_type = ASSET_TYPES[rng.integers(0, len(ASSET_TYPES))]
        age_years = max(0.5, float(rng.exponential(8)))
        rated_capacity_kva = float(rng.choice([100, 200, 300, 500, 1000, 2000, 5000, 15000]))
        loading_pct = float(np.clip(rng.normal(78, 25), 5, 150))
        voltage_deviation_pct = max(0.0, float(rng.exponential(4)))
        ambient_temp_c = float(rng.normal(33, 5))
        humidity_pct = float(np.clip(rng.normal(68, 18), 20, 98))
        rainfall_last30d_mm = max(0.0, float(rng.exponential(80)))
        months_since_last_maintenance = max(0, int(rng.integers(1, 48)))
        n_faults_12months = max(0, int(rng.poisson(1.5)))
        n_faults_3years = max(n_faults_12months, int(rng.poisson(5)))
        harmonic_distortion_pct = max(0.0, float(rng.exponential(3.5)))
        corrosion_index = float(np.clip(rng.beta(2 + age_years * 0.1, 5), 0, 1))
        overload_events_6months = max(0, int(rng.poisson(loading_pct / 60)))
        insulation_resistance_mohm = max(0.1, float(rng.exponential(500 / (age_years + 1))))

        failure_score = (
            (loading_pct / 150) * 0.25
            + min(age_years / 20, 1.0) * 0.20
            + (n_faults_3years / 15) * 0.18
            + (months_since_last_maintenance / 48) * 0.12
            + corrosion_index * 0.10
            + (harmonic_distortion_pct / 20) * 0.08
            + (overload_events_6months / 10) * 0.07
        )
        noise = float(rng.normal(0, 0.04))
        failure_within_90d = int((failure_score + noise) > 0.48)

        records.append({
            "asset_id": asset_id, "disco": disco, "asset_type": asset_type,
            "latitude": round(lat + rng.normal(0, 0.15), 6),
            "longitude": round(lon + rng.normal(0, 0.15), 6),
            "age_years": round(age_years, 1),
            "rated_capacity_kva": rated_capacity_kva,
            "loading_pct": round(loading_pct, 1),
            "voltage_deviation_pct": round(voltage_deviation_pct, 2),
            "ambient_temp_c": round(ambient_temp_c, 1),
            "humidity_pct": round(humidity_pct, 1),
            "rainfall_last30d_mm": round(rainfall_last30d_mm, 1),
            "months_since_last_maintenance": months_since_last_maintenance,
            "n_faults_12months": n_faults_12months,
            "n_faults_3years": n_faults_3years,
            "harmonic_distortion_pct": round(harmonic_distortion_pct, 2),
            "corrosion_index": round(corrosion_index, 4),
            "overload_events_6months": overload_events_6months,
            "insulation_resistance_mohm": round(insulation_resistance_mohm, 2),
            "failure_within_90d": failure_within_90d,
        })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_grid_dataset()
    path = output_dir / "grid_data.csv"
    df.to_csv(path, index=False)
    return path
