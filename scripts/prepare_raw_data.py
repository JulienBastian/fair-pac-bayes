#!/usr/bin/env python3
"""
Prepare raw datasets for offline cluster runs.

This script is meant to run on a machine with internet access.
It downloads raw sources once and stores them under data/raw/.
Later, init.py can load these local files without network access.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from folktables import ACSDataSource
from aif360.sklearn.datasets import (
    fetch_compas,
    fetch_german,
    fetch_lawschool_gpa,
    fetch_meps,
)


def _save_df(df: pd.DataFrame, out_base: Path) -> Path:
    """
    Save DataFrame with a robust strategy:
    - prefer parquet (compact + schema)
    - fallback to pickle if parquet engine is unavailable.
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = out_base.with_suffix(".parquet")
    pkl_path = out_base.with_suffix(".pkl")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_pickle(pkl_path)
        return pkl_path


def prepare_acs(raw_root: Path) -> None:
    out_base = raw_root / "acs" / "acs_2018_1Y_person_AL_AZ"
    print("[ACS] downloading 2018 1-Year person data for states AL/AZ...")
    source = ACSDataSource(
        survey_year="2018",
        horizon="1-Year",
        survey="person",
    )
    # We store raw table as-is; init.py will still apply task-specific processing.
    acs = source.get_data(states=["AL", "AZ"], download=True)
    saved = _save_df(acs, out_base)
    print(f"[ACS] saved -> {saved}")


def prepare_aif360(raw_root: Path) -> None:
    out_dir = raw_root / "aif360"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[AIF360] downloading COMPAS...")
    compas = fetch_compas(binary_race=True, numeric_only=True)
    c_x = _save_df(compas.X, out_dir / "compas_X")
    np.save(out_dir / "compas_y.npy", compas.y.to_numpy())
    print(f"[AIF360] saved -> {c_x}, {out_dir / 'compas_y.npy'}")

    print("[AIF360] downloading GERMAN...")
    german = fetch_german(numeric_only=True, binary_age=True)
    g_x = _save_df(german.X, out_dir / "german_X")
    np.save(out_dir / "german_y.npy", german.y.to_numpy())
    print(f"[AIF360] saved -> {g_x}, {out_dir / 'german_y.npy'}")

    print("[AIF360] downloading LAW...")
    law = fetch_lawschool_gpa(numeric_only=True)
    l_x = _save_df(law.X, out_dir / "law_X")
    np.save(out_dir / "law_y.npy", law.y.to_numpy())
    print(f"[AIF360] saved -> {l_x}, {out_dir / 'law_y.npy'}")

    print("[AIF360] downloading MEPS...")
    meps = fetch_meps(panel=19, accept_terms=True)
    m_x = _save_df(meps.X, out_dir / "meps_X")
    np.save(out_dir / "meps_y.npy", meps.y.to_numpy())
    print(f"[AIF360] saved -> {m_x}, {out_dir / 'meps_y.npy'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw data for offline usage.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for raw cached files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["acs", "aif360"],
        choices=["acs", "aif360"],
        help="Subset to prepare.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving raw data into: {raw_root.resolve()}")

    if "acs" in args.datasets:
        prepare_acs(raw_root)
    if "aif360" in args.datasets:
        prepare_aif360(raw_root)

    print("\nDone.")
    print(f"Now transfer this directory to cluster: {raw_root.resolve()}")


if __name__ == "__main__":
    main()
