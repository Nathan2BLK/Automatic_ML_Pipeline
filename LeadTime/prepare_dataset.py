# prediction_model/LeadTime/prepare_dataset.py
import os, sys, pathlib
SRC = pathlib.Path(__file__).resolve().parents[2]  # -> .../src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conf import config_ahp_factory
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

DEFAULT_IN  = Path(r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\train_data\output_initiative.csv")
DEFAULT_OUT = Path(r"\\ncehospms01\AHP_Factory\prediction_model\LeadTime\train_data\filtered_dataset_final.csv")

def prepare_dataset(input_csv: Path | str = DEFAULT_IN,
                    output_csv: Path | str = DEFAULT_OUT) -> Tuple[pd.DataFrame, str]:
    input_csv  = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv, low_memory=False)

    # Convert release dates
    release_dates = {k: pd.to_datetime(v['dev_cutoff']) for k, v in config_ahp_factory.by_release.items()}
    df["dev_roadmap_date"] = df["epic_dev_roadmap"].map(release_dates)
    df["fix_version_date"] = df["epic_fix_version"].map(release_dates)

    error = []

    def process_dev_roadmap(roadmap_str):
        if pd.isna(roadmap_str):
            return None
        valid_dates = []
        roadmaps = str(roadmap_str).replace(",", "#").split("#")
        for r in roadmaps:
            r = r.strip()
            if r in release_dates and r != 'to_plan' and 'hit-pi' not in r:
                valid_dates.append(release_dates[r])
            else:
                error.append(r)
        return min(valid_dates) if valid_dates else None

    def process_fix_version(fix_str):
        if pd.isna(fix_str):
            return None
        valid_dates = []
        versions = str(fix_str).split("#")
        for r in versions:
            r = r.strip()
            if r in release_dates and r != 'to_plan' and 'hit-pi' not in r:
                valid_dates.append(release_dates[r])
            else:
                error.append(r)
        return max(valid_dates) if valid_dates else None

    df["dev_roadmap_date"] = df["epic_dev_roadmap"].apply(process_dev_roadmap)
    df["fix_version_date"] = df["epic_fix_version"].apply(process_fix_version)
    df["existing_lead_time"] = (df["fix_version_date"] - df["dev_roadmap_date"]).dt.days

    df["extract_date"] = pd.to_datetime(df["extract_date"], errors="coerce")
    df = df[df["extract_date"] >= pd.Timestamp("2022-01-01")]

    statuses_to_keep = ["Accepted", "Completed", "Development Completed", "Done", "E2E Validated for Community"]
    finished_tickets = df[
        (df["existing_lead_time"].notna()) &
        (df["existing_lead_time"] >= 0) &
        (df["epic_status"].isin(statuses_to_keep)) &
        (df["epic_level"] == "ART Epic")
    ]

    finished_tickets = finished_tickets.loc[
        finished_tickets.groupby("key")["extract_date"].idxmax()
    ]

    lead_time_mapping = finished_tickets.set_index("key")["existing_lead_time"]
    df["needed_lead_time"] = df["key"].map(lead_time_mapping)

    final_tickets = df[df["key"].isin(finished_tickets["key"])].copy()
    assert final_tickets["needed_lead_time"].isna().sum() == 0, "🚨 Some keys did not get mapped to needed_lead_time!"
    final_tickets = final_tickets[final_tickets["needed_lead_time"] > 0]

    for col in ["needed_lead_time", "existing_lead_time"]:
        final_tickets[col] = final_tickets[col].apply(lambda x: x if pd.notna(x) and x >= 0 else 0)
        mask = final_tickets[col] > 0
        values = final_tickets.loc[mask, col].astype(float)
        quarters = np.ceil(values / 90).clip(lower=1).astype(int)
        final_tickets.loc[mask, col] = quarters

    missing_rate = df.isna().mean()
    cols_to_drop = missing_rate[missing_rate > 0.95].index.tolist()
    final_tickets = final_tickets.drop_duplicates(
        subset=[c for c in final_tickets.columns if c != "extract_date"]
    ).drop(columns=cols_to_drop, errors="ignore")

    print(final_tickets['needed_lead_time'].value_counts())
    final_tickets["needed_lead_time"] = final_tickets["needed_lead_time"].apply(lambda x: "8+" if x >= 8 else str(int(x)))
    print(final_tickets['needed_lead_time'].value_counts())
    print(cols_to_drop)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_tickets.to_csv(output_csv, index=False)
    print(f"✅ Final dataset saved to {output_csv} — shape: {final_tickets.shape}")
    return final_tickets, str(output_csv)

if __name__ == "__main__":
    # Zero-argument friendly run
    prepare_dataset()
