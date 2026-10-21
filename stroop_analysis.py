#!/usr/bin/env python
"""
Compute the Stroop effect and accuracy per subject for two time points (T1, T2).

Expected input layout:
    data/data_stroop/
        <subject_folder>/
            ...any sub-folders.../
                <file containing "T1" or "T2" in its name>.csv

Each CSV must contain the columns:
    condition  - 'congruent' or 'incongruent'
    rt         - reaction time
    correct    - 1 for a correct trial, 0 otherwise

Output:
    analysis/stroop_results.csv
        One row per subject and time point, with columns
        SubjectID, Time, Stroop_Effect, Stroop_Accuracy.

Stroop effect = median RT (incongruent) - median RT (congruent),
computed on correct trials only. Medians are used and no RTs are filtered out.
"""

from pathlib import Path
import pandas as pd

# Set these paths to match your own folder structure.
DATA_ROOT = Path("data/data_stroop")
OUTPUT_DIR = Path("analysis")
OUTPUT_FILE = OUTPUT_DIR / "stroop_results.csv"

# Time points to analyse. T3 etc. are skipped; change accordingly.
TIME_POINTS = ("T1", "T2")


def get_time_point(filename):
    """Return 'T1' or 'T2' if found in the file name, otherwise None."""
    name = filename.upper()
    for tp in TIME_POINTS:
        if tp in name:
            return tp
    return None


def analyse_subject(subj_folder):
    """Return a list of result dicts (one per time point) for one subject."""
    subj_data = []

    # Search recursively for csv files, look into any possible folder
    for csv_file in subj_folder.rglob("*.csv"):
        tp = get_time_point(csv_file.name)
        if tp is None:
            continue

        try:
            df = pd.read_csv(csv_file)
            clean = df[df["correct"] == 1]  # keep only correct trials
            medians = clean.groupby("condition")["rt"].median()

            if "incongruent" in medians and "congruent" in medians:
                subj_data.append({
                    "SubjectID": subj_folder.name,
                    "Time": tp,
                    "Stroop_Effect": medians["incongruent"] - medians["congruent"],
                    "Stroop_Accuracy": df["correct"].mean() * 100,
                })
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")

    return subj_data


def main():
    if not DATA_ROOT.is_dir():
        raise SystemExit(f"Data folder not found: {DATA_ROOT.resolve()}")

    # Sanity check: show how many subject folders exist
    folders = [f.name for f in DATA_ROOT.iterdir() if f.is_dir()]
    print(f"Found {len(folders)} folders. First few: {folders[:3]}")

    all_results = []
    for subj_folder in DATA_ROOT.iterdir():
        if not subj_folder.is_dir():
            continue

        subj_data = analyse_subject(subj_folder)

        # Only keep subjects with data for every time point
        if len(subj_data) == len(TIME_POINTS):
            all_results.extend(subj_data)
        else:
            print(f"Skipping {subj_folder.name}: Found {len(subj_data)} sessions, "
                  f"need {len(TIME_POINTS)}.")

    if not all_results:
        raise SystemExit("No subjects with complete data were found.")

    summary_df = pd.DataFrame(all_results).sort_values(["SubjectID", "Time"])
    print(f"\nFinal count: {summary_df['SubjectID'].nunique()} subjects with complete data.")
    print(summary_df.to_string(index=False))

    # Save to CSV (index=False avoids an extra column of row numbers)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
