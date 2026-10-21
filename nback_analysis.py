#!/usr/bin/env python
"""
N-back preprocessing and analysis: mean reaction time and d' per subject and
time point (T1, T2), for the 0-back and 2-back conditions.

Expected input layout (raw PsychoPy output, folder structure can be messy):
    data/data_nback/
        <subject_folder>/
            ...any sub-folders.../
                <file with "back" and "T1" or "T2" in its name>.csv

Steps:
    1. Preprocessing: for each subject, find the N-back csv for T1 and T2
       (the largest file wins if there are duplicates) and copy it to
       data/data_nback/Clean_Nback_Data/<SubjectID>_<T1|T2>_nback.csv
    2. Reaction time: mean RT (button press time - stimulus onset) for 0-back
       and 2-back.
    3. d' (sensitivity): from hits and false alarms, with a +0.5 correction
       to avoid infinite z-scores at 0% or 100% rates.
    4. Merge RT and d' and save to analysis/nback_results.csv

Adapted from a script shared by another researcher.
"""

import ast
import shutil
from pathlib import Path

import pandas as pd
import scipy.stats as stats

# --- CONFIGURATION ---
# Set these paths to match your own folder structure.
DATA_ROOT = Path("data/data_nback")
CLEAN_DIR_NAME = "Clean_Nback_Data"
CLEAN_DIR = DATA_ROOT / CLEAN_DIR_NAME
OUTPUT_DIR = Path("analysis")
OUTPUT_FILE = OUTPUT_DIR / "nback_results.csv"

CONDITIONS = ("0back", "2back")


# ---------------------------------------------------------------------------
# 1. PREPROCESSING
# ---------------------------------------------------------------------------
def pick_largest(files):
    """From a list of (path, size) tuples, return the path of the largest file."""
    return sorted(files, key=lambda x: x[1])[-1][0]


def clean_nback_files():
    """Copy one N-back csv per subject and time point into CLEAN_DIR.

    PsychoPy's default folder structure is messy, so this sorts it out.
    """
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)  # safe when rerunning

    print(f"Searching for files in: {DATA_ROOT}")
    print(f"Clean files will be saved to: {CLEAN_DIR}")
    print("\n--- Starting Safe Copy ---")

    copied_count = 0

    for subj_folder in DATA_ROOT.iterdir():
        # Only scan subject folders; ignore anything else and the clean folder
        if not subj_folder.is_dir() or subj_folder.name == CLEAN_DIR_NAME:
            continue

        candidates = [f for f in subj_folder.rglob("*.csv") if "back" in f.name.lower()]
        if not candidates:
            continue

        # Bucket into T1 / T2, keeping (file, size) so duplicates can be resolved
        t1_files, t2_files = [], []
        for f in candidates:
            fname = f.name.upper()
            fsize = f.stat().st_size
            if "T1" in fname:
                t1_files.append((f, fsize))
            elif "T2" in fname:
                t2_files.append((f, fsize))

        # Select winners (largest file per time point)
        targets = []
        if t1_files:
            targets.append(("T1", pick_largest(t1_files)))
        if t2_files:
            targets.append(("T2", pick_largest(t2_files)))

        # Copy to the clean folder, using the folder name as SubjectID
        for tp, original_file in targets:
            destination = CLEAN_DIR / f"{subj_folder.name}_{tp}_nback.csv"
            try:
                shutil.copy2(original_file, destination)  # keeps timestamps
                print(f"Copied: {subj_folder.name} {tp}")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {original_file.name}: {e}")

    print(f"\nSuccess! Copied {copied_count} files to '{CLEAN_DIR_NAME}'.")


# ---------------------------------------------------------------------------
# 2. REACTION TIME
# ---------------------------------------------------------------------------
def parse_list(x):
    """Turn a string like '[1.23, 4.56]' into its first element (1.23).

    Returns None if the value is not a non-empty list string.
    """
    if isinstance(x, str) and x.startswith("["):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list) and len(val) > 0:
                return val[0]
        except (ValueError, SyntaxError):
            return None
    return None


def parse_filename(csv_file):
    """Extract (SubjectID, time point) from e.g. 'SUB001_T1_nback.csv'."""
    parts = csv_file.name.split("_")
    return parts[0], parts[1]


def compute_rt():
    """Mean RT per subject/time point for 0-back and 2-back."""
    results = []
    print(f"--- Analyzing files in {CLEAN_DIR.name} ---")

    for csv_file in CLEAN_DIR.glob("*_nback.csv"):
        subj_id, tp = parse_filename(csv_file)

        try:
            df = pd.read_csv(csv_file, delimiter="\t")
            means = {}

            for cond in CONDITIONS:
                times_col = f"buttonBox_{cond}.times"
                onset_col = f"letter_{cond}.started"
                # Check that the column exists to be safe
                if times_col in df.columns:
                    press = df[times_col].apply(parse_list)
                    means[cond] = (press - df[onset_col]).mean()
                else:
                    means[cond] = None

            results.append({
                "SubjectID": subj_id,
                "Time": tp,
                "NBack_0_RT": means["0back"],
                "NBack_2_RT": means["2back"],
            })

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")

    nback_df = pd.DataFrame(results).sort_values(["SubjectID", "Time"])
    print(f"\nRT analysis complete. Rows: {len(nback_df)}")
    return nback_df


# ---------------------------------------------------------------------------
# 3. D PRIME
# ---------------------------------------------------------------------------
def calculate_dprime(hits, misses, false_alarms, total_targets, total_non_targets):
    """d' = z(hit rate) - z(false alarm rate), with a +0.5 correction."""
    hr_adj = (hits + 0.5) / (total_targets + 1) if total_targets > 0 else 0.5
    far_adj = (false_alarms + 0.5) / (total_non_targets + 1) if total_non_targets > 0 else 0.5
    return stats.norm.ppf(hr_adj) - stats.norm.ppf(far_adj)


def has_press(btn):
    """True if the button column holds a press, e.g. the string "['space']"."""
    return isinstance(btn, str) and btn.strip() not in ("None", "[]")


def compute_dprime():
    """d' per subject/time point for 0-back and 2-back."""
    results = []
    print(f"--- Calculating d' (Accuracy) in {CLEAN_DIR.name} ---")

    for csv_file in CLEAN_DIR.glob("*_nback.csv"):
        subj_id, tp = parse_filename(csv_file)

        try:
            df = pd.read_csv(csv_file, delimiter="\t")

            counts = {
                cond: {"hits": 0, "misses": 0, "fa": 0, "targets": 0, "non_targets": 0}
                for cond in CONDITIONS
            }

            # Row-by-row processing (safest for N-back logic)
            for _, row in df.iterrows():
                for cond in CONDITIONS:
                    btn_col = f"buttonBox_{cond}.buttons"
                    target_col = f"target_{cond}"
                    if btn_col not in df.columns or target_col not in df.columns:
                        continue

                    pressed = has_press(row[btn_col])
                    c = counts[cond]

                    if row[target_col] == 1:  # target trial: hit or miss
                        c["targets"] += 1
                        if pressed:
                            c["hits"] += 1
                        else:
                            c["misses"] += 1
                    else:  # non-target trial: press = false alarm
                        c["non_targets"] += 1
                        if pressed:
                            c["fa"] += 1

            d = {
                cond: calculate_dprime(
                    counts[cond]["hits"],
                    counts[cond]["misses"],
                    counts[cond]["fa"],
                    counts[cond]["targets"],
                    counts[cond]["non_targets"],
                )
                for cond in CONDITIONS
            }

            results.append({
                "SubjectID": subj_id,
                "Time": tp,
                "dprime_0": d["0back"],
                "dprime_2": d["2back"],
            })

        except Exception as e:
            print(f"Error in {csv_file.name}: {e}")

    acc_df = pd.DataFrame(results).sort_values(["SubjectID", "Time"])
    print(f"\nProcessed {len(acc_df)} sessions.")
    return acc_df


# ---------------------------------------------------------------------------
# 4. MERGE AND SAVE
# ---------------------------------------------------------------------------
def main():
    if not DATA_ROOT.is_dir():
        raise SystemExit(f"Data folder not found: {DATA_ROOT.resolve()}")

    clean_nback_files()
    nback_df = compute_rt()
    nback_acc_df = compute_dprime()

    if nback_df.empty or nback_acc_df.empty:
        raise SystemExit("No N-back files could be analysed.")

    # Merge reaction time and accuracy, keeping only the intersection
    master = pd.merge(nback_df, nback_acc_df, on=["SubjectID", "Time"], how="inner")
    master = master[[
        "SubjectID", "Time",
        "NBack_0_RT", "dprime_0",
        "NBack_2_RT", "dprime_2",
    ]]
    master = master.sort_values(["SubjectID", "Time"]).reset_index(drop=True)

    print("Final N-Back Columns:", master.columns.tolist())
    print(f"Total Rows: {len(master)}")
    print(master.to_string(index=False))

    # Save to CSV (index=False avoids an extra column of row numbers)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
