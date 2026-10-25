#!/usr/bin/env python
# coding: utf-8


# N-BACK PREPROCESSING


import shutil
import os # only for mac os
from pathlib import Path

# --- CONFIGURATION ---
# 1. N-Back raw data folder
nback_root = Path("data/data_nback")

# N-back folders are very messy due to PsychoPy default folder structure, so in the next step I sort it out

# 2. Create the clean folder inside that same N-back root
clean_dir = nback_root / "Clean_Nback_Data"
clean_dir.mkdir(exist_ok=True) # safety line for when rerunning the script

print(f"Searching for files in: {nback_root}")
print(f"Clean files will be saved to: {clean_dir}")

copied_count = 0

print("\n--- Starting Safe Copy ---")

# 3. Iterate through all subject folders and look for the biggest csv table

for subj_folder in nback_root.iterdir():
    if not subj_folder.is_dir() or subj_folder.name == "Clean_Nback_Data": 
        continue # only subject folders should be scanned through, ignore anything else and ignore the clean data folder   

    candidates = [
        f for f in subj_folder.rglob("*.csv") 
        if "back" in f.name.lower() 
    ]

    if not candidates: continue # it will look into every folder but focus only on the csv files

    # 4. Bucket into T1 / T2
    t1_files = []
    t2_files = []

    for f in candidates:
        fname = f.name.upper() # converting all file names into uppercase
        fsize = f.stat().st_size # in case there is a name duplicate, it will assume the biggest one is correct

        if "T1" in fname:
            t1_files.append((f, fsize))
        elif "T2" in fname:
            t2_files.append((f, fsize)) # saving the files as a tuple with name and the weight

    # 5. Select Winners (Largest File)
    targets = []
    if t1_files:
        targets.append(("T1", sorted(t1_files, key=lambda x: x[1])[-1][0])) 
    if t2_files:
        targets.append(("T2", sorted(t2_files, key=lambda x: x[1])[-1][0]))
    # ignore the names list the files based on size [1] and pick the largest (the last item [-1]), now save the file and drop the size when saving [0]    

    # 6. COPY to the new folder
    for tp, original_file in targets:
        # Use folder name as SubjectID (e.g., SUB001)
        new_name = f"{subj_folder.name}_{tp}_nback.csv"
        destination = clean_dir / new_name

        try:
            shutil.copy2(original_file, destination) # copy2 saves the file creation date and time as well
            print(f"Copied: {subj_folder.name} {tp}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {original_file.name}: {e}")

print(f"\n Success! Copied {copied_count} files to 'Clean_Nback_Data'.")


# N-BACK REACTION TIME ANALYSIS
# adapted from a script shared by Valeria Oliva

import pandas as pd
import ast
from pathlib import Path

# --- CONFIGURATION ---
# Define the path to the clean data folder
clean_dir = nback_root / "Clean_Nback_Data" 

# Helper Function for python syntax
def parse_list(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list) and len(val) > 0:
                return val[0]
        except:
            return None
    return None

all_nback_results = []

print(f"--- Analyzing files in {clean_dir.name} ---")

# Iterate directly over the clean files
for csv_file in clean_dir.glob("*_nback.csv"):

    # Extract Info from Filename (SUB001_T1_nback.csv)
    parts = csv_file.name.split('_')
    subj_id = parts[0]
    tp = parts[1]

    try:
        # Read the file
        df = pd.read_csv(csv_file, delimiter='\t')

        # --- 0-BACK RT ---
        # Check if columns exist to be safe
        if 'buttonBox_0back.times' in df.columns:
            df['press_0'] = df['buttonBox_0back.times'].apply(parse_list)
            df['RT_0'] = df['press_0'] - df['letter_0back.started']
            mean_0 = df['RT_0'].mean()
        else:
            mean_0 = None

        # --- 2-BACK RT ---
        if 'buttonBox_2back.times' in df.columns:
            df['press_2'] = df['buttonBox_2back.times'].apply(parse_list)
            df['RT_2'] = df['press_2'] - df['letter_2back.started']
            mean_2 = df['RT_2'].mean()
        else:
            mean_2 = None

        all_nback_results.append({
            'SubjectID': subj_id,
            'Time': tp,
            'NBack_0_RT': mean_0,
            'NBack_2_RT': mean_2
        })

    except Exception as e:
        print(f"Error reading {csv_file.name}: {e}")

# Build Final Table
nback_df = pd.DataFrame(all_nback_results).sort_values(['SubjectID', 'Time'])

print(f"\nFinal Analysis Complete. Rows: {len(nback_df)}")
print(nback_df)


# N-BACK D PRIME ANALYSIS
# adapted from a script shared by another researcher

import pandas as pd
import scipy.stats as stats # Needed for the Z-score (norm.ppf)
import ast
from pathlib import Path

# --- CONFIGURATION ---
clean_dir = nback_root / "Clean_Nback_Data"

# --- HELPER FUNCTION: Logic to calculate d' ---
def calculate_dprime(hits, misses, false_alarms, total_targets, total_non_targets):
    # Hautus correction: Add 0.5 to avoid infinity with 100% or 0% rates
    HR_adj = (hits + 0.5) / (total_targets + 1) if total_targets > 0 else 0.5
    FAR_adj = (false_alarms + 0.5) / (total_non_targets + 1) if total_non_targets > 0 else 0.5

    # Calculate d' using the Inverse Normal Distribution (Z-score)
    d_prime = stats.norm.ppf(HR_adj) - stats.norm.ppf(FAR_adj)
    return d_prime

all_dprime_results = []

print(f"--- Calculating d' (Accuracy) in {clean_dir.name} ---")

for csv_file in clean_dir.glob("*_nback.csv"):

    # 1. Parse Filename
    parts = csv_file.name.split('_')
    subj_id = parts[0]
    tp = parts[1]

    try:
        df = pd.read_csv(csv_file, delimiter='\t')

        # --- Initialize Counters ---
        # We use a dictionary to keep track of counts for 0-back and 2-back
        stats_count = {
            '0back': {'hits': 0, 'misses': 0, 'fa': 0, 'targets': 0, 'non_targets': 0},
            '2back': {'hits': 0, 'misses': 0, 'fa': 0, 'targets': 0, 'non_targets': 0}
        }

        # --- ROW-BY-ROW Processing (Safest for N-back logic) ---
        for _, row in df.iterrows():

            # === 0-BACK LOOP ===
            if 'buttonBox_0back.buttons' in df.columns and 'target_0back' in df.columns:
                btn = row['buttonBox_0back.buttons']
                target = row['target_0back']

                # Check if button was pressed (handling the string "['space']")
                has_press = False
                if isinstance(btn, str) and btn.strip() != 'None' and btn.strip() != '[]':
                     has_press = True

                # Logic: Hit vs False Alarm
                if target == 1:
                    stats_count['0back']['targets'] += 1
                    if has_press: stats_count['0back']['hits'] += 1
                    else:         stats_count['0back']['misses'] += 1
                else: # target == 0
                    stats_count['0back']['non_targets'] += 1
                    if has_press: stats_count['0back']['fa'] += 1

            # === 2-BACK LOOP ===
            if 'buttonBox_2back.buttons' in df.columns and 'target_2back' in df.columns:
                btn = row['buttonBox_2back.buttons']
                target = row['target_2back']

                has_press = False
                if isinstance(btn, str) and btn.strip() != 'None' and btn.strip() != '[]':
                     has_press = True

                if target == 1:
                    stats_count['2back']['targets'] += 1
                    if has_press: stats_count['2back']['hits'] += 1
                    else:         stats_count['2back']['misses'] += 1
                else:
                    stats_count['2back']['non_targets'] += 1
                    if has_press: stats_count['2back']['fa'] += 1

        # --- Final Calculation ---
        d_0 = calculate_dprime(
            stats_count['0back']['hits'], 
            stats_count['0back']['misses'], 
            stats_count['0back']['fa'], 
            stats_count['0back']['targets'], 
            stats_count['0back']['non_targets']
        )

        d_2 = calculate_dprime(
            stats_count['2back']['hits'], 
            stats_count['2back']['misses'], 
            stats_count['2back']['fa'], 
            stats_count['2back']['targets'], 
            stats_count['2back']['non_targets']
        )

        all_dprime_results.append({
            'SubjectID': subj_id,
            'Time': tp,
            'dprime_0': d_0,
            'dprime_2': d_2
        })

    except Exception as e:
        print(f" Error in {csv_file.name}: {e}")

# Create DataFrame
nback_acc_df = pd.DataFrame(all_dprime_results).sort_values(['SubjectID', 'Time'])

print(f"\nProcessed {len(nback_acc_df)} sessions.")
print(nback_acc_df)


# Merge Reaction Time (nback_df) and Accuracy (nback_acc_df)
nback_master_df = pd.merge(nback_df, nback_acc_df, on=['SubjectID', 'Time'], how='inner') # keep only intersection

# Organize the columns logically
nback_master_df = nback_master_df[[
    'SubjectID', 'Time', 
    'NBack_0_RT', 'dprime_0', 
    'NBack_2_RT', 'dprime_2'
]]

# Sort specifically to make it easy to read (Subject 1 T1, then T2...)
nback_master_df = nback_master_df.sort_values(['SubjectID', 'Time']).reset_index(drop=True)

print("Final N-Back Columns:", nback_master_df.columns.tolist())
print(f"Total Rows: {len(nback_master_df)}")
print(nback_master_df)

# SAVE TO CSV
# Define the output path
output_file = Path("analysis") / "nback_results.csv"

# Save the dataframe
# index=False prevents pandas from adding an annoying extra column of row numbers (0, 1, 2...)
nback_master_df.to_csv(output_file, index=False)

print(f"Success! Saved results to: {output_file}")