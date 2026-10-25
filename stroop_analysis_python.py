#!/usr/bin/env python
# coding: utf-8

# STROOP PREPROCESSING

from pathlib import Path
import pandas as pd

# Change to the real folder
data_root = Path('data/data_stroop')

# List only the first 3 subject folders it finds, sanity check
folders = [f.name for f in data_root.iterdir() if f.is_dir()]
print(f"Found {len(folders)} folders. First few: {folders[:3]}")


# MAIN ANALYSIS LOOP, both stroop effect and accuracy

all_results = []

# 1. Iterate over every item in the main stroop results folder 
for subj_folder in data_root.iterdir():
    if not subj_folder.is_dir(): continue # safety check, if it is not a folder skip it

    # Temporary subject specific list
    subj_data = []

    #  2. Search recursively for csv files, look into any possible folder
    for csv_file in subj_folder.rglob("*.csv"):
        fname = csv_file.name.upper() # convert all file names to uppercase

        # 3. Identifying the time points
        if "T1" in fname: tp = "T1"
        elif "T2" in fname: tp = "T2"
        else: 
            continue # I am only interested in pre and post treatment so I am skipping T3, change accordingly

        # 4. Loading data
        try:   
            df = pd.read_csv(csv_file)
            
            # Isolate actual experimental trials
            real_trials = df[df['condition'].isin(['congruent', 'incongruent'])]
            
            clean = real_trials[(real_trials['correct'] == 1)] # 5. Clean data, keep only correct trials
            medians = clean.groupby('condition')['rt'].median() # 6. Calculate RT Medians
        # I am using medians instead of means and am not filtering out any RT

        # 7. Stroop Effect and Accuracy
            if 'incongruent' in medians and 'congruent' in medians:
                subj_data.append({
                  'SubjectID': subj_folder.name,
                  'Time': tp,
                  'Stroop_Effect': medians['incongruent'] - medians['congruent'],
                  'Stroop_Accuracy': real_trials['correct'].mean() * 100
                 })
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")


    # ONLY add to the final list if we found both T1 and T2 for this subject
    if len(subj_data) == 2:
        all_results.extend(subj_data)
    else:
        print(f"Skipping {subj_folder.name}: Found {len(subj_data)} sessions, need 2.")

# Build Table
summary_df = pd.DataFrame(all_results)
summary_df = summary_df.sort_values(['SubjectID', 'Time'])

print(f"\nFinal count: {summary_df['SubjectID'].nunique()} subjects with complete data.")
print(summary_df)


# SAVE TO CSV
# Define the output path
output_file = Path("analysis") / "stroop_results.csv"

# Save the dataframe
# index=False prevents pandas from adding an annoying extra column of row numbers (0, 1, 2...)
summary_df.to_csv(output_file, index=False)

print(f"Success! Saved results to: {output_file}")