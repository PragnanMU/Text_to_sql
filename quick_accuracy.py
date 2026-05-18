import pandas as pd
import json

df = pd.read_csv("spider_exec_results_1500_fewshot.csv")

# Only rows where both executed
valid = df[(df['pred_exec_status'] == 'ok') & (df['gold_exec_status'] == 'ok')].copy()

print("="*70)
print("VALUE-BASED ACCURACY RECALCULATION")
print("="*70)
print(f"\nTotal rows in CSV: {len(df)}")
print(f"Both executed OK: {len(valid)}")

# Simple value extraction
matches = 0
for idx, row in valid.iterrows():
    pred = str(row['pred_exec_result']).lower()
    gold = str(row['gold_exec_result']).lower()
    
    # Extract numeric values only (ignore column names)
    import re
    pred_nums = sorted(re.findall(r"[-+]?\d*\.?\d+", pred))
    gold_nums = sorted(re.findall(r"[-+]?\d*\.?\d+", gold))
    
    # Also check for string values
    pred_strs = sorted(re.findall(r"'([^']*)'", pred))
    gold_strs = sorted(re.findall(r"'([^']*)'", gold))
    
    if pred_nums == gold_nums and pred_strs == gold_strs:
        matches += 1

accuracy = 100 * matches / len(valid) if len(valid) > 0 else 0

print(f"\nMatches (by values): {matches}/{len(valid)}")
print(f"Accuracy: {accuracy:.1f}%")

# Old accuracy
old_matches = (df['exec_match'] == True).sum()
old_accuracy = 100 * old_matches / len(valid) if len(valid) > 0 else 0

print(f"\nOld accuracy (column-based): {old_accuracy:.1f}%")
print(f"New accuracy (value-based):  {accuracy:.1f}%")
print(f"Improvement: +{accuracy - old_accuracy:.1f} pp")
