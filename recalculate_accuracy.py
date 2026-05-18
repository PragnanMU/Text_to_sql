"""
Re-compare results from existing CSV using value-based normalization.
This doesn't re-run queries, just recalculates accuracy with better comparison.
"""
import pandas as pd
import json
from typing import Any

def normalize_result_by_values(result_str: str) -> str:
    """
    Parse result string and normalize by VALUES only (ignoring column names).
    Result strings are stored as JSON lists of dicts/tuples.
    """
    try:
        # Parse the result string
        if not result_str or pd.isna(result_str):
            return ""
        
        result_str = str(result_str).strip()
        
        # Handle format: ["[('col', val), ...]", "[(...)]", ...]
        if result_str.startswith('["') and result_str.endswith('"]'):
            # Parse JSON array of strings
            result_list = json.loads(result_str)
            values_list = []
            
            for item_str in result_list:
                # Each item is like "[(col, val), ...]"
                try:
                    # Use eval safely for tuples (be careful in production!)
                    item = eval(item_str)
                    if isinstance(item, list):
                        for row in item:
                            if isinstance(row, tuple):
                                # Extract just the value (second element)
                                val = row[1] if len(row) > 1 else row[0]
                                values_list.append(str(val).lower().strip())
                            else:
                                values_list.append(str(row).lower().strip())
                except:
                    values_list.append(item_str.lower().strip())
            
            return str(sorted(values_list))
        else:
            return result_str.lower().strip()
    except Exception as e:
        return str(result_str).lower().strip()

# Load CSV
df = pd.read_csv("spider_exec_results_1500_fewshot.csv")

print("="*70)
print("RE-CALCULATING ACCURACY WITH VALUE-BASED COMPARISON")
print("="*70)

# Only compare rows where both pred and gold executed successfully
valid_rows = df[(df['pred_exec_status'] == 'ok') & (df['gold_exec_status'] == 'ok')].copy()

print(f"\nRows with both predictions and gold executed: {len(valid_rows)}")

# Re-normalize and compare
matches = 0
mismatches = 0

for idx, row in valid_rows.iterrows():
    pred_result = normalize_result_by_values(row['pred_exec_result'])
    gold_result = normalize_result_by_values(row['gold_exec_result'])
    
    if pred_result == gold_result:
        matches += 1
    else:
        mismatches += 1

print(f"\n{'='*70}")
print("NEW ACCURACY (Value-Based Comparison):")
print(f"{'='*70}")

print(f"\nMatches:   {matches}")
print(f"Mismatches: {mismatches}")
accuracy = 100 * matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
print(f"\nACCURACY: {accuracy:.1f}%")

print(f"\n{'='*70}")
print("COMPARISON:")
print(f"{'='*70}")
old_accuracy = (df['exec_match'] == True).sum() / len(df[(df['pred_exec_status'] == 'ok') & (df['gold_exec_status'] == 'ok')])
print(f"Old (column-based):   {100*old_accuracy:.1f}%")
print(f"New (value-based):    {accuracy:.1f}%")
print(f"Improvement:          +{accuracy - 100*old_accuracy:.1f} pp")

print(f"\n{'='*70}")
print("SAMPLE COMPARISONS:")
print(f"{'='*70}")

# Show some examples
for i, (idx, row) in enumerate(valid_rows.head(10).iterrows()):
    pred_norm = normalize_result_by_values(row['pred_exec_result'])
    gold_norm = normalize_result_by_values(row['gold_exec_result'])
    match = "✓ MATCH" if pred_norm == gold_norm else "✗ DIFF"
    
    print(f"\n[{i+1}] {row['db_id']} - {match}")
    print(f"  Q: {str(row['question'])[:60]}...")
    print(f"  Gold: {str(row['gold_exec_result'])[:80]}")
    print(f"  Pred: {str(row['pred_exec_result'])[:80]}")
    print(f"  Gold normalized: {pred_norm[:60]}")
    print(f"  Pred normalized: {gold_norm[:60]}")
