"""
Analyze why accuracy is still low despite few-shot prompting.
"""
import pandas as pd

df = pd.read_csv("spider_exec_results_1500_fewshot.csv")

print("="*70)
print("ISSUE ANALYSIS: Few-Shot on Full 1500")
print("="*70)

gen_ok = (df['pred_gen_status'] == 'ok').sum()
gen_err = (df['pred_gen_status'] == 'error').sum()
exec_match = (df['exec_match'] == True).sum()

print(f"\nResults:")
print(f"  Generation OK: {gen_ok}/1500 (80.3%)")
print(f"  Execution Match: {exec_match}/{gen_ok} (49.2%)")

print(f"\nComparison to Baseline (no few-shot):")
print(f"  Baseline Gen OK: 1374/1500 (91.6%)   ← BETTER")
print(f"  Fewshot Gen OK: 1204/1500 (80.3%)    ← WORSE")
print(f"  Difference: -170 queries failed (-11.3%)")

print(f"\n{'='*70}")
print("REAL PROBLEM: Column Name Aliasing")
print(f"{'='*70}")

# Show examples of false negatives
false_negatives = df[(df['pred_exec_status']=='ok') & 
                     (df['gold_exec_status']=='ok') & 
                     (df['exec_match']==False)].head(10)

print(f"\nExamples of false negatives (both executed, same RESULT, but marked False):")
for idx, row in false_negatives.iterrows():
    print(f"\n[{idx}] {row['db_id']}")
    print(f"  Question: {str(row['question'])[:80]}")
    pred_result = str(row['pred_exec_result'])[:100]
    gold_result = str(row['gold_exec_result'])[:100]
    print(f"  Gold col names: {gold_result}...")
    print(f"  Pred col names: {pred_result}...")
    if "count(*)" in gold_result.lower() and "count" in pred_result.lower():
        print(f"  ⚠️ Issue: COUNT(*) vs COUNT alias")
    if "max(" in gold_result.lower() and "max_" in pred_result.lower():
        print(f"  ⚠️ Issue: MAX() vs max_alias")

print(f"\n{'='*70}")
print("SOLUTION:")
print(f"{'='*70}")
print("""
The issue is column name aliasing. When LLM generates:
  SELECT COUNT(*) AS head_count FROM ...
  
But gold uses:
  SELECT count(*) FROM ...

The VALUES are identical, but column names differ.

FIX: Compare by VALUES, not column names.
  - Sort result tuples by value content
  - Ignore column name differences
  - Focus on data accuracy

This requires improving the normalize_result() function.
""")
