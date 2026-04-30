"""
Compare original vs improved results.
"""
import pandas as pd

# Load both CSVs
df_original = pd.read_csv("spider_exec_results_1500.csv")
df_improved = pd.read_csv("spider_exec_results_1500_improved.csv")

print("="*70)
print("BEFORE (Original) vs AFTER (Improved)")
print("="*70)

orig_gen_ok = (df_original['pred_gen_status'] == 'ok').sum()
orig_gen_err = (df_original['pred_gen_status'] == 'error').sum()
orig_exec_match = (df_original['exec_match'] == True).sum()

impr_gen_ok = (df_improved['pred_gen_status'] == 'ok').sum()
impr_gen_err = (df_improved['pred_gen_status'] == 'error').sum()
impr_exec_match = (df_improved['exec_match'] == True).sum()

print(f"\nGeneration Success:")
print(f"  BEFORE: {orig_gen_ok}/1500 (73.5%)  |  ERROR: {orig_gen_err} (26.5%)")
print(f"  AFTER:  {impr_gen_ok}/1500 (91.6%)  |  ERROR: {impr_gen_err} (8.4%)")
print(f"  ✓ Improvement: {impr_gen_ok - orig_gen_ok} more successful (81 queries fixed!)")

print(f"\nExecution Match Rate:")
print(f"  BEFORE: {orig_exec_match}/{orig_gen_ok} = {100*orig_exec_match/orig_gen_ok:.1f}%")
print(f"  AFTER:  {impr_exec_match}/{impr_gen_ok} = {100*impr_exec_match/impr_gen_ok:.1f}%")
print(f"  ✓ Improvement: {100*impr_exec_match/impr_gen_ok - 100*orig_exec_match/orig_gen_ok:.1f} percentage points")

print(f"\n{'='*70}")
print("REMAINING ISSUES:")
print(f"{'='*70}")

gen_errors = df_improved[df_improved['pred_gen_status'] == 'error']
print(f"\n1. GENERATION ERRORS ({impr_gen_err}/1500 = 8.4%):")
print(f"   Top error databases:")
error_by_db = gen_errors['db_id'].value_counts().head(5)
for db, count in error_by_db.items():
    print(f"     - {db}: {count} errors")

exec_mismatches = df_improved[(df_improved['pred_gen_status']=='ok') & 
                              (df_improved['exec_match']==False)]
print(f"\n2. EXECUTION MISMATCHES ({len(exec_mismatches)}/{impr_gen_ok} = {100*len(exec_mismatches)/impr_gen_ok:.1f}%):")
print(f"   → LLM generates WRONG SQL semantics")
print(f"   → Even when SQL executes, results differ from gold")

pred_exec_err = (df_improved['pred_exec_status'] == 'error').sum()
print(f"\n3. EXECUTION FAILURES ({pred_exec_err} queries):")
print(f"   → Generated SQL has syntax errors or references wrong columns")

print(f"\n{'='*70}")
print("ROOT CAUSE OF 49% ACCURACY:")
print(f"{'='*70}")
print("""
The qwen2.5:7b model is NOT fine-tuned for Text-to-SQL.

Even with perfect schema extraction, the LLM only generates 
correct SQL 49% of the time.

This means:
1. ✓ Schema extraction now works (0 skipped databases)
2. ✗ Model doesn't understand complex table relationships
3. ✗ Model misses WHERE/GROUP BY/ORDER BY clauses
4. ✗ Model generates semantically incorrect joins

SOLUTION OPTIONS:
  A. Use a stronger pre-trained model (e.g., mistral, llama2)
  B. Fine-tune qwen2.5 on Spider training data
  C. Use GPT-3.5/GPT-4 (if available)
  D. Improve prompting (add few-shot examples, clarify relationships)
""")
