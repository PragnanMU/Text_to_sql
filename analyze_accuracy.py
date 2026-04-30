"""
Analyze execution results to identify accuracy issues.
"""
import pandas as pd
import json
from collections import defaultdict

# Load results
df = pd.read_csv('spider_exec_results_1500.csv')

print("\n" + "="*80)
print("EXECUTION ACCURACY ANALYSIS")
print("="*80)

# Overall stats
total = len(df)
gen_ok = (df['pred_gen_status'] == 'ok').sum()
gen_err = (df['pred_gen_status'] == 'error').sum()
exec_ok = (df['pred_exec_status'] == 'ok').sum()
exec_match = (df['exec_match'] == True).sum()
exec_mismatch = (df['exec_match'] == False).sum()

print(f"\nOVERALL METRICS:")
print(f"  Total queries:           {total}")
print(f"  Generation Success:      {gen_ok} ({100*gen_ok/total:.1f}%)")
print(f"  Generation Errors:       {gen_err} ({100*gen_err/total:.1f}%)")
print(f"  Execution Success:       {exec_ok} ({100*exec_ok/gen_ok:.1f}% of generated)")
print(f"  Result Match:            {exec_match} ({100*exec_match/gen_ok:.1f}% of generated)")
print(f"  Result Mismatch:         {exec_mismatch} ({100*exec_mismatch/gen_ok:.1f}% of generated)")

# Error breakdown by DB
print(f"\nGENERATION ERRORS BY DATABASE:")
gen_error_by_db = df[df['pred_gen_status'] == 'error'].groupby('db_id').size().sort_values(ascending=False)
for db, count in gen_error_by_db.head(10).items():
    print(f"  {db:30s} {count:3d} errors")

# Mismatch reasons
print(f"\nEXECUTION MISMATCH ANALYSIS (on successfully generated queries):")
success_df = df[df['pred_gen_status'] == 'ok']
pred_exec_err = (success_df['pred_exec_status'] == 'error').sum()
gold_exec_err = (success_df['gold_exec_status'] == 'error').sum()
both_ok_diff = ((success_df['pred_exec_status'] == 'ok') & 
                (success_df['gold_exec_status'] == 'ok') & 
                (success_df['exec_match'] == False)).sum()

print(f"  Predicted SQL execution errors: {pred_exec_err}")
print(f"  Gold SQL execution errors:      {gold_exec_err}")
print(f"  Both executed OK, results differ: {both_ok_diff}")

# Sample prediction errors
print(f"\nSAMPLE PREDICTED SQL ERRORS:")
pred_errors = df[(df['pred_gen_status']=='ok') & (df['pred_exec_status']=='error')][['predicted_sql','pred_exec_error']].head(3)
for idx, row in pred_errors.iterrows():
    sql = str(row['predicted_sql'])[:80]
    err = str(row['pred_exec_error'])[:80]
    print(f"  SQL: {sql}")
    print(f"  ERR: {err}\n")

# Sample mismatches (both executed, different results)
print(f"\nSAMPLE RESULT MISMATCHES (both executed, different output):")
mismatches = success_df[(success_df['pred_exec_status']=='ok') & 
                        (success_df['gold_exec_status']=='ok') & 
                        (success_df['exec_match']==False)][['db_id','predicted_sql','gold_sql','pred_exec_result','gold_exec_result']].head(2)
for idx, row in mismatches.iterrows():
    print(f"\n  DB: {row['db_id']}")
    print(f"  Predicted SQL: {str(row['predicted_sql'])[:100]}")
    print(f"  Gold SQL:      {str(row['gold_sql'])[:100]}")
    pred_res = str(row['pred_exec_result'])[:80]
    gold_res = str(row['gold_exec_result'])[:80]
    print(f"  Pred result: {pred_res}")
    print(f"  Gold result: {gold_res}")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print(f"""
1. GENERATION FAILURES: {gen_err} errors from 7 DBs with missing schemas
   - Skip these automatically to get {gen_ok} valid queries

2. LOW ACCURACY: {100*exec_match/gen_ok:.1f}% match rate indicates LLM is generating 
   semantically DIFFERENT SQL than gold standard
   
3. EXECUTION ERRORS: {pred_exec_err} of {gen_ok} predicted queries fail to execute
   - Usually "no such column" errors = schema mismatches in generated SQL
   
4. FIX NEEDED: Improve LLM prompt or fine-tune model to generate SQL 
   that matches gold standard more closely
""")
