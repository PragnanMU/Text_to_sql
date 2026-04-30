import pandas as pd
import json

df = pd.read_csv('spider_exec_results_1500.csv')

print(f'Total: {len(df)}')
print(f'Gen errors: {(df["pred_gen_status"]=="error").sum()}')
print(f'Exec match TRUE: {(df["exec_match"]==True).sum()}')
print(f'Exec match FALSE: {(df["exec_match"]==False).sum()}')

print(f'\nPred exec status breakdown:')
print(df['pred_exec_status'].value_counts())

print(f'\nGold exec status breakdown:')
print(df['gold_exec_status'].value_counts())

print(f'\n=== Sample Generation Errors ===')
gen_err = df[df['pred_gen_status']=='error'][['db_id','question','pred_gen_error']].head(3)
for idx, row in gen_err.iterrows():
    print(f"\nDB: {row['db_id']}")
    print(f"Q: {row['question'][:100]}")
    print(f"Error: {row['pred_gen_error'][:150]}")

print(f'\n=== Sample Execution Errors (Predicted SQL) ===')
pred_exec_err = df[(df['pred_exec_status']=='error') & (df['pred_gen_status']=='ok')][['db_id','predicted_sql','pred_exec_error']].head(3)
for idx, row in pred_exec_err.iterrows():
    print(f"\nDB: {row['db_id']}")
    print(f"SQL: {str(row['predicted_sql'])[:120]}")
    print(f"Error: {str(row['pred_exec_error'])[:150]}")

print(f'\n=== Sample Execution Errors (Gold SQL) ===')
gold_exec_err = df[(df['gold_exec_status']=='error')][['db_id','gold_sql','gold_exec_error']].head(3)
for idx, row in gold_exec_err.iterrows():
    print(f"\nDB: {row['db_id']}")
    print(f"SQL: {str(row['gold_sql'])[:120]}")
    print(f"Error: {str(row['gold_exec_error'])[:150]}")

print(f'\n=== Sample Mismatches (both executed ok but results differ) ===')
mismatches = df[(df['exec_match']==False) & (df['pred_exec_status']=='ok') & (df['gold_exec_status']=='ok')][['db_id','predicted_sql','gold_sql','pred_exec_result','gold_exec_result']].head(3)
for idx, row in mismatches.iterrows():
    print(f"\nDB: {row['db_id']}")
    print(f"Pred SQL: {str(row['predicted_sql'])[:100]}")
    print(f"Gold SQL: {str(row['gold_sql'])[:100]}")
    pred_res = str(row['pred_exec_result'])[:100]
    gold_res = str(row['gold_exec_result'])[:100]
    print(f"Pred result: {pred_res}")
    print(f"Gold result: {gold_res}")
