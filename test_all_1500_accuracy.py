"""
Comprehensive accuracy test for all 1500 queries with detailed breakdown.
"""
import pandas as pd
import sqlite3
from pathlib import Path

def normalize_result(result):
    """Normalize by VALUES ONLY, ignoring column names."""
    def extract_values(obj):
        if isinstance(obj, dict):
            return sorted([extract_values(v) for v in obj.values()], key=str)
        elif isinstance(obj, (list, tuple)):
            return sorted([extract_values(item) for item in obj], key=str)
        else:
            return str(obj).lower().strip()
    
    try:
        if isinstance(result, list) and result:
            normalized_rows = []
            for row in result:
                if isinstance(row, dict):
                    values = sorted([str(v).lower().strip() if not isinstance(v, (dict, list)) else str(extract_values(v)) 
                                   for v in row.values()], key=str)
                    normalized_rows.append(tuple(values))
                elif isinstance(row, (list, tuple)):
                    normalized_rows.append(tuple(sorted([str(v).lower().strip() for v in row], key=str)))
                else:
                    normalized_rows.append((str(row).lower().strip(),))
            return str(sorted(normalized_rows))
        elif isinstance(result, dict):
            values = sorted([str(v).lower().strip() for v in result.values()], key=str)
            return str(tuple(values))
        elif isinstance(result, list):
            return str(sorted([str(r).lower().strip() for r in result]))
        else:
            return str(result).lower().strip()
    except Exception:
        return str(result)

def execute_sql(db_path, sql_query):
    """Execute SQL and return (success, result_normalized)."""
    try:
        if not sql_query or not sql_query.strip():
            return False, None
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        if sql_query.strip().lower().startswith("select"):
            rows = cursor.fetchall()
            result_list = [dict(row) for row in rows]
            result_str = normalize_result(result_list)
            conn.close()
            return True, result_str
        else:
            conn.commit()
            conn.close()
            return True, f"{cursor.rowcount} rows"
    except Exception as e:
        return False, None

# Load CSV
df = pd.read_csv('spider_exec_results_1500_improved_fewshot_06_05_2026.csv')
spider_root = Path('spider')

print("="*90)
print("COMPREHENSIVE ACCURACY TEST FOR ALL 1500 QUERIES")
print("="*90)

# Statistics
total = len(df)
gen_ok = (df['pred_gen_status'] == 'ok').sum()
gen_error = (df['pred_gen_status'] == 'error').sum()

matches = 0
mismatches = 0
pred_errors = 0
gold_errors = 0
db_not_found = 0

print(f"\nProcessing {total} queries...")
print(f"Generation successful: {gen_ok}")
print(f"Generation errors: {gen_error}\n")

for idx, row in df.iterrows():
    if (idx + 1) % 250 == 0:
        print(f"Progress: {idx + 1}/{total} | Matches: {matches} | Accuracy so far: {100*matches/(matches+mismatches+pred_errors) if (matches+mismatches+pred_errors) > 0 else 0:.1f}%")
    
    # Skip if generation failed
    if row['pred_gen_status'] != 'ok':
        continue
    
    db_id = str(row['db_id'])
    db_path = spider_root / 'database' / db_id / f'{db_id}.sqlite'
    
    if not db_path.exists():
        db_not_found += 1
        continue
    
    pred_sql = str(row['predicted_sql']).strip()
    gold_sql = str(row['gold_sql']).strip()
    
    if not pred_sql or not gold_sql:
        continue
    
    # Execute
    pred_ok, pred_result = execute_sql(db_path, pred_sql)
    gold_ok, gold_result = execute_sql(db_path, gold_sql)
    
    # Compare
    if not pred_ok:
        pred_errors += 1
    elif not gold_ok:
        gold_errors += 1
    elif pred_result == gold_result:
        matches += 1
    else:
        mismatches += 1

total_executable = matches + mismatches + pred_errors
all_executable = matches + mismatches + pred_errors + gold_errors

print(f"\n" + "="*90)
print("FINAL RESULTS - ALL 1500 QUERIES")
print("="*90)

print(f"\nGENERATION PHASE:")
print(f"  Total queries: {total}")
print(f"  Generation OK: {gen_ok} ({100*gen_ok/total:.1f}%)")
print(f"  Generation ERRORS: {gen_error} ({100*gen_error/total:.1f}%)")

print(f"\nEXECUTION PHASE (on {gen_ok} generated queries):")
print(f"  Database found: {gen_ok - db_not_found}")
print(f"  Database not found: {db_not_found}")

print(f"\nMATCH ANALYSIS (on {total_executable} executables):")
print(f"  ✓ MATCHES: {matches} ({100*matches/total_executable if total_executable > 0 else 0:.1f}%)")
print(f"  ✗ MISMATCHES: {mismatches} ({100*mismatches/total_executable if total_executable > 0 else 0:.1f}%)")
print(f"  ✗ PRED EXEC ERRORS: {pred_errors} ({100*pred_errors/total_executable if total_executable > 0 else 0:.1f}%)")
print(f"  ✗ GOLD EXEC ERRORS: {gold_errors} ({100*gold_errors/all_executable if all_executable > 0 else 0:.1f}%)")

print(f"\n{'='*90}")
print(f"ACCURACY METRICS:")
print(f"{'='*90}")

# Accuracy on executables only
if total_executable > 0:
    exec_accuracy = 100 * matches / total_executable
    print(f"Execution Accuracy (on executables): {exec_accuracy:.1f}% ({matches}/{total_executable})")

# Overall accuracy on all 1500
overall_accuracy = 100 * matches / total
print(f"Overall Accuracy (on all 1500):      {overall_accuracy:.1f}% ({matches}/{total})")

# Comparison
print(f"\n{'='*90}")
print(f"IMPROVEMENT SUMMARY:")
print(f"{'='*90}")
print(f"Baseline (39.5%):        592/1500")
print(f"Improved (this run):     {matches}/1500")
print(f"Improvement:             +{matches - 592} queries ({100*(matches-592)/1500:.1f}%)")

print(f"\n{'='*90}")
