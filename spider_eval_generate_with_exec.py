"""
Spider evaluation with SQL generation AND execution validation.

Run dev split → CSV with execution results:
    python spider_eval_generate_with_exec.py --spider-root spider --dataset dev.json --output spider_dev_exec_results.csv --model qwen2.5:7b --limit 1500

Run train split → CSV with execution results:
    python spider_eval_generate_with_exec.py --spider-root spider --dataset train_spider.json --output spider_train_exec_results.csv --model qwen2.5:7b --limit 1500

Generates SQL, then executes both predicted_sql and gold_sql on the actual Spider SQLite database.
Compares results and reports execution accuracy.
"""

import argparse
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

import pandas as pd

from LLM_model import LLM_model
from SchemaRetriever import SchemaRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spider Text-to-SQL: generate + execute predictions with accuracy validation."
    )
    parser.add_argument(
        "--spider-root",
        type=str,
        default="spider",
        help="Path to Spider root (contains dev.json, train_spider.json, database/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train_spider.json",
        help="Dataset JSON (e.g. dev.json, train_spider.json) or absolute path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="spider_exec_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip row indices already present in output CSV; append new rows.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Write CSV every N completed rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Max rows (0 = all in selected range).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Print this many sample rows at end; 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_dataset_path(spider_root: Path, dataset_arg: str) -> Path:
    candidate = Path(dataset_arg)
    if candidate.exists():
        return candidate
    return spider_root / dataset_arg


def resolve_schema_path(spider_root: Path, db_id: str) -> Optional[Path]:
    db_dir = spider_root / "database" / db_id
    if not db_dir.exists():
        return None
    for name in (db_dir / "schema.sql", db_dir / f"{db_id}.sql"):
        if name.exists():
            return name
    sql_files = sorted(db_dir.glob("*.sql"))
    return sql_files[0] if sql_files else None


def resolve_db_path(spider_root: Path, db_id: str) -> Optional[Path]:
    """Resolve path to the SQLite database file for a given db_id."""
    db_dir = spider_root / "database" / db_id
    if not db_dir.exists():
        return None
    db_file = db_dir / f"{db_id}.sqlite"
    if db_file.exists():
        return db_file
    # Fallback: look for any .sqlite file
    sqlite_files = sorted(db_dir.glob("*.sqlite"))
    return sqlite_files[0] if sqlite_files else None


def extract_schema_from_sqlite(db_path: Path) -> str:
    """
    Extract schema (CREATE TABLE statements) directly from SQLite database.
    Returns SQL schema string.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all table creation statements from sqlite_master
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        tables = cursor.fetchall()
        conn.close()
        
        schema_sql = "\n".join([table[0] for table in tables if table[0]])
        return schema_sql if schema_sql else ""
    except Exception as e:
        return ""


def normalize_result(result: Any) -> str:
    """Normalize query result for comparison (case-insensitive column names, sorted order)."""
    if isinstance(result, list) and result:
        # For list of dicts, normalize each dict's keys to lowercase and sort
        normalized_rows = []
        for row in result:
            if isinstance(row, dict):
                # Convert all keys to lowercase
                normalized_row = {k.lower(): v for k, v in row.items()}
                normalized_rows.append(normalized_row)
            else:
                normalized_rows.append(row)
        # Sort by JSON representation for consistent comparison
        return str(sorted([str(sorted(row.items()) if isinstance(row, dict) else row) for row in normalized_rows]))
    elif isinstance(result, list):
        return str(sorted([str(r) for r in result]))
    elif isinstance(result, dict):
        # Normalize dict keys to lowercase
        normalized = {k.lower(): v for k, v in result.items()}
        return str(sorted(normalized.items()))
    else:
        return str(result)


def execute_sql_safe(db_path: Path, sql_query: str) -> Tuple[bool, str, str]:
    """
    Execute SQL query safely on a SQLite database.
    Returns: (success: bool, error_msg: str, result_str: str)
    """
    try:
        if not sql_query or not sql_query.strip():
            return False, "Empty SQL query", ""

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(sql_query)
            if sql_query.strip().lower().startswith("select"):
                rows = cursor.fetchall()
                result_list = [dict(row) for row in rows]
                result_str = normalize_result(result_list)
                conn.close()
                return True, "", result_str
            else:
                conn.commit()
                result_str = f"{cursor.rowcount} rows affected"
                conn.close()
                return True, "", result_str
        except sqlite3.OperationalError as e:
            conn.close()
            error_msg = str(e)
            # Truncate very long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return False, error_msg, ""
        except Exception as e:
            conn.close()
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return False, error_msg, ""

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return False, error_msg, ""


def load_completed_indices(output_path: Path) -> Set[int]:
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["index"])
        return set(int(x) for x in df["index"].dropna().tolist())
    except Exception:
        return set()


def append_rows_to_csv(output_path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    # Increase display max column width for pandas
    with pd.option_context('max_colwidth', None):
        df.to_csv(
            output_path,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8",
        )


def print_results_report(output_path: Path, preview_n: int, just_processed: int, skipped: int = 0) -> None:
    if not output_path.exists():
        print("Output file was not created.")
        return

    df = pd.read_csv(output_path)
    n = len(df)
    n_err_gen = int((df["pred_gen_status"] == "error").sum()) if "pred_gen_status" in df.columns else 0
    n_ok_gen = n - n_err_gen

    n_exec_match = int((df["exec_match"] == True).sum()) if "exec_match" in df.columns else 0
    n_exec_mismatch = int((df["exec_match"] == False).sum()) if "exec_match" in df.columns else 0

    avg_t = float(df["elapsed_sec"].mean()) if "elapsed_sec" in df.columns and n else 0.0

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (with execution validation)")
    print("=" * 80)
    print(f"CSV path:              {output_path}")
    print(f"Skipped (no schema):   {skipped}")
    print(f"Total rows in CSV:     {n}")
    print(f"  Generation OK:       {n_ok_gen}")
    print(f"  Generation ERROR:    {n_err_gen}")
    if n_ok_gen > 0:
        print(f"Execution Validation (on generated queries):")
        print(f"  Exec Match:         {n_exec_match} ({100.0*n_exec_match/n_ok_gen:.1f}%)" if n_ok_gen > 0 else "  Exec Match:         0")
        print(f"  Exec Mismatch:      {n_exec_mismatch}")
    if avg_t > 0:
        print(f"Avg sec/row:           {avg_t:.3f}")
    if just_processed > 0:
        print(f"Rows processed (new):  {just_processed}")
    print("=" * 80)

    if preview_n <= 0 or n_ok_gen == 0:
        return

    ok_df = df[df["pred_gen_status"] == "ok"].head(preview_n)
    print(f"\nSample predictions with execution results (first {len(ok_df)} rows):\n")
    for idx, r in ok_df.iterrows():
        q = str(r.get("question", ""))[:100]
        pred = str(r.get("predicted_sql", ""))[:120]
        pred_exec = str(r.get("pred_exec_status", ""))
        exec_match_val = r.get("exec_match", False)

        print(f"[{r.get('index')}] {r.get('db_id')}")
        print(f"  Q: {q}{'...' if len(str(r.get('question',''))) > 100 else ''}")
        print(f"  SQL: {pred}{'...' if len(str(r.get('predicted_sql',''))) > 120 else ''}")
        print(f"  Pred Exec: {pred_exec} | Match: {exec_match_val}")
        if pred_exec == "ok" and not exec_match_val:
            gold_result = str(r.get("gold_exec_result", ""))[:150]
            pred_result = str(r.get("pred_exec_result", ""))[:150]
            print(f"    Gold result: {gold_result}{'...' if len(str(r.get('gold_exec_result',''))) > 150 else ''}")
            print(f"    Pred result: {pred_result}{'...' if len(str(r.get('pred_exec_result',''))) > 150 else ''}")
        print()


def main() -> None:
    args = parse_args()
    spider_root = Path(args.spider_root).resolve()
    dataset_path = resolve_dataset_path(spider_root, args.dataset).resolve()
    output_path = Path(args.output).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {dataset_path}")

    total_rows = len(data)
    start_index = max(0, args.start_index)
    end_index = (
        total_rows if args.limit <= 0 else min(total_rows, start_index + args.limit)
    )
    slice_rows = data[start_index:end_index]
    if not slice_rows:
        raise ValueError("No rows selected.")

    print(f"Dataset: {dataset_path}")
    print(f"Rows: {start_index}..{end_index - 1} (count {len(slice_rows)})")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print("(With SQL execution validation on Spider SQLite databases.)\n")

    llm_cache: Dict[str, LLM_model] = {}
    buffer: List[dict] = []
    completed = load_completed_indices(output_path) if args.resume else set()
    if completed:
        print(f"Resume: skipping {len(completed)} indices already in CSV.\n")

    run_count = 0
    for offset, sample in enumerate(slice_rows):
        idx = start_index + offset
        if args.resume and idx in completed:
            continue

        db_id = str(sample.get("db_id", "")).strip()
        question = str(sample.get("question", "")).strip()
        gold_sql = str(sample.get("query", "")).strip()

        row = {
            "index": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": "",
            "summary": "",
            "pred_gen_status": "ok",
            "pred_gen_error": "",
            "pred_exec_status": "",
            "pred_exec_error": "",
            "pred_exec_result": "",
            "gold_exec_status": "",
            "gold_exec_error": "",
            "gold_exec_result": "",
            "exec_match": False,
            "elapsed_sec": 0.0,
        }
        t0 = time.time()
        try:
            if not db_id:
                raise ValueError("Missing db_id")
            if not question:
                raise ValueError("Missing question")

            # ===== STEP 1: Generate SQL with LLM =====
            schema_path = resolve_schema_path(spider_root, db_id)
            
            # If schema.sql doesn't exist, try to extract from SQLite database
            if schema_path is None:
                db_path_temp = resolve_db_path(spider_root, db_id)
                if db_path_temp is None:
                    raise FileNotFoundError(f"No schema SQL or SQLite DB for db_id={db_id!r}")
                # Extract schema from SQLite
                schema_sql = extract_schema_from_sqlite(db_path_temp)
                if not schema_sql:
                    raise FileNotFoundError(f"Cannot extract schema from SQLite for db_id={db_id!r}")
                # Create temporary schema file in memory (will be handled by SchemaRetriever)
                # For now, we'll create a temp file
                temp_schema_path = Path(f"./temp_schema_{db_id}.sql")
                temp_schema_path.write_text(schema_sql)
                schema_path = temp_schema_path
            
            if db_id not in llm_cache:
                retriever = SchemaRetriever(str(schema_path))
                retriever.collection_name = db_id
                retriever.collection = retriever.client.get_or_create_collection(
                    name=db_id,
                    embedding_function=retriever.embed_model,
                )
                retriever.store_schema()
                llm_cache[db_id] = LLM_model(
                    collection_name=db_id, ollama_model=args.model
                )

            out = llm_cache[db_id].generate_sql(question)
            row["predicted_sql"] = str(out.get("sql_query", "")).strip()
            row["summary"] = str(out.get("summary", "")).strip()

            # ===== STEP 2: Execute predicted and gold SQL on Spider DB =====
            db_path = resolve_db_path(spider_root, db_id)
            if db_path is None:
                raise FileNotFoundError(f"No SQLite DB found for db_id={db_id!r}")

            # Execute predicted SQL
            if row["predicted_sql"]:
                pred_ok, pred_err, pred_result = execute_sql_safe(db_path, row["predicted_sql"])
                row["pred_exec_status"] = "ok" if pred_ok else "error"
                row["pred_exec_error"] = pred_err if pred_err else ""
                row["pred_exec_result"] = pred_result if pred_ok else ""
            else:
                row["pred_exec_status"] = "error"
                row["pred_exec_error"] = "Empty predicted SQL"

            # Execute gold SQL
            if gold_sql:
                gold_ok, gold_err, gold_result = execute_sql_safe(db_path, gold_sql)
                row["gold_exec_status"] = "ok" if gold_ok else "error"
                row["gold_exec_error"] = gold_err if gold_err else ""
                row["gold_exec_result"] = gold_result if gold_ok else ""
            else:
                row["gold_exec_status"] = "error"
                row["gold_exec_error"] = "Empty gold SQL"

            # Compare results
            if (
                row["pred_exec_status"] == "ok"
                and row["gold_exec_status"] == "ok"
                and row["pred_exec_result"] == row["gold_exec_result"]
            ):
                row["exec_match"] = True
            else:
                row["exec_match"] = False

        except Exception as e:
            row["pred_gen_status"] = "error"
            row["pred_gen_error"] = str(e)
        finally:
            row["elapsed_sec"] = round(time.time() - t0, 3)
            buffer.append(row)
            run_count += 1

        if args.flush_every > 0 and len(buffer) >= args.flush_every:
            append_rows_to_csv(output_path, buffer)
            buffer.clear()

        if (offset + 1) % 10 == 0 or offset == len(slice_rows) - 1:
            print(f"Processed {offset + 1}/{len(slice_rows)}")

    append_rows_to_csv(output_path, buffer)
    
    # Cleanup temporary schema files
    import glob
    for temp_file in glob.glob("./temp_schema_*.sql"):
        try:
            Path(temp_file).unlink()
        except:
            pass
    
    print_results_report(output_path, args.preview, run_count)


if __name__ == "__main__":
    main()
