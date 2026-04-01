"""
Spider evaluation: LLM SQL generation only (no SQLValidatorAgent).

Run dev split → CSV:
    python spider_eval_generate_only.py --spider-root spider --dataset dev.json --output spider_dev_predictions.csv --model qwen2.5:7b

Run train split → CSV:
    python spider_eval_generate_only.py --spider-root spider --dataset train_spider.json --output spider_train_predictions.csv --model qwen2.5:7b

Uses only LLM_model + SchemaRetriever. Predicted SQL is the raw model output.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from LLM_model import LLM_model
from SchemaRetriever import SchemaRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spider Text-to-SQL: generate predictions only (no validation agent)."
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
        default="dev.json",
        help="Dataset JSON (e.g. dev.json, train_spider.json) or absolute path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="spider_predictions.csv",
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
        default=0,
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
        default=3,
        help="Print this many sample rows (ok) at end; 0 to disable.",
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
    df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


def print_results_report(output_path: Path, preview_n: int, just_processed: int) -> None:
    if not output_path.exists():
        print("Output file was not created.")
        return
    df = pd.read_csv(output_path)
    n = len(df)
    n_err = int((df["status"] == "error").sum()) if "status" in df.columns else 0
    n_ok = n - n_err
    avg_t = float(df["elapsed_sec"].mean()) if "elapsed_sec" in df.columns and n else 0.0

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (generate-only, no validation)")
    print("=" * 60)
    print(f"CSV path:     {output_path}")
    print(f"Rows in CSV:  {n}")
    print(f"  status=ok:   {n_ok}")
    print(f"  status=err:  {n_err}")
    if avg_t > 0:
        print(f"Avg sec/row:  {avg_t:.3f} (over all rows in file)")
    if just_processed > 0:
        print(f"Rows in this run (before resume skips): {just_processed}")
    print("=" * 60)

    if preview_n <= 0 or n_ok == 0:
        return
    ok_df = df[df["status"] == "ok"].head(preview_n)
    print(f"\nSample predictions (first {len(ok_df)} ok rows):\n")
    for _, r in ok_df.iterrows():
        q = str(r.get("question", ""))[:120]
        pred = str(r.get("predicted_sql", ""))[:200]
        print(f"  [{r.get('index')}] {r.get('db_id')}")
        print(f"    Q: {q}{'...' if len(str(r.get('question',''))) > 120 else ''}")
        print(f"    SQL: {pred}{'...' if len(str(r.get('predicted_sql',''))) > 200 else ''}\n")


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
    print("(No SQLValidatorAgent — predictions are direct from LLM.)\n")

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
            "status": "ok",
            "error": "",
            "elapsed_sec": 0.0,
        }
        t0 = time.time()
        try:
            if not db_id:
                raise ValueError("Missing db_id")
            if not question:
                raise ValueError("Missing question")
            schema_path = resolve_schema_path(spider_root, db_id)
            if schema_path is None:
                raise FileNotFoundError(f"No schema SQL for db_id={db_id!r}")

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
        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)
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

    print_results_report(output_path, args.preview, run_count)


if __name__ == "__main__":
    main()
