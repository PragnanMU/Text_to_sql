import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Set, List

import pandas as pd

from LLM_model import LLM_model
from SchemaRetriever import SchemaRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Text-to-SQL generation on Spider and export results to CSV."
    )
    parser.add_argument(
        "--spider-root",
        type=str,
        default="spider",
        help="Path to Spider root folder (contains dev.json/train_spider.json and database/).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dev.json",
        help="Spider dataset file name or absolute path (e.g., dev.json, train_spider.json).",
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
        help="If output CSV exists, skip rows whose `index` is already present and append new results.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Append to CSV every N processed rows (useful for long runs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum rows to process (0 means all).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start row index in dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name.",
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

    preferred_files = [db_dir / "schema.sql", db_dir / f"{db_id}.sql"]
    for file_path in preferred_files:
        if file_path.exists():
            return file_path

    sql_files = sorted(db_dir.glob("*.sql"))
    if sql_files:
        return sql_files[0]
    return None


def load_completed_indices(output_path: Path) -> Set[int]:
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["index"])
        indices = set(int(x) for x in df["index"].dropna().tolist())
        return indices
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
        raise ValueError(
            f"Unsupported dataset format in {dataset_path}. Expected a list of examples."
        )

    total_rows = len(data)
    start_index = max(0, args.start_index)
    end_index = total_rows if args.limit <= 0 else min(total_rows, start_index + args.limit)
    rows = data[start_index:end_index]

    if not rows:
        raise ValueError("No rows selected. Check --start-index and --limit.")

    print(f"Dataset: {dataset_path}")
    print(f"Selected rows: {start_index} to {end_index - 1} (count: {len(rows)})")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")

    llm_cache: Dict[str, LLM_model] = {}
    buffer = []

    completed_indices: Set[int] = set()
    if args.resume:
        completed_indices = load_completed_indices(output_path)
        if completed_indices:
            print(f"Resume enabled. Found {len(completed_indices)} completed rows in CSV.")

    for offset, sample in enumerate(rows):
        idx = start_index + offset
        if args.resume and idx in completed_indices:
            continue

        db_id = str(sample.get("db_id", "")).strip()
        question = str(sample.get("question", "")).strip()
        gold_sql = str(sample.get("query", "")).strip()

        row_result = {
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
                raise ValueError("Missing db_id in sample.")
            if not question:
                raise ValueError("Missing question in sample.")

            schema_path = resolve_schema_path(spider_root, db_id)
            if schema_path is None:
                raise FileNotFoundError(f"Schema SQL not found for db_id='{db_id}'")

            if db_id not in llm_cache:
                retriever = SchemaRetriever(str(schema_path))
                retriever.collection_name = db_id
                retriever.collection = retriever.client.get_or_create_collection(
                    name=db_id,
                    embedding_function=retriever.embed_model,
                )
                retriever.store_schema()
                llm_cache[db_id] = LLM_model(collection_name=db_id, ollama_model=args.model)

            generated = llm_cache[db_id].generate_sql(question)
            row_result["predicted_sql"] = str(generated.get("sql_query", "")).strip()
            row_result["summary"] = str(generated.get("summary", "")).strip()
        except Exception as e:
            row_result["status"] = "error"
            row_result["error"] = str(e)
        finally:
            row_result["elapsed_sec"] = round(time.time() - t0, 3)
            buffer.append(row_result)

        if args.flush_every > 0 and len(buffer) >= args.flush_every:
            append_rows_to_csv(output_path, buffer)
            buffer.clear()

        if (offset + 1) % 10 == 0 or offset == len(rows) - 1:
            print(f"Processed {offset + 1}/{len(rows)}")

    append_rows_to_csv(output_path, buffer)

    try:
        df_all = pd.read_csv(output_path)
        print(f"Saved results to: {output_path}")
        print(f"Total rows in CSV: {len(df_all)}, Errors: {(df_all['status'] == 'error').sum()}")
    except Exception:
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
