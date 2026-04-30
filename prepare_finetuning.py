"""
Prepare Spider dataset for fine-tuning qwen2.5:7b on Text-to-SQL task.

This script:
1. Loads Spider training data
2. Extracts schema from SQLite databases
3. Creates a fine-tuning dataset in JSONL format
4. Provides instructions for fine-tuning

Fine-tuning can be done with:
- Hugging Face: unsloth or transformers
- Ollama: Via model creation (GGUF format)
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict

def extract_schema_from_sqlite(db_path: Path) -> str:
    """Extract schema from SQLite database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        tables = cursor.fetchall()
        conn.close()
        schema_sql = "\n".join([table[0] for table in tables if table[0]])
        return schema_sql if schema_sql else ""
    except Exception as e:
        print(f"Error extracting schema: {e}")
        return ""

def create_finetuning_dataset(limit: int = 1000) -> None:
    """Create fine-tuning dataset from Spider training data."""
    
    print(f"Preparing fine-tuning dataset (limit={limit})...")
    
    spider_root = Path("spider")
    train_path = spider_root / "train_spider.json"
    
    if not train_path.exists():
        print(f"Error: {train_path} not found")
        return
    
    with open(train_path) as f:
        data = json.load(f)
    
    finetuning_data = []
    skipped = 0
    
    for idx, sample in enumerate(data[:limit]):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{limit}...")
        
        db_id = sample.get("db_id", "").strip()
        question = sample.get("question", "").strip()
        gold_sql = sample.get("query", "").strip()
        
        if not all([db_id, question, gold_sql]):
            skipped += 1
            continue
        
        # Extract schema from SQLite
        db_path = spider_root / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            skipped += 1
            continue
        
        schema_sql = extract_schema_from_sqlite(db_path)
        if not schema_sql:
            skipped += 1
            continue
        
        # Create instruction-response format for fine-tuning
        instruction = f"""Generate a SQL query for this question using the given schema.

SCHEMA:
{schema_sql}

QUESTION: {question}"""
        
        response = gold_sql
        
        finetuning_data.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "db_id": db_id
        })
    
    # Save as JSONL for fine-tuning
    output_path = Path("finetuning_data.jsonl")
    with open(output_path, "w") as f:
        for item in finetuning_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"\n✓ Fine-tuning dataset created:")
    print(f"  File: {output_path}")
    print(f"  Samples: {len(finetuning_data)}")
    print(f"  Skipped: {skipped}")
    
    # Print instructions
    print("\n" + "="*70)
    print("NEXT STEPS FOR FINE-TUNING:")
    print("="*70)
    print("""
Option A: Fine-tune using Hugging Face (Recommended)
=========================================================
Install: pip install -U transformers unsloth[colab-new] xformers bitsandbytes

Python code:
  from unsloth import FastLanguageModel
  from datasets import load_dataset
  from trl import SFTTrainer, TrainingArguments
  
  # Load model
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name="Qwen/Qwen2.5-7B",
      max_seq_length=2048,
      load_in_4bit=True,
  )
  
  # Load dataset
  dataset = load_dataset("json", data_files="finetuning_data.jsonl", split="train")
  
  # Fine-tune
  trainer = SFTTrainer(
      model=model,
      tokenizer=tokenizer,
      train_dataset=dataset,
      dataset_text_field="instruction",
      args=TrainingArguments(
          per_device_train_batch_size=4,
          gradient_accumulation_steps=4,
          warmup_steps=100,
          num_train_epochs=3,
          learning_rate=2e-4,
          output_dir="qwen-sql-finetuned",
          save_steps=100,
          save_strategy="steps",
          logging_steps=10,
      ),
      max_seq_length=2048,
      packing=False,
  )
  trainer.train()

Option B: Using Ollama with LoRA (Advanced)
=============================================
Create Modelfile:
  FROM qwen2.5:7b
  ADAPTER ./lora_adapters.gguf

Then:
  ollama create qwen-sql-finetuned -f Modelfile
  ollama run qwen-sql-finetuned "Your prompt here"

Option C: Using LangChain Chain-of-Thought (Simple, No Training)
================================================================
Enhance prompting with chain-of-thought examples:
- Show step-by-step reasoning
- Include schema analysis
- Add constraint checking

Already implemented in improved LLM_model.py!
Run: python spider_eval_generate_with_exec.py --model qwen2.5:7b --limit 100 --preview 5
""")

if __name__ == "__main__":
    create_finetuning_dataset(limit=1000)
