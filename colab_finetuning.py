"""
Google Colab Fine-Tuning Script for Qwen2.5-7B on Spider Text-to-SQL

Instructions:
1. Go to https://colab.research.google.com
2. Create new notebook
3. Copy-paste this entire script into a cell
4. Run the cell (takes ~1-2 hours on free T4 GPU)
5. Download the fine-tuned model from the output
6. Use it locally with the local_inference.py script

NOTE: This script is optimized for Google Colab environment
"""

# ============================================================================
# PART 1: Install dependencies (run this first)
# ============================================================================

print("Installing dependencies...")
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                      "transformers", "torch", "accelerate", "datasets", "bitsandbytes"])

print("✓ Dependencies installed")

# ============================================================================
# PART 2: Download dataset from your local machine
# ============================================================================

print("\nPreparing dataset...")

# Create finetuning_data.jsonl content
# This is the same 1000-sample dataset prepared earlier

import json
import gzip
from io import BytesIO

# You'll need to upload finetuning_data.jsonl to Colab
# Use the file upload widget in Colab:
#   from google.colab import files
#   uploaded = files.upload()

# For now, we'll create a smaller version for testing
# In production, upload your full finetuning_data.jsonl

SAMPLE_DATA = [
    {
        "instruction": "Generate a SQL query for this question using the given schema.\n\nSCHEMA:\nCREATE TABLE student (student_id int, student_name text, major_id int);\n\nQUESTION: How many students are there?",
        "input": "",
        "output": "SELECT COUNT(*) FROM student",
        "db_id": "student"
    },
    {
        "instruction": "Generate a SQL query for this question using the given schema.\n\nSCHEMA:\nCREATE TABLE student (student_id int, student_name text, major_id int); CREATE TABLE major (major_id int, major_name text);\n\nQUESTION: Find students in Computer Science major.",
        "input": "",
        "output": "SELECT student_name FROM student JOIN major ON student.major_id = major.major_id WHERE major.major_name = 'Computer Science'",
        "db_id": "student"
    }
]

# Save sample data
with open("finetuning_data_sample.jsonl", "w") as f:
    for item in SAMPLE_DATA:
        f.write(json.dumps(item) + "\n")

print("✓ Dataset ready (using sample data for quick test)")
print("\nTO USE YOUR FULL DATASET:")
print("  1. Run: from google.colab import files; files.upload()")
print("  2. Upload finetuning_data.jsonl")
print("  3. Change 'finetuning_data_sample.jsonl' to 'finetuning_data.jsonl' below")

# ============================================================================
# PART 3: Fine-tuning Code
# ============================================================================

print("\n" + "="*70)
print("STARTING FINE-TUNING")
print("="*70)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load model and tokenizer
print("\nLoading Qwen2.5-7B...")
model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✓ Model loaded: {model.num_parameters() / 1e9:.1f}B parameters")

# Load and prepare dataset
print("\nPreparing dataset...")
dataset = load_dataset("json", data_files="finetuning_data_sample.jsonl", split="train")

def format_prompt(example):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": prompt}

dataset = dataset.map(format_prompt, remove_columns=["instruction", "input", "output", "db_id"])

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"✓ Train samples: {len(dataset['train'])}")
print(f"✓ Val samples: {len(dataset['test'])}")

# Training arguments - adjusted for Colab
print("\nSetting up training...")
training_args = TrainingArguments(
    output_dir="/content/qwen-sql-finetuned",
    num_train_epochs=2,  # Reduced for demo
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    weight_decay=0.01,
    learning_rate=2e-4,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=torch.cuda.is_available(),
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# Train
print("Starting training... (this may take a while)")
trainer.train()

# Save model
print("\nSaving model...")
model.save_pretrained("/content/qwen-sql-finetuned")
tokenizer.save_pretrained("/content/qwen-sql-finetuned")

print("✓ Model saved to /content/qwen-sql-finetuned")

# ============================================================================
# PART 4: Download model
# ============================================================================

print("\n" + "="*70)
print("DOWNLOADING MODEL")
print("="*70)

import shutil

# Create zip file
print("\nCreating zip file...")
shutil.make_archive("qwen-sql-finetuned", "zip", "/content", "qwen-sql-finetuned")

print("✓ Model packaged: qwen-sql-finetuned.zip")

# Download
from google.colab import files
print("\nDownloading to your computer...")
files.download("qwen-sql-finetuned.zip")

print("\n" + "="*70)
print("DONE! Follow these steps:")
print("="*70)
print("""
1. Download qwen-sql-finetuned.zip from the popup
2. Extract it locally:
   - Windows: Right-click → Extract All
   - Mac/Linux: unzip qwen-sql-finetuned.zip

3. Copy local_inference.py to your Text_to_sql folder

4. Run locally:
   python local_inference.py --model ./qwen-sql-finetuned

5. Evaluate with fine-tuned model:
   python spider_eval_generate_with_exec.py --model ./qwen-sql-finetuned --limit 100
""")
