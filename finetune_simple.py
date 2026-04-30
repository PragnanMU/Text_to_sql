"""
Fine-tune Qwen2.5-7B on Spider Text-to-SQL dataset.
Compatible with transformers 4.30+ and 5.x versions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json

def setup_model_and_tokenizer():
    """Load base model and tokenizer."""
    print("Loading Qwen2.5-7B model...")
    
    model_name = "Qwen/Qwen2.5-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded: {model_name}")
    print(f"  Parameters: {model.num_parameters() / 1e9:.1f}B")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Load and prepare fine-tuning dataset."""
    print("\nPreparing dataset...")
    
    # Load JSONL
    dataset = load_dataset("json", data_files="finetuning_data.jsonl", split="train")
    
    def format_prompt(example):
        """Format for training."""
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
        return {"text": prompt}
    
    dataset = dataset.map(format_prompt, remove_columns=["instruction", "input", "output", "db_id"])
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
    
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Train/val split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Dataset ready:")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Val: {len(dataset['test'])} samples")
    
    return dataset

def train_model(model, tokenizer, dataset):
    """Fine-tune using Hugging Face Trainer."""
    print("\nStarting fine-tuning...")
    
    try:
        from transformers import TrainingArguments, Trainer
    except ImportError:
        print("❌ Error: transformers library not installed properly")
        print("   Run: pip install -U transformers")
        return None
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./qwen-sql-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced for stability
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=2e-4,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=[],  # No wandb
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Train
    print("Training started...")
    try:
        trainer.train()
    except Exception as e:
        print(f"⚠ Training error: {e}")
        print("  Attempting to save partial model...")
        model.save_pretrained("./qwen-sql-finetuned-partial")
        tokenizer.save_pretrained("./qwen-sql-finetuned-partial")
        return None
    
    # Save
    model.save_pretrained("./qwen-sql-finetuned")
    tokenizer.save_pretrained("./qwen-sql-finetuned")
    
    print("\n✓ Fine-tuning complete!")
    print("  Model saved to: ./qwen-sql-finetuned")
    
    return model

def test_model(model, tokenizer):
    """Test fine-tuned model."""
    print("\nTesting model...")
    
    test_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate a SQL query for this question.

SCHEMA:
CREATE TABLE student (id int, name text, age int);

QUESTION: How many students are there?

### Response:"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test output:\n{response}")

def main():
    print("="*70)
    print("FINE-TUNING QWEN2.5-7B ON SPIDER")
    print("="*70)
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare data
    dataset = prepare_dataset(tokenizer)
    
    # Train
    model = train_model(model, tokenizer, dataset)
    
    if model:
        test_model(model, tokenizer)
    
    print("\n" + "="*70)
    print("EVALUATE FINE-TUNED MODEL:")
    print("="*70)
    print("""
python spider_eval_generate_with_exec.py \\
  --model ./qwen-sql-finetuned \\
  --limit 100 \\
  --output spider_results_finetuned_100.csv
""")

if __name__ == "__main__":
    main()
