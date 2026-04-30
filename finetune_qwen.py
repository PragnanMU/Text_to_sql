"""
Fine-tune Qwen2.5-7B on Spider Text-to-SQL dataset using Hugging Face.

STEP 1: Install dependencies
  pip install -U transformers torch accelerate unsloth xformers bitsandbytes

STEP 2: Run this script
  python finetune_qwen.py

This will:
- Load qwen2.5:7b from Hugging Face
- Train on 1000 Spider examples
- Save fine-tuned model to ./qwen-sql-finetuned
- Test on sample questions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import json

def setup_model_and_tokenizer():
    """Load base model and tokenizer from Hugging Face."""
    print("Loading Qwen2.5-7B model...")
    
    model_name = "Qwen/Qwen2.5-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded: {model_name}")
    print(f"  Parameters: {model.num_parameters() / 1e9:.1f}B")
    print(f"  Device: {next(model.parameters()).device}")
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Load and prepare fine-tuning dataset."""
    print("\nPreparing dataset...")
    
    # Load from JSONL file
    dataset = load_dataset("json", data_files="finetuning_data.jsonl", split="train")
    
    def format_prompt(example):
        """Format instruction-response for training."""
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
        return {"text": prompt}
    
    # Format dataset
    dataset = dataset.map(format_prompt, remove_columns=["instruction", "input", "output", "db_id"])
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Split into train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Dataset prepared:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Val samples: {len(dataset['test'])}")
    
    return dataset

def train_model(model, tokenizer, dataset):
    """Fine-tune the model on Spider dataset."""
    print("\nStarting fine-tuning...")
    
    training_args = TrainingArguments(
        output_dir="./qwen-sql-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-4,
        save_strategy="epoch",
        eval_strategy="epoch",  # Changed from evaluation_strategy
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),  # Use bfloat16 if CUDA available
        report_to=["none"],  # Disable wandb
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    model.save_pretrained("./qwen-sql-finetuned")
    tokenizer.save_pretrained("./qwen-sql-finetuned")
    
    print("\n✓ Fine-tuning complete!")
    print("  Model saved to: ./qwen-sql-finetuned")
    
    return model

def test_finetuned_model(model, tokenizer):
    """Test the fine-tuned model on sample questions."""
    print("\nTesting fine-tuned model...")
    
    test_prompts = [
        "Generate a SQL query: How many students are there? Schema: CREATE TABLE student (id int, name text);",
        "Generate a SQL query: List all employee names. Schema: CREATE TABLE employee (id int, name text, dept_id int);",
    ]
    
    model.eval()
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

def main():
    print("="*70)
    print("FINE-TUNING QWEN2.5-7B ON SPIDER TEXT-TO-SQL DATASET")
    print("="*70)
    
    # Step 1: Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Step 2: Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Step 3: Train
    model = train_model(model, tokenizer, dataset)
    
    # Step 4: Test
    test_finetuned_model(model, tokenizer)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. Convert to Ollama format (optional):
   ollama create qwen-sql-finetuned -f ./Modelfile
   
2. Use fine-tuned model in evaluation:
   python spider_eval_generate_with_exec.py --model qwen-sql-finetuned --limit 100

3. Expected accuracy improvement: +20-30%
   - Before: 74% (with few-shot prompting)
   - After fine-tuning: 85%+ expected
""")

if __name__ == "__main__":
    main()
