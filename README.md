# Text-to-SQL with LangChain + Ollama

This project converts natural-language questions into SQL using a local LLM (`qwen2.5:7b` via Ollama).  
It supports:

- interactive usage through Streamlit (`app.py`)
- batch generation on Spider dataset (`spider_eval_generate_only.py`)

## Tech Stack

- `LangChain` + `Ollama` for SQL generation
- `ChromaDB` for schema storage/retrieval
- `SQLite` for query execution
- `Streamlit` for UI

## Project Files

- `app.py`: Streamlit app for schema upload + question answering
- `LLM_model.py`: SQL generation model wrapper (Ollama)
- `SchemaRetriever.py`: schema extraction and Chroma storage
- `SQLValidatorAgent.py`: optional SQL validator logic
- `spider_eval.py`: Spider batch script with resumable CSV output
- `spider_eval_generate_only.py`: Spider batch script (generation-only, no validator)

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Install and prepare Ollama

Install from [ollama.com](https://ollama.com), then:

```bash
ollama pull qwen2.5:7b
```

Make sure Ollama is running before starting scripts/app.

## Run Streamlit App

```bash
streamlit run app.py
```

## Spider Batch Evaluation (No UI)

### Generate dev predictions CSV

```bash
python spider_eval_generate_only.py --spider-root spider --dataset dev.json --output spider_dev_predictions_no_validator.csv --model qwen2.5:7b
```

### Generate train predictions CSV

```bash
python spider_eval_generate_only.py --spider-root spider --dataset train_spider.json --output spider_train_predictions_no_validator.csv --model qwen2.5:7b
```

### Useful options

- `--resume`: skip already completed indices in existing CSV
- `--flush-every 1`: write every row immediately (best safety if stopped with Ctrl+C)
- `--limit N`: run only first N rows
- `--start-index N`: start from a specific row
- `--preview N`: print N sample predictions at end

## Output CSV Columns

- `index`
- `db_id`
- `question`
- `gold_sql`
- `predicted_sql`
- `summary`
- `status`
- `error`
- `elapsed_sec`

## Notes

- No Google API key is required.
- Inference is local through Ollama.
- Prediction CSV files can get large; keep them out of commits unless needed.

