# Text-to-SQL Dynamic Query App with Voice Input

A powerful Streamlit application that converts natural language questions to SQL queries using AI, with the added capability of voice input for hands-free interaction. The app supports both voice and text input methods, making it accessible and user-friendly.

## 🚀 Features

### Core Functionality
- **Natural Language to SQL**: Convert plain English questions to SQL queries using AI
- **Voice Input**: Record questions using your microphone for hands-free operation
- **Text Input**: Traditional text input for manual question entry
- **SQL Schema Upload**: Upload your own SQL schema files for custom databases
- **Query Validation**: Automatic validation of generated SQL queries
- **Results Display**: View query results in formatted tables
- **Local Speech Recognition**: No API keys required for audio transcription

### Voice Input Capabilities
- **Real-time Recording**: Record audio directly through your microphone
- **Speech-to-Text**: Convert spoken questions to text using Google's free Speech Recognition API
- **Editable Transcription**: Edit transcribed text if needed before processing
- **Seamless Integration**: Voice input flows directly into the same NLP pipeline as text input

## 📁 Project Structure

```
Text_to_sql/
├── app.py                 # Main Streamlit application
├── audio.py              # Audio transcription and recording functions
├── main.py               # Core application logic and database operations
├── LLM_model.py          # AI model for SQL generation
├── SchemaRetriever.py    # Schema retrieval and storage functionality
├── SQLValidatorAgent.py  # SQL query validation
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
├── ecommerce_schema.sql  # Sample SQL schema file
├── ecommerce.db         # Sample database
└── schema/              # ChromaDB schema storage directory
```

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Microphone (for voice input)
- Internet connection (for speech recognition)
- Google API key (for AI model)

### Step 1: Clone or Download
```bash
# If using git
git clone https://github.com/PragnanMU/Text_to_sql.git
cd Text_to_sql

# Or download and extract the project files
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Google API Key
The application requires a Google API key for the AI model functionality. Follow these steps:

1. **Get a Google API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated API key

2. **Create Environment File**:
   ```bash
   # Create a .env file in the project root
   touch .env
   ```

3. **Add API Key to .env**:
   ```bash
   # Add this line to your .env file
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   **Example .env file:**
   ```
   GOOGLE_API_KEY=AIzaSyC_Your_Actual_API_Key_Here
   ```

4. **Security Note**: 
   - Never commit your `.env` file to version control
   - The `.gitignore` file is already configured to exclude `.env` files
   - Keep your API key secure and don't share it publicly

### Step 4: Run the Application
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload SQL Schema
- Click "Browse files" to upload your `.sql` schema file
- The app supports any standard SQL schema format
- A sample `ecommerce_schema.sql` is included for testing

### 2. Input Your Question

#### Voice Input Method:
1. **Start Recording**: Click "🎤 Start Recording" button
2. **Speak Clearly**: Ask your question in natural language
3. **Stop Recording**: Click "⏹️ Stop Recording" when done
4. **Transcribe**: Click "📝 Transcribe Recording" to convert speech to text
5. **Edit (Optional)**: Modify the transcribed text if needed
6. **Ask**: Click "Ask" to process the question

#### Text Input Method:
1. **Type Question**: Enter your question in the text area
2. **Ask**: Click "Ask" to process the question

### 3. View Results
- **Generated SQL**: View the AI-generated SQL query
- **Query Summary**: Read the explanation of what the query does
- **Results Table**: See the query results in a formatted table
- **Validation Status**: Confirm the SQL query is valid

## 🎯 Example Questions

Try these sample questions with the included `ecommerce_schema.sql`:

- "Show me all customers from New York"
- "What are the top 5 products by sales?"
- "List orders from the last 30 days"
- "How many customers do we have in each state?"
- "Show me products with price greater than $100"

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Speech Recognition**: Google Speech Recognition API (free)
- **Audio Processing**: PyAudio for microphone access
- **AI Model**: Google Gemini AI for SQL generation
- **Database**: SQLite for data storage
- **Schema Storage**: ChromaDB for vector storage

### Key Components
- **`app.py`**: Main Streamlit interface and user interaction
- **`audio.py`**: Audio recording and transcription functions
- **`main.py`**: Core application logic and database operations
- **`LLM_model.py`**: AI model for converting natural language to SQL
- **`SchemaRetriever.py`**: Manages database schema storage and retrieval
- **`SQLValidatorAgent.py`**: Validates generated SQL queries

## 🎤 Voice Recording Tips

- **Speak Clearly**: Enunciate your words for better transcription
- **Minimize Background Noise**: Record in a quiet environment
- **Normal Pace**: Speak at your normal conversational speed
- **Complete Sentences**: Form complete questions for better results
- **Edit if Needed**: Review and edit transcribed text before processing

## 🔒 Privacy & Security

- **Google API Key**: Required for AI model functionality (speech recognition is free)
- **Local Processing**: All SQL generation and validation happens locally
- **No Data Storage**: Audio recordings are not stored permanently
- **Secure**: API keys are stored in `.env` files (excluded from version control)

## 📞 Support

If you encounter any issues or have questions:
1. Ensure your Google API key is properly set in the `.env` file
2. Check that all dependencies are installed correctly
3. Verify your microphone permissions
4. Try with the sample `ecommerce_schema.sql` file first

---

**Happy Querying! 🎉**

## Spider Batch Testing (No UI)

Use `spider_eval.py` to test model outputs directly on Spider questions and export predictions to CSV.

```bash
python spider_eval.py --spider-root spider --dataset dev.json --output spider_dev_predictions.csv --model qwen2.5:7b
```

Useful options:

- `--limit 100` -> run only first 100 examples
- `--start-index 200` -> resume from a specific dataset index
- `--dataset train_spider.json` -> switch split

CSV columns include: `index`, `db_id`, `question`, `gold_sql`, `predicted_sql`, `summary`, `status`, `error`, `elapsed_sec`.

