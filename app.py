import streamlit as st
import tempfile
import os
import pandas as pd
from main import initialize_sqlite_db, DatabaseExecutorAgent, LLM_model, SQLValidatorAgent, SchemaRetriever
from audio import transcribe_audio_data, record_audio

st.title("Text-to-SQL Dynamic Query App")

# Upload SQL file
uploaded_file = st.file_uploader("Upload your .sql schema/data file", type=["sql"])
temp_sql_path = None

if uploaded_file is not None:
    # Save uploaded SQL file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sql") as tmp:
        tmp.write(uploaded_file.read())
        temp_sql_path = tmp.name
    
    # Use a consistent collection name based on the uploaded file
    collection_name = os.path.splitext(uploaded_file.name)[0]
    
    st.success(f"Uploaded: {uploaded_file.name}")
    
    # Store schema once when file is uploaded
    try:
        # Create a custom SchemaRetriever with consistent collection name
        retriever = SchemaRetriever(temp_sql_path)
        # Override the collection name to be consistent
        retriever.collection_name = collection_name
        retriever.collection = retriever.client.get_or_create_collection(
            name=collection_name,
            embedding_function=retriever.embed_model
        )
        retriever.store_schema()
        st.success("Schema stored successfully!")
    except Exception as e:
        st.warning(f"Schema storage warning: {str(e)}")

    # -----------------------------
    # Combined Audio + Text Input
    # -----------------------------
    st.subheader("Input Your Question (Voice or Text)")

    # Initialize session state
    if "question" not in st.session_state:
        st.session_state["question"] = ""

    # Voice Input Controls
    st.markdown("**🎤 Voice Input (optional)**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎤 Start Recording"):
            try:
                with st.spinner("Recording... Speak now! Click 'Stop Recording' when done."):
                    audio_data = record_audio()
                st.session_state.audio_data = audio_data
                st.session_state.recording_done = True
                st.success("Recording completed!")
            except Exception as e:
                st.error(f"Error recording: {str(e)}")

    with col2:
        if st.button("⏹️ Stop Recording"):
            st.info("Recording stopped. Use 'Transcribe' to convert to text.")

    if st.button("📝 Transcribe Recording"):
        if hasattr(st.session_state, 'audio_data') and st.session_state.audio_data:
            try:
                with st.spinner("Transcribing..."):
                    transcribed_text = transcribe_audio_data(st.session_state.audio_data)
                    st.session_state["question"] = transcribed_text
                st.success("Transcription completed!")
            except Exception as e:
                st.error(f"Error transcribing: {str(e)}")
        else:
            st.warning("No recording found. Please record audio first.")

    # Unified Input Field (Editable — works for both text and transcribed audio)
    st.markdown("**⌨️ Type or Edit Your Question Below:**")
    st.session_state["question"] = st.text_area("Your Question:", value=st.session_state["question"], key="final_question")

    # Ask button
    if st.button("Ask") and st.session_state["question"].strip():
        question = st.session_state["question"].strip()
        temp_db_path = temp_sql_path.replace(".sql", ".db")
        db_executor = None  # Initialize to None
        try:
            # Step 1: Initialize DB
            initialize_sqlite_db(temp_sql_path, temp_db_path)

            # Step 2: Load LLM (schema already stored)
            llm = LLM_model(collection_name=collection_name)

            # Step 3: SQL Validator
            validator = SQLValidatorAgent()

            # Step 4: DB Executor
            db_executor = DatabaseExecutorAgent(temp_db_path)

            st.info(f"Question: {question}")
            
            # Generate SQL with progress indicator
            with st.spinner("Generating SQL query..."):
                result_obj = llm.generate_sql(question)
            
            summary = result_obj.get("summary", "")
            sql = result_obj.get("sql_query", "")

            if summary:
                st.markdown(f"**Summary:** {summary}")
            st.code(sql, language="sql")

            try:
                validated_sql = validator.validate_sql(question, llm.get_schema(), sql)
                st.success("SQL validated!")
            except Exception as e:
                st.warning(f"Validation failed: {str(e)}. Using original SQL.")
                validated_sql = sql

            result = db_executor.execute_query(validated_sql)
            if isinstance(result, list) and result:
                df = pd.DataFrame(result)
                st.dataframe(df)
            elif isinstance(result, list) and not result:
                st.info("Query executed successfully, but no results found.")
            else:
                st.error(result)

        except Exception as e:
            error_str = str(e)
            if "ollama" in error_str.lower() or "connection refused" in error_str.lower():
                st.error("🚫 **Ollama is not available**")
                st.warning(
                    "Please make sure Ollama is running and the model is pulled:\n\n"
                    "1. Start Ollama\n"
                    "2. Run: `ollama pull qwen2.5:7b`\n"
                    "3. Retry your question"
                )
                st.error(f"**Technical Details:**\n```\n{error_str}\n```")
            else:
                st.error(f"Error: {error_str}")
        finally:
            if db_executor is not None:
                db_executor.close()
            os.remove(temp_sql_path)
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
