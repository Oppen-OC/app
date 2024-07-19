@echo off
REM Activate the virtual environment
call .\venv\scripts\activate

REM Run the Streamlit application
streamlit run pdf_chatbot.py
