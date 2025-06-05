# README for PDF Processor

    ## Overview

    This Python script extracts text from a PDF file and generates a concise summary using a large language model (LLM). It is designed to handle large documents by chunking the text and summarizing each part before producing a final comprehensive summary.

    ## Features

    - Extracts and cleans text from PDF files.
    - Handles encrypted PDFs gracefully.
    - Splits long documents into manageable chunks.
    - Uses a transformer-based LLM (default: `deepseek-ai/deepseek-llm-7b-chat`) for summarization.
    - Saves both the extracted text and the summary to disk.
    - Provides clear console output for each processing step.

    ## Requirements

    - Python 3.8+
    - PyPDF2
    - torch
    - transformers
    - textwrap