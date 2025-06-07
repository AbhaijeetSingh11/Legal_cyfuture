# 🧠 Legal PDF Summarizer Web Application

A powerful web-based tool to **upload**, **analyze**, and **summarize lengthy legal PDF documents** using a large language model (`deepseek-ai/deepseek-llm-7b-chat`). Ideal for lawyers, legal researchers, or anyone needing fast insights from legal content.

---

## 📌 Features

* 📄 Upload PDF legal documents.
* 🔍 Automatic extraction and cleaning of text.
* 🤖 Chunk-wise summarization using a 7B LLM with instruction-tuned prompts.
* 🕒 Built-in timeout protection (7 minutes max per request).
* 💾 Download summaries in `.txt` format.
* 🧪 Fallback to partial content summary if the full process fails.

---

## 🏗️ Architecture Overview

```
Frontend (HTML + JS)
    ↓
Flask Backend (Python)
    ↓
Text Extraction (PyPDF2)
    ↓
Chunking & Summarization (Transformers - DeepSeek LLM)
    ↓
Final Summary Response
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legal-pdf-summarizer.git
cd legal-pdf-summarizer
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist yet, here’s a typical content:

```txt
Flask
PyPDF2
transformers
torch
```

Ensure your environment supports GPU (CUDA or bfloat16) for optimal performance.

---

### 4. Run the Web App

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 Sample Usage via CLI (Optional)

You can also use the script directly via command line:

```bash
python pdf_summarizer.py
```

You'll be prompted to input the path of a PDF file. The script extracts, summarizes, and saves both the cleaned text and summary as `.txt` files.

---

## 📂 Project Structure

```
.
├── app.py                # Flask backend
├── pdf_summarizer.py     # Core summarization logic
├── templates/
│   └── index.html        # Frontend HTML page
├── static/               # Optional: styles or JS
└── README.md             # Project documentation
```

---

## ⚙️ Configuration & Customization

* **Model Selection**: You can change the `model_name` in `summarize_long_text()` to another `transformers`-compatible LLM.
* **Summary Length**: Adjust `max_new_tokens` in `generate_summary()` or final summary generation.
* **Timeout**: Configurable in JavaScript frontend (`420000 ms = 7 minutes`).

---

## ⚠️ Notes

* For large PDFs (>50MB), consider pre-splitting before uploading.
* Encrypted or scanned PDFs may not yield proper results.
* The LLM summarizer assumes a stable internet or cached model if run locally with HuggingFace transformers.

---

## 📄 MIT License

```
MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
---

## 🙏 Acknowledgements

* [`deepseek-ai/deepseek-llm-7b-chat`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)
* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* Flask, PyPDF2
