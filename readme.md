```markdown
# ⚖️ LawBrief AI – Legal Tools Suite

**LawBrief AI** is a powerful AI-powered legal assistant platform built using **Flask**, offering three essential legal tools in one place:

- 📄 **PDF Summarizer** – Instantly summarize legal documents, contracts, and case files.
- 🤖 **Legal Assistant (GPT-2)** – Ask legal questions and receive AI-powered, context-aware responses.
- ⚖️ **Case Outcome Predictor** – Predict the likely outcome of a legal case based on factual details and party type.

---

## 🌟 Features

- ✨ Modern, responsive web interface
- 🔒 Secure PDF upload and summarization
- 🧠 Natural language legal Q&A using GPT-2
- 📊 ML-based legal case outcome prediction
- 🧩 Modular codebase for easy customization
- 📄 MIT Licensed for personal and commercial use

---

## 🗂️ Project Structure

```
```
LawBrief-AI/
├── app_combine.py         # Main Flask backend combining all tools
├── pdf_summarizer.py      # Logic for extracting and summarizing PDF content
├── model.joblib           # ML model for case outcome prediction
├── vectorizer.joblib      # Text vectorizer used with the prediction model
├── templates/             # HTML templates for the frontend
│   ├── main.html          # Homepage with navigation
│   ├── index.html         # UI for the PDF Summarizer
│   ├── index2.html        # UI for the Legal Assistant (GPT-2)
│   └── index3.html        # UI for the Case Outcome Predictor
├── static/                # (Optional) CSS, JavaScript, images, etc.
├── LICENSE                # MIT License file
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies (to be created if not present)
```
````
````

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Example <code>requirements.txt</code>:</summary>

```
flask
flask_cors
transformers
torch
huggingface_hub
joblib
nltk
werkzeug
```

</details>

### 3. Prepare NLTK Resources

```python
# Run this once in Python shell
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Run the Application

```bash
python app_combine.py
```

Open your browser and go to [http://localhost:5000](http://localhost:5000)

---

## 🧪 Usage

* 🏠 **Home** – Navigate between tools
* 📄 **PDF Summarizer** – Upload a PDF and get a summary
* 🤖 **Legal Assistant** – Ask legal questions via chat interface
* ⚖️ **Case Predictor** – Enter details and predict case outcome

---

## 🛠️ Customization

* 🔁 To change the Hugging Face model, update the model name in `app_combine.py`.
* 🧠 Retrain or replace `model.joblib` and `vectorizer.joblib` for a better case predictor.
* 🎨 Edit HTML files in the `templates/` folder to redesign the UI.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgments

* [Flask](https://flask.palletsprojects.com/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)
---

> © 2025 LawBrief AI – Designed to make legal tech smarter.
