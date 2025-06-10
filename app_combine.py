from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import tempfile
from werkzeug.utils import secure_filename
import traceback
import signal
import sys

# For PDF summarizer
try:
    from pdf_summarizer import extract_pdf_text, summarize_long_text
except ImportError:
    extract_pdf_text = summarize_long_text = None

# For Legal Assistant
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    from huggingface_hub import login
except ImportError:
    GPT2LMHeadModel = GPT2Tokenizer = torch = login = None

# For Case Outcome Predictor
try:
    import joblib
    import nltk
    from nltk.corpus import stopwords
except ImportError:
    joblib = nltk = stopwords = None

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- PDF Summarizer Config ---
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# --- Legal Assistant Config ---
HF_TOKEN = "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"
model = tokenizer = device = None

def load_legal_assistant_model():
    global model, tokenizer, device
    if GPT2Tokenizer is None:
        return
    try:
        login(token=HF_TOKEN)
        tokenizer = GPT2Tokenizer.from_pretrained("arshdeepawar/gpt2LegalFinetuned")
        model = GPT2LMHeadModel.from_pretrained("arshdeepawar/gpt2LegalFinetuned")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    except Exception as e:
        print(f"Error loading legal assistant model: {str(e)}")

# --- Case Outcome Predictor Config ---
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
if nltk:
    nltk.data.path.append(nltk_data_path)
    try:
        stopwords.words('english')
    except Exception:
        nltk.download('stopwords', download_dir=nltk_data_path)

try:
    outcome_model = joblib.load("model.joblib") if joblib else None
    vectorizer = joblib.load("vectorizer.joblib") if joblib else None
except Exception as e:
    print("Error loading model or vectorizer:", e)
    outcome_model = None
    vectorizer = None

def preprocess_input(text):
    if not nltk:
        return text
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- Error handler for SIGTERM ---
def handle_sigterm(signum, frame):
    print("Received SIGTERM, shutting down gracefully...")
    sys.exit(0)
signal.signal(signal.SIGTERM, handle_sigterm)

# --- Home Page with Navigation ---
@app.route('/')
def main_home():
    return render_template('main.html')

# --- PDF Summarizer ---
@app.route('/pdf-summarizer')
def pdf_summarizer_home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if extract_pdf_text is None or summarize_long_text is None:
        return jsonify({'status': 'error', 'error': 'PDF summarizer not available'}), 500
    if 'pdf' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file uploaded'}), 400
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'error': 'Only PDF files are allowed'}), 400
    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        text = extract_pdf_text(temp_path)
        if not text:
            return jsonify({
                'status': 'error',
                'error': 'Failed to extract text from PDF (may be empty or encrypted)'
            }), 400
        summary = summarize_long_text(text)
        if not summary:
            summary = text[:4000]
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    except Exception as e:
        app.logger.error(f"Error processing PDF: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to process PDF (see server logs)'
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

# --- Legal Assistant ---
@app.route('/legal-assistant')
def legal_assistant_home():
    return render_template('index2.html')

@app.route('/api/ask-legal', methods=['POST'])
def ask_legal():
    if model is None or tokenizer is None:
        load_legal_assistant_model()
    if model is None or tokenizer is None:
        return jsonify({"success": False, "error": "Legal assistant model not loaded."}), 500
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        formatted_question = f"### Question:\n{question}\n\n### Answer:\n"
        input_ids = tokenizer.encode(formatted_question, return_tensors="pt").to(device)
        output = model.generate(
            input_ids,
            max_length=300,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = full_response.split("### Answer:")[-1].strip()
        return jsonify({
            "success": True,
            "question": question,
            "answer": answer
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# --- Case Outcome Predictor ---
@app.route('/case-outcome-predictor')
def case_outcome_predictor_home():
    return render_template('index3.html')

@app.route('/api/predict-outcome', methods=['POST'])
def predict_outcome():
    try:
        if outcome_model is None or vectorizer is None:
            return jsonify({
                "success": False,
                "error": "Model or vectorizer not loaded properly."
            })
        data = request.get_json()
        if not data or 'case_details' not in data or 'party_type' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required fields: case_details and party_type."
            })
        case_details = data['case_details']
        party_type = data['party_type'].lower()
        preprocessed = preprocess_input(case_details)
        input_vector = vectorizer.transform([preprocessed])
        prediction = outcome_model.predict(input_vector)[0]
        result = "win" if prediction == 1 else "lose"
        if party_type == "prosecution":
            return jsonify({
                "success": True,
                "result": f"As the prosecution, you are likely to {result} the case."
            })
        elif party_type == "defense":
            return jsonify({
                "success": True,
                "result": f"As the defense, you are likely to {result} the case."
            })
        else:
            return jsonify({
                "success": False,
                "error": "Invalid party type. Please select either Prosecution or Defense."
            })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"An error occurred: {str(e)}"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
