from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from huggingface_hub import login
import os
from flask_cors import CORS  # Add CORS support

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Hugging Face token
HF_TOKEN = "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"

# Login to Hugging Face Hub
login(token=HF_TOKEN)

# Global model variables
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    try:
        print("Loading model...")
        tokenizer = GPT2Tokenizer.from_pretrained("arshdeepawar/gpt2LegalFinetuned")
        model = GPT2LMHeadModel.from_pretrained("arshdeepawar/gpt2LegalFinetuned")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Load model when starting the app
load_model()

@app.route('/legal-assistant')
def legal_assistant():
    return render_template('legal_assistant.html')

@app.route('/api/ask-legal', methods=['POST'])
def ask_legal():
    try:
        # Get JSON data
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Format the question
        formatted_question = f"### Question:\n{question}\n\n### Answer:\n"
        
        # Tokenize and generate
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
        
        # Decode and clean the output
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

@app.route('/')
def home():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)