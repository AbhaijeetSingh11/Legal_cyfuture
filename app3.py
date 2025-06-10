# import gradio as gr
# import joblib
# import nltk
# from nltk.corpus import stopwords

# # Download NLTK stopwords
# nltk.download('stopwords')

# # Load the saved model and vectorizer
# model = joblib.load("model.joblib")
# vectorizer = joblib.load("vectorizer.joblib")

# # Define preprocessing function
# def preprocess_input(text):
#     stop_words = set(stopwords.words('english'))
#     words = text.lower().split()
#     words = [word for word in words if word not in stop_words]
#     return ' '.join(words)

# # Define prediction function
# def predict_outcome(case_details, party_type):
#     try:
#         preprocessed = preprocess_input(case_details)
#         input_vector = vectorizer.transform([preprocessed])
#         prediction = model.predict(input_vector)[0]
#         result = "win" if prediction == 1 else "lose"

#         if party_type.lower() == "prosecution":
#             return f"As the prosecution, you are likely to {result} the case."
#         elif party_type.lower() == "defense":
#             return f"As the defense, you are likely to {result} the case."
#         else:
#             return "Invalid party type. Please select either Prosecution or Defense."
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

# # Gradio interface
# interface = gr.Interface(
#     fn=predict_outcome,
#     inputs=[
#         gr.Textbox(lines=5, label="Case Details"),
#         gr.Radio(["Prosecution", "Defense"], label="Party Type")
#     ],
#     outputs="text",
#     title="Legal Case Outcome Predictor",
#     description="Predict legal case outcomes using case details and party type."
# )

# interface.launch()
# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
import os
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure NLTK stopwords are available
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Load model and vectorizer
try:
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except Exception as e:
    print("Error loading model or vectorizer:", e)
    model = None
    vectorizer = None

# Preprocessing function
def preprocess_input(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index3.html')

# API endpoint for prediction
@app.route('/api/predict-outcome', methods=['POST'])
def predict_outcome():
    try:
        if model is None or vectorizer is None:
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
        prediction = model.predict(input_vector)[0]
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
    app.run(port=5000, debug=True)
