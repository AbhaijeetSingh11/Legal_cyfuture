from flask import Flask, request, jsonify, render_template
import os
import tempfile
from werkzeug.utils import secure_filename
from pdf_summarizer import extract_pdf_text, summarize_long_text
import traceback
import signal
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Error handler for SIGTERM
def handle_sigterm(signum, frame):
    print("Received SIGTERM, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'pdf' not in request.files:
        return jsonify({'status': 'error', 'error': 'No file uploaded'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'status': 'error', 'error': 'Only PDF files are allowed'}), 400
    
    temp_path = None
    try:
        # Save to temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Extract text
        text = extract_pdf_text(temp_path)
        if not text:
            return jsonify({
                'status': 'error',
                'error': 'Failed to extract text from PDF (may be empty or encrypted)'
            }), 400
        
        # Generate summary (with timeout safety)
        summary = summarize_long_text(text)
        if not summary:
            summary = text[:4000]  # Fallback
            
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
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)