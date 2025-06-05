# ----------------------------------------------------------------------------------------------------------------------
import PyPDF2
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap
import time
from typing import List

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file with improved paragraph handling"""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if pdf_reader.is_encrypted:
                print("Error: PDF is encrypted. Text extraction aborted.")
                return ""
            
            extracted_text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Clean up text and preserve paragraph structure
                    page_text = ' '.join(page_text.split()).replace(' .', '.').replace(' ,', ',')
                    extracted_text.append(page_text)
            
            return '\n\n'.join(extracted_text)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return ""

def chunk_text(text: str, max_chunk_size: int = 3500) -> List[str]:
    """Split text into meaningful chunks while preserving context"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:  # +2 for newlines
            current_chunk += f"{para}\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = f"{para}\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def initialize_model(model_name: str = "deepseek-ai/deepseek-llm-7b-chat"):
    """Initialize and return the tokenizer and model"""
    print(f"Loading {model_name}...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Model loaded in {time.time()-start_time:.1f} seconds")
    return tokenizer, model

def generate_summary(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    text: str,
    max_new_tokens: int = 512
) -> str:
    """Generate summary for a single chunk"""
    prompt = (
        "### Instruction:\n"
        "explain me about context written in the text."
        "bring the outcomes of this text in a concise way, "
        "all important details:\n\n"
        f"{text}\n\n"
        "### Response:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.split("### Response:")[-1].strip()

def summarize_long_text(text: str, model_name: str = "deepseek-ai/deepseek-llm-7b-chat") -> str:
    """Handle long document summarization with proper chunking"""
    try:
        tokenizer, model = initialize_model(model_name)
        chunks = chunk_text(text)
        
        print(f"Processing {len(chunks)} text chunks...")
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Summarizing chunk {i}/{len(chunks)}...")
            summary = generate_summary(tokenizer, model, chunk)
            chunk_summaries.append(summary)
            time.sleep(1)  # Prevent overheating
        
        # Combine and summarize the chunk summaries
        combined_summaries = "\n\n".join(chunk_summaries)
        print("Generating final comprehensive summary...")
        final_summary = generate_summary(tokenizer, model, combined_summaries, max_new_tokens=1024)
        
        return final_summary
    
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return ""

def main():
    print("Enhanced PDF to Summary Converter")
    print("=" * 50)
    
    pdf_path = input("Enter path to PDF file: ").strip().strip('"\'')
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {os.path.abspath(pdf_path)}")
        return
    
    print("\nExtracting and cleaning text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    if not text:
        print("Failed to extract text from PDF")
        return
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    extracted_path = f"{base_name}_extracted.txt"
    with open(extracted_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✓ Cleaned text saved to: {os.path.abspath(extracted_path)}")
    
    print("\nStarting summarization process...")
    summary = summarize_long_text(text)
    # here you can change length of your summary mannualy brother 
    if not summary:
        print("Summarization failed. Using fallback method.")
        summary = text[:4000] #<<------ here you can change length of your summary mannualy brother
    
    summary_path = f"{base_name}_summary.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        #f.write(summary)
        f.write(textwrap.fill(summary, width=80))
    
    print("\n" + "=" * 50)
    print(f"✓ Final summary saved to: {os.path.abspath(summary_path)}")
    print("\nSUMMARY RESULT:")
    print("=" * 50)
    print(textwrap.fill(summary, width=80))
    print("=" * 50)

if __name__ == "__main__":
    main()
