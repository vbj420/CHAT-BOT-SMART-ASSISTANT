from flask import Flask, request, jsonify
from groq import Groq
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np
import os


# Initialize Flask application
app = Flask(__name__)

# Initialize Groq client
API_KEY = "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"
client = Groq(api_key=API_KEY)

UPLOAD_FOLDER = "uploaded_pdfs"  # Folder containing uploaded PDFs

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

# Function to embed chunks
def embed_chunks(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Function to create FAISS index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Dimensionality of the embedding vectors
    index = faiss.IndexFlatL2(dim)  # FAISS index for L2 distance
    index.add(embeddings.cpu().numpy())  # Add embeddings to the FAISS index
    return index

# Process PDFs into individual FAISS indices
def process_uploaded_pdfs():
    pdf_indices = {}
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    for pdf_file in os.listdir(UPLOAD_FOLDER):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text_into_chunks(text, chunk_size=1000)
            embeddings = embed_chunks(chunks)
            index = create_faiss_index(embeddings)
            pdf_indices[pdf_file] = {"index": index, "chunks": chunks, "model": model}
    return pdf_indices

@app.route('/query', methods=['POST'])
def query_response():
    try:
        # Get JSON data from the POST request
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Please provide a 'query'."}), 400
        
        # Process the PDFs and build indices
        pdf_indices = process_uploaded_pdfs()
        if not pdf_indices:
            return jsonify({"error": "No PDF files found in the folder."}), 400
        
        # Retrieve relevant context from all PDFs
        all_contexts = []
        for pdf_name, data in pdf_indices.items():
            index = data["index"]
            chunks = data["chunks"]
            model = data["model"]
            
            # Embed the query and search the index
            query_embedding = model.encode([query], convert_to_tensor=True)
            distances, indices = index.search(query_embedding.cpu().numpy(), 3)  # Top 3 results
            
            # Retrieve the most relevant chunks
            context = [chunks[i] for i in indices[0] if i < len(chunks)]
            all_contexts.append((pdf_name, context))
        
        # Combine the retrieved contexts into a single string
        combined_context = ""
        for pdf_name, context in all_contexts:
            combined_context += f"From {pdf_name}:\n" + "\n".join(context) + "\n\n"

        # Use Groq to get a chatbot response
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": combined_context + "\n" + query
            }],
            model="llama3-70b-8192"
        )

        # Return the chatbot's response
        response = chat_completion.choices[0].message.content
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, use_reloader=False)
