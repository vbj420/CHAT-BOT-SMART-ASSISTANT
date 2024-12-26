"""
from groq import Groq
from sentence_transformers import SentenceTransformer

API_KEY = "your_api_key"
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def embed_chunks(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose a different model
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

import faiss
import numpy as np

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # Dimensionality of the embedding vectors
    index = faiss.IndexFlatL2(dim)  # FAISS index for L2 distance
    index.add(embeddings.cpu().numpy())  # Add embeddings to the FAISS index
    return index

def retrieve_context(query, index, chunks, model, top_k=3):
    # Embed the query using the same model
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Perform the search
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    
    # Retrieve the most relevant chunks
    context = [chunks[i] for i in indices[0]]
    return context

def chatbot_context_retrieval(pdf_path, query):
    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Split the text into chunks
    chunks = split_text_into_chunks(text, chunk_size=1000)
    
    # Step 3: Embed the chunks using a pre-trained model
    embeddings = embed_chunks(chunks)
    
    # Step 4: Create the FAISS index for the embeddingss
    faiss_index = create_faiss_index(embeddings)
    
    # Step 5: Retrieve the most relevant context based on the query
    context = retrieve_context(query, faiss_index, chunks, SentenceTransformer('paraphrase-MiniLM-L6-v2'))
    
    return context

pdf_path = 'usingnow.pdf'
query = "How many characters are alive"

context = chatbot_context_retrieval(pdf_path, query)
print("Relevant context retrieved from the PDF:")

Context = ""
for c in context:
    Context +="\n"+c
client = Groq(api_key=API_KEY)

chat_completion = client.chat.completions.create(
    messages = [{
        "role":"user",
        "content":Context+"\n"+query
        }],
    model = "llama3-70b-8192"
)
print("BOT : ")
print(chat_completion.choices[0].message.content)
"""
from flask import Flask, request, jsonify
from groq import Groq
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import faiss
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Initialize Groq client
API_KEY = "your_api_key"
client = Groq(api_key=API_KEY)

# Function to extract text from PDF
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

# Function to retrieve context
def retrieve_context(query, index, chunks, model, top_k=3):
    # Embed the query using the same model
    query_embedding = model.encode([query], convert_to_tensor=True)
    
    # Perform the search
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    
    # Retrieve the most relevant chunks
    context = [chunks[i] for i in indices[0]]
    return context

@app.route('/query', methods=['POST'])
def query_response():
    try:
        # Get JSON data from the POST request
        data = request.json
        pdf_path = "usingnow.pdf"
        query = data.get('query')
        
        # Ensure both `pdf_path` and `query` are provided
        if not pdf_path or not query:
            if not query :
                return jsonify({"error": "Please provide both 'pdf_path' ."}), 400
            else :
                return jsonify({"error": "Please provide both 'query'."}), 400
                
        # Step 1: Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)

        # Step 2: Split the text into chunks
        chunks = split_text_into_chunks(text, chunk_size=1000)

        # Step 3: Embed the chunks using a pre-trained model
        embeddings = embed_chunks(chunks)

        # Step 4: Create the FAISS index for the embeddings
        faiss_index = create_faiss_index(embeddings)

        # Step 5: Retrieve the most relevant context based on the query
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        context = retrieve_context(query, faiss_index, chunks, model)

        # Combine the context into a single string
        context_text = "\n".join(context)

        # Step 6: Use Groq to get a chatbot response
        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": context_text + "\n" + query
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
