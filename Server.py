from flask import Flask, request, jsonify
import requests
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


app = Flask(__name__)

# Hugging Face API Token and Model Configurations
API_TOKEN = "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"
MODEL_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Retry mechanism configurations
MAX_RETRIES = 15
RETRY_INTERVAL = 20  # seconds

# Load and extract content from the PDF
PDF_PATH = "usingnow.pdf"
def extract_pdf_content(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + " "
        return content.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Extract content from usingnow.pdf
PDF_CONTENT = extract_pdf_content(PDF_PATH)
if not PDF_CONTENT:
    print("Failed to load PDF content. Please check the file path or content.")
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Ensure the sentence tokenizer is available

def preprocess_pdf_content(pdf_content):
    """
    Preprocess PDF content to ensure proper sentence segmentation.
    """
    # Normalize spaces and line breaks
    return " ".join(pdf_content.replace("\n", " ").split())

def find_relevant_context(user_query, pdf_content):
    """
    Improved context extraction using vectorization and fallback to relevant sections.
    """
    try:
        # Preprocess and tokenize PDF content
        processed_content = preprocess_pdf_content(pdf_content)
        sentences = sent_tokenize(processed_content)

        # Vectorize sentences and user query
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sentences + [user_query])
        query_vector = vectors[-1]
        sentence_vectors = vectors[:-1]

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, sentence_vectors).flatten()

        # Define similarity threshold
        threshold = 0.2  # Adjust based on experimentation
        relevant_indices = [i for i, score in enumerate(similarities) if score > threshold]

        if relevant_indices:
            # Retrieve top N sentences
            top_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)[:10]
            relevant_sentences = [sentences[i] for i in top_indices]
            return " ".join(relevant_sentences)

        # Fallback: Return entire section related to the query
        for section in pdf_content.split("\n\n"):
            if user_query.lower() in section.lower():
                return section.strip()
        return "No relevant context found. Try rephrasing your query."
    except Exception as e:
        print(f"Error in context matching: {e}")
        return "Error in finding relevant context."


def query_huggingface_api(final_query):
    """
    Send a query to Hugging Face API with retries in case the model is still loading.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                MODEL_URL,
                headers=HEADERS,
                json={"inputs": final_query}
            )
            if response.status_code == 200:
                # Successful response
                return response.json()
            elif response.status_code == 503:
                # Model is still loading
                print(f"Attempt {attempt + 1}: Model is loading, retrying in {RETRY_INTERVAL} seconds...")
                time.sleep(RETRY_INTERVAL)
            else:
                # Other errors
                error_message = response.json().get("error", "Unknown error occurred.")
                print(f"Error from Hugging Face API: {error_message}")
                return {"error": error_message}
        except Exception as e:
            print(f"Exception during API call: {str(e)}")
            return {"error": "Internal server error"}
    
    # If retries are exhausted
    return {"error": "Model did not load in time. Please try again later."}

@app.route("/query", methods=["POST"])
def query():
    """Handle user queries and respond using the Hugging Face model."""
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query is empty"}), 400

    # Debugging: Log the user query
    print(f"User Query: {user_query}")

    # Find relevant context from the PDF
    relevant_context = find_relevant_context(user_query, PDF_CONTENT)

    # Combine the query with relevant context
    final_query = (
        f"Context: {relevant_context}\nUser Query: {user_query}"
        if relevant_context
        else f"User Query: {user_query}"
    )

    # Debugging: Log the final query and matched context
    print(f"Matched Context: {relevant_context}")
    print(f"Final Query Sent to Model: {final_query}")

    # Query Hugging Face API with retry mechanism
    response = query_huggingface_api(final_query)

    # Adjust for list-based response format
    if isinstance(response, list):
        generated_text = response[0] if response else "I couldn't generate a response."
    elif isinstance(response, dict):
        generated_text = response.get("generated_text", "I couldn't understand that.")
    else:
        generated_text = "Unexpected response format received."

    # Debugging: Log the model response
    print(f"Model Response: {generated_text}")

    if "error" in response:
        error_message = response.get("error", "Failed to fetch response.")
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500

    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
