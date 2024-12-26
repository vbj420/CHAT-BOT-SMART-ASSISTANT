from flask import Flask, request, jsonify
from groq import Groq
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import re

nltk.download('punkt')  # Ensure the sentence tokenizer is available

app = Flask(__name__)

# Groq API Configuration
API_KEY = "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" # Replace with your Groq API key
client = Groq(api_key=API_KEY)

# PDF Configuration
PDF_PATH = "usingnow.pdf"  # Path to your PDF file

# Function to extract content from the PDF
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

# Load PDF content
PDF_CONTENT = extract_pdf_content(PDF_PATH)
if not PDF_CONTENT:
    print("Failed to load PDF content. Please check the file path or content.")

# Preprocess PDF content for better sentence segmentation
def preprocess_pdf_content(pdf_content):
    """
    Normalize spaces and line breaks in the PDF content.
    """
    return " ".join(pdf_content.replace("\n", " ").split())

# Find relevant context from the PDF using TfidfVectorizer and cosine similarity
def find_relevant_context(user_query, pdf_content):
    """
    Extract relevant context from the PDF based on the user query.
    """
    try:
        processed_content = preprocess_pdf_content(pdf_content)
        sentences = sent_tokenize(processed_content)

        # Vectorize the sentences and query
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sentences + [user_query])
        query_vector = vectors[-1]
        sentence_vectors = vectors[:-1]

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, sentence_vectors).flatten()
        threshold = 0.2  # Adjust this threshold as needed
        relevant_indices = [i for i, score in enumerate(similarities) if score > threshold]

        if relevant_indices:
            # Retrieve top N relevant sentences
            top_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)[:10]
            relevant_sentences = [sentences[i] for i in top_indices]
            return " ".join(relevant_sentences)

        # Fallback: Return sections containing the query
        for section in pdf_content.split("\n\n"):
            if user_query.lower() in section.lower():
                return section.strip()
        return "No relevant context found. Try rephrasing your query."
    except Exception as e:
        print(f"Error in finding relevant context: {e}")
        return "Error in finding relevant context."

# Query Groq API
def query_groq_api(final_query):
    """
    Send a query to the Groq API and retrieve the response.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": final_query}],
            model="llama3-70b-8192"  # Specify the model
        )
        return {"generated_text": chat_completion.choices[0].message.content}
    except Exception as e:
        print(f"Error querying Groq API: {str(e)}")
        return {"error": "Failed to fetch response from Groq API"}

# Handle various types of user queries
def handle_user_query(user_query):
    """
    Identify the type of user query and respond accordingly.
    """
    # Regex patterns for common actions
    summarize_pattern = re.compile(r"\b(summarize|overview|gist|brief)\b", re.IGNORECASE)
    extract_section_pattern = re.compile(r"\b(section|chapter|topic)\b", re.IGNORECASE)
    specific_question_pattern = re.compile(r"\b(who|what|when|where|why|how|explain|describe)\b", re.IGNORECASE)

    # Special cases handling
    if summarize_pattern.search(user_query):
        # Summarize the entire PDF content
        return f"Summarize the following document:\n{PDF_CONTENT}"
    elif extract_section_pattern.search(user_query):
        # Attempt to find a section based on keywords
        return f"Attempting to extract relevant sections for: {user_query}\n{find_relevant_context(user_query, PDF_CONTENT)}"
    elif specific_question_pattern.search(user_query):
        # Handle specific questions
        context = find_relevant_context(user_query, PDF_CONTENT)
        return f"Context: {context}\nUser Query: {user_query}"
    else:
        # Default to context matching for general queries
        context = find_relevant_context(user_query, PDF_CONTENT)
        return f"Context: {context}\nUser Query: {user_query}"


# Flask route for querying the model
@app.route("/query", methods=["POST"])
def query():
    """
    Handle user queries and respond with generated text from the Groq model.
    """
    data = request.json
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query is empty"}), 400

    # Debugging: Log the user query
    print(f"User Query: {user_query}")

    # Handle the user query
    final_query = handle_user_query(user_query)

    # Debugging: Log the final query
    print(f"Final Query Sent to Model: {final_query}")

    # Query the Groq API
    response = query_groq_api(final_query)

    # Handle errors in the Groq API response
    if "error" in response:
        error_message = response.get("error", "Failed to fetch response.")
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500

    generated_text = response.get("generated_text", "I couldn't understand that.")

    # Debugging: Log the model response
    print(f"Model Response: {generated_text}")

    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
