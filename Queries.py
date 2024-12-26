import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import requests
import os

SERVER_URL = "http://127.0.0.1:5000"  # Replace with your server URL

def upload_pdf():
    """Upload a PDF file and prepare it for the chatbot."""
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    
    if file_path:
        try:
            # Copy the selected file to 'usingnow.pdf'
            with open(file_path, "rb") as src_file:
                with open("usingnow.pdf", "wb") as dest_file:
                    dest_file.write(src_file.read())
            messagebox.showinfo("Success", "PDF uploaded successfully! Opening chatbot interface...")
            open_chat_interface()  # Open the chatbot interface
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload PDF: {str(e)}")

def send_query():
    """Send a user query to the server and display the response."""
    query = query_entry.get().strip()
    if not query:
        messagebox.showwarning("Warning", "Query cannot be empty!")
        return

    try:
        # Send the query to the server
        response = requests.post(f"{SERVER_URL}/query", json={"query": query})
        if response.status_code == 200:
            bot_response = response.json().get("response", "No response received.")
            # Display the chat messages in sequence
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, f"You: {query}\n", "user")
            chat_display.insert(tk.END, f"Bot: {bot_response}\n\n", "bot")
            chat_display.config(state=tk.DISABLED)
            chat_display.see(tk.END)  # Scroll to the latest message
            query_entry.delete(0, tk.END)  # Clear the entry field
        else:
            messagebox.showerror("Error", f"Server error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send query: {str(e)}")

def open_chat_interface():
    """Open the chatbot interface for querying."""
    upload_frame.pack_forget()  # Hide the upload frame
    chat_frame.pack(fill=tk.BOTH, expand=True)  # Show the chat interface

# Tkinter GUI
root = tk.Tk()
root.title("PDF Chatbot")
root.geometry("600x500")

# Upload PDF Frame
upload_frame = tk.Frame(root)
upload_frame.pack(fill=tk.BOTH, expand=True)

upload_label = tk.Label(upload_frame, text="Upload a PDF to start chatting:", font=("Arial", 14))
upload_label.pack(pady=20)

upload_button = tk.Button(upload_frame, text="Upload PDF", command=upload_pdf)
upload_button.pack(pady=20)

# Chat Frame
chat_frame = tk.Frame(root)

chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=70, height=25, state=tk.DISABLED)
chat_display.pack(pady=10, padx=10)

# Add tags for styling chat display
chat_display.tag_configure("user", foreground="blue", font=("Arial", 12, "bold"))
chat_display.tag_configure("bot", foreground="green", font=("Arial", 12))

input_frame = tk.Frame(chat_frame)
input_frame.pack(fill=tk.X, padx=10, pady=5)

query_entry = tk.Entry(input_frame, font=("Arial", 12), width=55)
query_entry.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(input_frame, text="Send", command=send_query, font=("Arial", 12))
send_button.pack(side=tk.RIGHT)
# Start the main event loop
root.mainloop()
