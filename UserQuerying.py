import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import requests
import time
from PyPDF2 import PdfReader, PdfWriter


SERVER_URL = "http://127.0.0.1:5000"  # Replace with your server URL

uploaded_files = []  # List to keep track of uploaded files

def upload_pdf():
    """Upload a PDF file and append its content to usingnow.pdf."""
    global uploaded_files

    if len(uploaded_files) >= 5:
        messagebox.showwarning("Warning", "You can upload a maximum of 5 files!")
        return

    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if file_path:
        try:
            uploaded_files.append(file_path)  # Add to the list of uploaded files

            # Merge the uploaded file into 'usingnow.pdf'
            writer = PdfWriter()

            # Include existing content of usingnow.pdf if it exists
            try:
                with open("usingnow.pdf", "rb") as existing_file:
                    reader = PdfReader(existing_file)
                    for page in reader.pages:
                        writer.add_page(page)

            except FileNotFoundError:
                pass  # If the file doesn't exist, proceed to create it

            # Add the new file's content
            with open(file_path, "rb") as src_file:
                reader = PdfReader(src_file)
                for page in reader.pages:
                    writer.add_page(page)

            # Write updated content to usingnow.pdf
            with open("usingnow.pdf", "wb") as dest_file:
                writer.write(dest_file)

            messagebox.showinfo("Success", f"File {len(uploaded_files)} uploaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload PDF: {str(e)}")

def finalize_upload():
    """Finalize the upload process and open the chatbot interface."""
    if not uploaded_files:
        messagebox.showwarning("Warning", "No files uploaded yet!")
        return

    messagebox.showinfo("Success", "All files uploaded successfully! Opening chatbot interface...")
    open_chat_interface()


def typing_effect(widget, text, tag):
    """Display text character by character to emulate typing effect."""
    for char in text:
        widget.insert(tk.END, char, tag)
        widget.update()
        time.sleep(0.02)  # Typing speed


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
            # Display the user's query
            chat_display.config(state=tk.NORMAL)
            chat_display.insert(tk.END, "You: ", "user_box")
            chat_display.insert(tk.END, f"{query}", "user")
            chat_display.insert(tk.END, "\n\n")  # Add newline separation

            # Clear the user input box
            query_entry.delete(0, tk.END)

            # Display the bot's response with typing effect
            chat_display.insert(tk.END, "Bot: ", "bot_box")
            typing_effect(chat_display, f"{bot_response}\n\n", "bot")

            # Auto-scroll to the latest content
            chat_display.see(tk.END)

            chat_display.config(state=tk.DISABLED)
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
root.geometry("700x550")  # Slightly wider frame

# Upload PDF Frame
upload_frame = tk.Frame(root)
upload_frame.pack(fill=tk.BOTH, expand=True)

upload_label = tk.Label(upload_frame, text="Upload up to 5 PDFs to start chatting:", font=("Arial", 14))
upload_label.pack(pady=20)

upload_button = tk.Button(upload_frame, text="Upload PDF", command=upload_pdf, font=("Arial", 12))
upload_button.pack(pady=10)

done_button = tk.Button(upload_frame, text="Done", command=finalize_upload, font=("Arial", 12))
done_button.pack(pady=10)

# Chat Frame
chat_frame = tk.Frame(root)

chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=80, height=30, state=tk.DISABLED, font=("Arial", 10))
chat_display.pack(pady=10, padx=10)

# Add tags for styling chat display
chat_display.tag_configure("user", foreground="blue", font=("Arial", 10))
chat_display.tag_configure("bot", foreground="green", font=("Arial", 10))
chat_display.tag_configure("user_box", background="lightblue", font=("Arial", 10, "bold"))
chat_display.tag_configure("bot_box", background="lightgreen", font=("Arial", 10, "bold"))

input_frame = tk.Frame(chat_frame)
input_frame.pack(fill=tk.X, padx=10, pady=5)

query_entry = tk.Entry(input_frame, font=("Arial", 12), width=55)
query_entry.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(input_frame, text="Send", command=send_query, font=("Arial", 12))
send_button.pack(side=tk.RIGHT)

# Start the main event loop
root.mainloop()
