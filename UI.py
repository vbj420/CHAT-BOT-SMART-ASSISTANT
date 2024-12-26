import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import os
import requests
import time
import shutil  # For folder cleanup

SERVER_URL = "http://127.0.0.1:5000"  # Replace with your server URL
UPLOAD_FOLDER = "uploaded_pdfs"  # Folder to store uploaded PDFs


def upload_pdf():
    """Upload a PDF file and save it in the designated folder."""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)  # Create the folder if it doesn't exist

    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if file_path:
        try:
            # Save the selected file to the UPLOAD_FOLDER
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(UPLOAD_FOLDER, file_name)
            
            # Avoid duplicate uploads
            if os.path.exists(dest_path):
                messagebox.showwarning("Warning", f"{file_name} already uploaded!")
                return
            
            with open(file_path, "rb") as src_file:
                with open(dest_path, "wb") as dest_file:
                    dest_file.write(src_file.read())

            messagebox.showinfo("Success", f"File '{file_name}' uploaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload PDF: {str(e)}")

def finalize_upload():
    """Finalize the upload process and open the chatbot interface."""
    if not os.listdir(UPLOAD_FOLDER):
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

def cleanup_uploaded_pdfs():
    """Delete all files in the uploaded_pdfs folder."""
    if os.path.exists(UPLOAD_FOLDER):
        for file_name in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_name}: {str(e)}")

def on_closing():
    """Handle application closing event."""
    cleanup_uploaded_pdfs()  # Clean up the uploaded PDFs folder
    root.destroy()  # Close the application

# Tkinter GUI
root = tk.Tk()
root.title("PDF Chatbot")
root.geometry("700x550")  # Slightly wider frame

# Set up the cleanup function on window close
root.protocol("WM_DELETE_WINDOW", on_closing)

# Upload PDF Frame
upload_frame = tk.Frame(root)
upload_frame.pack(fill=tk.BOTH, expand=True)

upload_label = tk.Label(upload_frame, text="Upload PDFs to start chatting:", font=("Arial", 14))
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
