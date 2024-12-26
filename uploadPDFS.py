import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import fitz  # PyMuPDF
import os


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as pdf_file:
            text = ""
            for page in pdf_file:
                text += page.get_text()
            return text
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read PDF: {str(e)}")
        return ""

# Function to save extracted text to a temp file
def save_text_to_temp_file(text):
    try:
        temp_file_path = "temptext.txt"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # Remove existing file
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(text)
        messagebox.showinfo("Done", "The extracted text has been saved to 'temptext.txt'.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save text: {str(e)}")

# Function to handle file upload
def upload_file():
    global extracted_text
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if file_path:
        extracted_text = extract_text_from_pdf(file_path)
        if extracted_text:
            text_area.delete(1.0, tk.END)  # Clear the text area
            text_area.insert(tk.END, extracted_text)  # Display extracted text

# Function to handle Done button click
def on_done():
    global extracted_text
    if extracted_text:
        save_text_to_temp_file(extracted_text)
    else:
        messagebox.showwarning("Warning", "No text available to save. Please upload a PDF file first.")



# Create the main window
root = tk.Tk()
root.title("PDF Text Extractor")
root.geometry("800x600")

# Initialize global variable
extracted_text = ""

# Create a button to upload files
upload_button = tk.Button(root, text="Upload PDF", command=upload_file)
upload_button.pack(pady=10)

# Create a scrolled text widget to display extracted text
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=30)
text_area.pack(pady=10)

# Create a Done button
done_button = tk.Button(root, text="Done", command=on_done)
done_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
