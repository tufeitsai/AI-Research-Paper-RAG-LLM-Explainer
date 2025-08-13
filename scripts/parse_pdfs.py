import os
import fitz  # PyMuPDF
import json

PDF_DIR = "../data/raw_pdfs"
TEXT_DIR = "../data/extracted_texts"


os.makedirs(TEXT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Failed to parse {pdf_path}: {e}")
        return ""

def extract_all():
    files = os.listdir(PDF_DIR)
    print(f"Found {len(files)} PDFs.")

    for filename in files:
        if not filename.endswith(".pdf"):
            continue
        paper_id = filename.replace(".pdf", "")
        output_path = os.path.join(TEXT_DIR, f"{paper_id}.txt")
        if os.path.exists(output_path):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"Extracting from: {filename}")
        text = extract_text_from_pdf(pdf_path)

        if len(text) > 500:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(f"⚠Skipped {filename} (too short)")

if __name__ == "__main__":
    extract_all()
    print(f"\n Done! Extracted text saved to `{TEXT_DIR}`.")
