import os
import io
import json
import csv
from bs4 import BeautifulSoup

def parse_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            return None, "Password Protected PDF"
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        return text, None
    except Exception as e:
        return None, f"PDF Parsing Error: {e}"

def parse_docx(file_bytes):
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text, None
    except Exception as e:
        return None, f"DOCX Parsing Error: {e}"

def parse_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8'), None
    except UnicodeDecodeError:
        try:
            return file_bytes.decode('latin-1'), None
        except Exception as e:
            return None, f"TXT Encoding Error: {e}"

def parse_csv(file_bytes):
    try:
        text = file_bytes.decode('utf-8')
        reader = csv.reader(io.StringIO(text))
        result = []
        for row in reader:
            result.append(" ".join(row))
        return "\n".join(result), None
    except Exception as e:
        return None, f"CSV Parsing Error: {e}"

def parse_json(file_bytes):
    try:
        data = json.loads(file_bytes.decode('utf-8'))
        # Try to flatten json values into a string
        if isinstance(data, dict):
            text = " ".join([str(v) for v in data.values()])
        elif isinstance(data, list):
            text = " ".join([str(item) for item in data])
        else:
            text = str(data)
        return text, None
    except Exception as e:
        return None, f"JSON Parsing Error: {e}"

def parse_html(file_bytes):
    try:
        text = file_bytes.decode('utf-8')
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator="\n"), None
    except Exception as e:
        return None, f"HTML Parsing Error: {e}"

def parse_image(file_bytes):
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        return text, None
    except ImportError:
        return None, "OCR libraries (pytesseract/Pillow) not installed"
    except Exception as e:
        return None, f"Image OCR Error: {e}"

def parse_file(filename, file_bytes):
    """
    Parses a file based on its extension.
    Returns (text, error_message)
    """
    if not file_bytes:
        return None, "Empty File"
        
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        return parse_pdf(file_bytes)
    elif ext == ".docx":
        return parse_docx(file_bytes)
    elif ext in [".txt", ".md"]:
        return parse_txt(file_bytes)
    elif ext == ".csv":
        return parse_csv(file_bytes)
    elif ext == ".json":
        return parse_json(file_bytes)
    elif ext in [".html", ".htm"]:
        return parse_html(file_bytes)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return parse_image(file_bytes)
    else:
        return None, f"Unsupported Format: {ext}"
