from docx import Document
import sys

def read_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error reading file: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_docx.py <path_to_docx>")
    else:
        file_path = sys.argv[1]
        print(read_docx(file_path))
