from docx import Document
import sys

def search_phrases(file_path, phrases):
    try:
        doc = Document(file_path)
        for i, para in enumerate(doc.paragraphs):
            for phrase in phrases:
                if phrase.lower() in para.text.lower():
                    print(f"--- Found '{phrase}' in paragraph {i} ---")
                    print(para.text)
                    print("----------------------------------------")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_phrases.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    phrases = ["acceleration", "torque", "barrier function", "cbf", "quadratic programming", "qp"]
    search_phrases(file_path, phrases)