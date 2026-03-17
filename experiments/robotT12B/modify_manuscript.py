from docx import Document
import sys

def modify_manuscript(input_path, output_path):
    doc = Document(input_path)
    
    replacements = [
        # 1. Findings
        {
            "target": "CBF safety filtering",
            "replacement": "CDF-based cost penalties"
        },
        {
            "target": "three-level safety system",
            "replacement": "multi-level safety system"
        },
        # 2. Abstract
        {
            "target": "Control Barrier Function (CBF) safety filter",
            "replacement": "Configuration-space Distance Field (CDF) based cost penalty"
        },
        {
            "target": "revise control commands in real time",
            "replacement": "guide the sampling process in real time"
        },
        {
            "target": "hard constraint safety assurance",
            "replacement": "safety assurance"
        },
        {
            "target": "strict satisfaction of constraints",
            "replacement": "high safety awareness and feasibility"
        },
        # 3. Keywords
        {
            "target": "Control Barrier Function;",
            "replacement": ""
        },
        # 4. Intro Summary
        {
            "target": "+ filter final guarantee\"",
            "replacement": "\""
        },
        {
            "target": "+ optimization hard constraint",
            "replacement": "+ optimization verification"
        },
        {
            "target": "at the execution layer, a safety filter based on first-order CBF is introduced to perform real-time projection correction on control commands.",
            "replacement": ""
        },
        # 5. Control Formulation - Aggressive matching
        {
            "target": "joint acceleration is taken as the control input",
            "replacement": "joint position increment is taken as the control input"
        },
         # Try shorter match if above fails
        {
            "target": "acceleration is taken as the control input",
            "replacement": "position increment is taken as the control input"
        },
        # Abstract detailed matching
        {
            "target": "hard constraint safety assurance",
            "replacement": "safety assurance"
        },
        {
            "target": "upper bound for joint acceleration",
            "replacement": "upper bound for joint position increment"
        },
        {
            "target": "cost penalty of MPPI;",
            "replacement": "CDF cost penalty of MPPI." # End sentence
        }
    ]

    count = 0
    for para in doc.paragraphs:
        original_text = para.text
        modified_text = original_text
        
        for rule in replacements:
            if rule["target"] in modified_text:
                modified_text = modified_text.replace(rule["target"], rule["replacement"])
        
        if original_text != modified_text:
            para.text = modified_text
            count += 1
            print(f"Modified Paragraph: {original_text[:30]}...")

    # Also check run-level text if paragraph-level replacement failed due to formatting splits (simple approach first)
    # The `para.text = ...` above nukes formatting. For this task, preserving bolding of keywords is nice but correctness is critical.
    # Given the academic paper context, I should try to preserve formatting if possible, but identifying "runs" that split strict strings is hard.
    # The `para.text` assignment keeps the paragraph style but loses run-level formatting (bold/italic inside the paragraph).
    # However, the replaced text is mostly plain text descriptions, not titles, so it might be acceptable.
    
    # Let's save.
    doc.save(output_path)
    print(f"Saved modified document to {output_path}. Total modifications: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python modify_manuscript.py <input> <output>")
    else:
        modify_manuscript(sys.argv[1], sys.argv[2])