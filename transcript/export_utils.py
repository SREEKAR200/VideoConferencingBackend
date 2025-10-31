import json
from fpdf import FPDF

def export_transcript_txt(formatted_transcript, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted_transcript)

def export_transcript_json(transcript, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

def export_transcript_pdf(formatted_transcript, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in formatted_transcript.split('\n'):
        pdf.cell(0, 10, line, ln=True)
    pdf.output(output_path)
