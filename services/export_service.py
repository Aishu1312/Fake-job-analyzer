import pandas as pd
import io
from fpdf import FPDF

def export_csv(results):
    """
    Export batch results to CSV bytes.
    """
    df = _results_to_dataframe(results)
    return df.to_csv(index=False).encode('utf-8')

def export_excel(results):
    """
    Export batch results to Excel bytes.
    """
    df = _results_to_dataframe(results)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis Results')
    return output.getvalue()

def export_pdf(results, summary_stats):
    """
    Export batch results to a PDF report.
    Returns PDF bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Fake Job Analyzer - Batch Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Total Files Analyzed: {summary_stats.get('total', 0)}", ln=True)
    pdf.cell(200, 10, txt=f"Legitimate Jobs: {summary_stats.get('legit', 0)}", ln=True)
    pdf.cell(200, 10, txt=f"Suspicious Jobs: {summary_stats.get('suspicious', 0)}", ln=True)
    pdf.cell(200, 10, txt=f"Fake Jobs: {summary_stats.get('fake', 0)}", ln=True)
    
    avg_score = summary_stats.get('avg_score', 0)
    pdf.cell(200, 10, txt=f"Average Fraud Score: {avg_score:.1f}/100", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    for r in results:
        if r["Status"] != "Success":
            continue
            
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(200, 10, txt=f"File: {r['Filename']}", ln=True)
        pdf.set_font("Arial", size=10)
        
        verdict = r['Verdict']
        score = r['Fraud Score']
        pdf.cell(200, 8, txt=f"Verdict: {verdict} | Risk Score: {score}/100", ln=True)
        
        reasons = r.get("Risk Stats", {}).get("Reasons", [])
        if reasons:
            pdf.cell(200, 8, txt="Suspicious Indicators:", ln=True)
            for reason in reasons:
                # Handle long reasons
                pdf.multi_cell(190, 6, txt=f"- {reason}")
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin1')

def _results_to_dataframe(results):
    """
    Helper to convert results dict list to a clean Pandas DataFrame
    """
    data = []
    for r in results:
        if r["Status"] == "Success":
            stats = r.get("Risk Stats", {})
            data.append({
                "Filename": r["Filename"],
                "Verdict": r["Verdict"],
                "Risk Level": r["Risk Level"],
                "Fraud Score": r["Fraud Score"],
                "AI Confidence (%)": r["AI Confidence"],
                "Urgency Score": stats.get("Urgency Manipulation", 0),
                "Salary Score": stats.get("Salary Manipulation", 0),
                "Grammar Score": stats.get("Grammar Quality", 100),
                "Missing Info Score": stats.get("Missing Information", 0),
            })
        else:
            data.append({
                "Filename": r["Filename"],
                "Verdict": "Error",
                "Risk Level": "N/A",
                "Fraud Score": "N/A",
                "AI Confidence (%)": "N/A"
            })
    return pd.DataFrame(data)
