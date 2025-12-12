from datetime import datetime
from fpdf import FPDF


def safe_text(text) -> str:
    """FPDF uchun matnni latin-1 formatga moslab tozalaydi."""
    if not isinstance(text, str):
        text = str(text)
    # eng ko'p muammo beradigan uzun chiziqlarni oddiy '-' bilan almashtiramiz
    text = (
        text.replace("—", "-")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def wrap_text(text: str, max_chars: int = 90):
    """
    Simple word-wrapping helper: splits long lines into multiple shorter lines.
    """
    words = str(text).split()
    lines = []
    current = []

    for w in words:
        test_line = " ".join(current + [w])
        if len(test_line) <= max_chars:
            current.append(w)
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]

    if current:
        lines.append(" ".join(current))

    return lines

    # latin-1 ga sig‘maydigan boshqa belgilarni tashlab yuborish
    return text.encode("latin-1", errors="ignore").decode("latin-1")

class ReportPDF(FPDF):
    """Custom PDF class with header and footer for ML reports."""

    def header(self):
        # Title
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, safe_text("ML Model Report"), ln=True, align="C")
        self.ln(2)

        # Separator line
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, 20, 200, 20)
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100)
        # Page number
        self.cell(0, 10, safe_text(f"Page {self.page_no()}"), align="C")

def generate_pdf_report(
    project_name: str,
    task_type: str,
    model_name: str,
    metrics: dict,
    target_col: str,
    feature_cols,
    interpretation_text=None,
    data_filename: str | None = None,  # 🔹 yangi parametr
) -> bytes:

    """
    Generate a simple ML model report as PDF.
    """
    # Safety checks
    if feature_cols is None:
        feature_cols = []
    if interpretation_text is None:
        interpretation_text = []

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Title block ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, safe_text(project_name), ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 8, safe_text(f"Source file: {data_filename}"), ln=True)
    pdf.cell(0, 8, safe_text(f"Generated: {now_str}"), ln=True)
    pdf.cell(0, 8, safe_text(f"Task Type: {task_type}"), ln=True)
    pdf.cell(0, 8, safe_text(f"Model: {model_name}"), ln=True)
    pdf.cell(0, 8, safe_text(f"Target: {target_col}"), ln=True)
    pdf.ln(4)

    # --- Features ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe_text("Features used:"), ln=True)
    pdf.set_font("Helvetica", "", 10)

    if feature_cols:
        features_line = ", ".join(map(str, feature_cols))
        for line in wrap_text(features_line, max_chars=90):
            pdf.multi_cell(0, 5, safe_text(line))
    else:
        pdf.multi_cell(0, 5, safe_text("No features provided."))
    pdf.ln(3)

    # --- Metrics ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe_text("Metrics:"), ln=True)
    pdf.set_font("Helvetica", "", 11)

    for key, value in metrics.items():
        # Classification report will be handled separately
        if key == "Classification Report":
            continue
        # Convert metrics to string safely
        pdf.cell(0, 7, safe_text(f"{key}: {value}"), ln=True)
    pdf.ln(3)

    # --- Classification report (if present) ---
    if "Classification Report" in metrics:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, safe_text("Classification Report:"), ln=True)
        pdf.set_font("Courier", "", 8)

        report_text = str(metrics["Classification Report"])
        for line in report_text.split("\n"):
            pdf.multi_cell(0, 4, safe_text(line))
        pdf.ln(3)

    # --- Interpretation (for regression) ---
    if interpretation_text:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, safe_text("Interpretation:"), ln=True)
        pdf.set_font("Helvetica", "", 10)

        for msg in interpretation_text:
            # Use ASCII-friendly bullet ("-") to avoid encoding issues
            for line in wrap_text("- " + str(msg), max_chars=100):
                pdf.multi_cell(0, 5, safe_text(line))
        pdf.ln(3)

    # Return PDF as bytes (handle both fpdf and fpdf2 behaviour)
    result = pdf.output(dest="S")
    if isinstance(result, str):
        # Old FPDF returns a latin-1 string
        pdf_bytes = result.encode("latin-1", errors="ignore")
    else:
        # fpdf2 returns bytes
        pdf_bytes = result

    return pdf_bytes
