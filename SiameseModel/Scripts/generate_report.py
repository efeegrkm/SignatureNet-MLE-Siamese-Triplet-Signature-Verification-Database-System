from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re

# Dosya adı
filename = "Signature_Verification_System_Report.pdf"

# Döküman oluştur
doc = SimpleDocTemplate(filename, pagesize=A4,
                        rightMargin=50, leftMargin=50,
                        topMargin=50, bottomMargin=50)

# Stilleri tanımla
styles = getSampleStyleSheet()

# Başlık Stili
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Title'],
    fontSize=20,
    spaceAfter=20,
    textColor=colors.darkblue,
    alignment=TA_CENTER
)

# H1 Stili
h1_style = ParagraphStyle(
    'CustomH1',
    parent=styles['Heading1'],
    fontSize=16,
    spaceBefore=20,
    spaceAfter=10,
    textColor=colors.darkblue,
    keepWithNext=True
)

# H2 Stili
h2_style = ParagraphStyle(
    'CustomH2',
    parent=styles['Heading2'],
    fontSize=14,
    spaceBefore=15,
    spaceAfter=8,
    textColor=colors.black,
    keepWithNext=True
)

# H3 Stili
h3_style = ParagraphStyle(
    'CustomH3',
    parent=styles['Heading3'],
    fontSize=12,
    spaceBefore=10,
    spaceAfter=5,
    textColor=colors.black,
    fontName='Helvetica-Bold'
)

# Normal Metin Stili
normal_style = ParagraphStyle(
    'CustomNormal',
    parent=styles['Normal'],
    fontSize=11,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=6
)

# Kod/ASCII Stili (Monospace)
code_style = ParagraphStyle(
    'CodeStyle',
    parent=styles['Code'],
    fontName='Courier',
    fontSize=9,
    leading=10,
    backColor=colors.whitesmoke,
    borderPadding=5,
    spaceAfter=10
)

story = []

# --- İçerik Oluşturma ---

# Başlık
story.append(Paragraph("Signature Verification System", title_style))
story.append(Paragraph("Model Development & Database Design Report", h2_style))
story.append(Spacer(1, 20))

# 1. Overview
story.append(Paragraph("1. Overview", h1_style))
overview_text = """This project implements a robust signature verification system based on a Siamese Neural Network (Contrastive Loss). 
It features a preprocessing pipeline for normalizing signature images, evaluation modules, and a database-backed identity system.
The system achieves <b>92% test accuracy</b> and performs strongly on real-world handwritten signatures."""
story.append(Paragraph(overview_text, normal_style))

# 2. Model Development Journey
story.append(Paragraph("2. Model Development Journey", h1_style))
story.append(Paragraph("Below is a detailed chronological summary of issues encountered and resolutions.", normal_style))

# 2.1
story.append(Paragraph("2.1 Issue: Incorrect Preprocessing (Small-Patch Overfitting)", h2_style))
story.append(Paragraph("<b>Problem:</b> Early scaling bugs caused images to be cropped into tiny fragments, leading to rapid overfitting and misleadingly high validation accuracy.", normal_style))
story.append(Paragraph("<b>Solution:</b> Preprocessing was rewritten. All signatures are now placed on a 400x400 canvas with corrected scaling.", normal_style))
story.append(Paragraph("<b>Result:</b> Realistic generalization appeared; validation accuracy dropped to a truthful 66%.", normal_style))

# 2.2
story.append(Paragraph("2.2 Issue: Underfitting at 50 Epochs", h2_style))
story.append(Paragraph("<b>Problem:</b> Loss decreased slowly and accuracy plateaued at ~66%.", normal_style))
story.append(Paragraph("<b>Solution:</b> Increased epochs from 50 to 90 and batch size from 16 to 32.", normal_style))
story.append(Paragraph("<b>Result:</b> Accuracy improved significantly from 66% to 92%.", normal_style))

# 2.3
story.append(Paragraph("2.3 Issue: Sensitivity to Background Noise", h2_style))
story.append(Paragraph("<b>Problem:</b> Real signatures failed due to background texture and brightness differences.", normal_style))
story.append(Paragraph("<b>Solution:</b> Added Color Jitter Augmentation (brightness=0.2, contrast=0.2).", normal_style))
story.append(Paragraph("<b>Result:</b> The model became background-invariant and robust.", normal_style))

# 3. Training and Evaluation Plots
story.append(Paragraph("3. Training and Evaluation Plots", h1_style))
plots_info = [
    "<b>Training Loss Curve:</b> Shows stable convergence and no overfitting.",
    "<b>Positive/Negative Distance Curves:</b> A clean growing gap indicates strong embedding separation.",
    "<b>Test Distance Distribution:</b> Genuine and forgery clusters are nearly perfectly separated.",
    "<b>ROC Curve:</b> AUC approx 0.9936",
    "<b>Precision-Recall Curve:</b> AUC approx 0.9938"
]
for p in plots_info:
    story.append(Paragraph(f"• {p}", normal_style))

# 4. Threshold Determination
story.append(Paragraph("4. Verification Threshold Determination", h1_style))
story.append(Paragraph("The optimal threshold is selected by maximizing TPR, TNR, and F1-score over the full distance range.", normal_style))
story.append(Paragraph("<b>Optimal Threshold = 1.016</b>", normal_style))

# Tablo Verisi
data = [
    ['Metric', 'Value'],
    ['True Positives', '223'],
    ['True Negatives', '219'],
    ['False Positives', '29'],
    ['False Negatives', '8'],
    ['Accuracy', '92.28%']
]

t = Table(data, colWidths=[200, 100])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
story.append(Spacer(1, 10))
story.append(t)

# 5. Code Architecture
story.append(Paragraph("5. Code Architecture Overview", h1_style))

code_sections = [
    ("model.py", "SignatureNet CNN: Accepts 400x400 input, outputs 128-dim normalized embedding."),
    ("preprocess.py", "Image Standardization: Converts input to clean canvas, removes noise, applies binarization."),
    ("siamese_dataset.py", "Pair Builder: Generates positive/negative pairs with augmentations (rotation, translation, jitter)."),
    ("siamese_train.py", "Training Engine: Handles Contrastive Loss optimization, logging, and model saving."),
    ("siamese_evaluate.py", "Evaluation: Computes distances, finds optimal threshold, generates ROC/PR curves.")
]

for title, desc in code_sections:
    story.append(Paragraph(f"<b>{title}:</b> {desc}", normal_style))

story.append(PageBreak())

# 6. Database Design
story.append(Paragraph("6. Database Design", h1_style))
story.append(Paragraph("The system manages users, raw signature images, embeddings, and mean embeddings.", normal_style))

db_diagram = """
+-------------------+
|      USER         |
+-------------------+
| NO (PK)           |
| Name              |
| Surname           |
| MeanEmbedding     |
+---------+---------+
          |
          | 1-to-many
          |
+---------------------------+
|       SIGNATURE           |
+---------------------------+
| SigID (PK)                |
| UserNO (FK -> USER.NO)    |
| ImagePath                 |
| EmbeddingVector           |
+---------------------------+
"""
story.append(Spacer(1, 10))
story.append(Preformatted(db_diagram, code_style))

# 7. Supported Queries
story.append(Paragraph("7. Supported Database Queries", h1_style))

queries = [
    "<b>Verify if PNG belongs to user:</b> Input (NO, PNG) -> Preprocess -> Compare with MeanEmbedding -> Return True/False.",
    "<b>Find user's NO by name:</b> Input (Name, Surname) -> Return NO.",
    "<b>Identify owner of PNG:</b> Input (PNG) -> Compare with all users -> Return closest match.",
    "<b>Compare two PNG signatures:</b> Input (Image A, Image B) -> Compute Distance -> Return Match/No Match."
]
for q in queries:
    story.append(Paragraph(f"{queries.index(q)+1}. {q}", normal_style))

# 8. Final Remarks
story.append(Paragraph("8. Final Remarks", h1_style))
final_text = """After correcting preprocessing, expanding training, and improving augmentation, the system now achieves 
<b>92%+ accuracy</b> and generalizes well to real signatures. The framework is production-ready and supports database-based identity verification."""
story.append(Paragraph(final_text, normal_style))

# PDF Oluştur
doc.build(story)