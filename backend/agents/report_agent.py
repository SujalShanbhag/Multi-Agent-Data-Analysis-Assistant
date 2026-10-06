"""
PDF Report Generation Agent.
Creates an executive-grade, publication-quality PDF report
using ReportLab, with the Gemini AI Analysed Conclusion placed at the end.
All table cells use wrapped Paragraph flowables with explicit column widths
and auto-wrap to guarantee text NEVER bleeds outside the table boundaries.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import html

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_report(
    dataset_name: str,
    profile: dict,
    cleaning: dict,
    eda: dict,
    statistics: dict,
    ml: dict,
    validation: dict,
    insights: list[str],
    gemini_conclusion: dict,
) -> str:
    """
    Generates a structured, multi-page PDF analysis report.
    Guarantees all text inside tables wraps cleanly without exceeding margins.
    """
    import os
    import tempfile

    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        report_dir = Path(tempfile.gettempdir()) / "data_analysis_assistant" / "reports"
    else:
        report_dir = PROJECT_ROOT / "reports"

    report_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_dir / "data_analysis_report.pdf"

    # Usable width: 612 (letter width) - 72 (36 left + 36 right margins) = 540 pt
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f3d6c"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4b6584"),
    )
    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f3d6c"),
        spaceBefore=12,
        spaceAfter=5,
    )
    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Heading3"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1b5088"),
        spaceBefore=7,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e272e"),
    )
    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e272e"),
        leftIndent=12,
        firstLineIndent=-8,
    )
    conclusion_style = ParagraphStyle(
        "DocConclusion",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=13,
        textColor=colors.HexColor("#1e272e"),
    )

    # Table cell paragraph styles with automatic word-wrap
    th_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        fontName="Helvetica-Bold",
        alignment=0,
    )
    th_center = ParagraphStyle(
        "TableHeaderCenter",
        parent=th_style,
        alignment=1,
    )
    td_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
        fontName="Helvetica",
    )
    td_bold = ParagraphStyle(
        "TableCellBold",
        parent=td_style,
        fontName="Helvetica-Bold",
    )
    td_center = ParagraphStyle(
        "TableCellCenter",
        parent=td_style,
        alignment=1,
    )
    td_metric = ParagraphStyle(
        "TableCellMetric",
        parent=td_style,
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#1e3a8a"),
        fontName="Helvetica",
    )

    story = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    story.append(Paragraph("Multi-Agent Data Science & Analytics Report", title_style))
    story.append(Paragraph(f"Dataset: <b>{html.escape(dataset_name)}</b> &nbsp;|&nbsp; Generated: <b>{timestamp}</b>", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1683ff"), spaceAfter=10))

    # 1. Executive Insights
    story.append(Paragraph("1. Executive Insights", h2_style))
    for ins in insights:
        story.append(Paragraph(f"• {html.escape(ins)}", bullet_style))
        story.append(Spacer(1, 2))
    story.append(Spacer(1, 6))

    # 2. Dataset Overview Table (Total 540 pt width = 110 + 160 + 110 + 160)
    story.append(Paragraph("2. Dataset Profile Overview", h2_style))
    profile_data = [
        [
            Paragraph("Total Rows", td_bold),
            Paragraph(f"{profile.get('rows', 0):,}", td_style),
            Paragraph("Total Columns", td_bold),
            Paragraph(f"{profile.get('columns', 0)}", td_style),
        ],
        [
            Paragraph("Missing Values", td_bold),
            Paragraph(f"{profile.get('missing_total', 0):,}", td_style),
            Paragraph("Duplicate Rows", td_bold),
            Paragraph(f"{profile.get('duplicates', 0):,}", td_style),
        ],
        [
            Paragraph("Memory Usage", td_bold),
            Paragraph(f"{profile.get('memory_mb', 0)} MB", td_style),
            Paragraph("Target Variable", td_bold),
            Paragraph(html.escape(str(ml.get('target_column', 'Auto-Selected'))), td_style),
        ],
    ]
    t_prof = Table(profile_data, colWidths=[110, 160, 110, 160])
    t_prof.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(t_prof)
    story.append(Spacer(1, 8))

    # 3. Data Cleaning & Preprocessing Summary
    story.append(Paragraph("3. Data Cleaning & Preprocessing", h2_style))
    actions = cleaning.get("actions", [])
    if actions:
        for a in actions:
            col_txt = html.escape(str(a.get("column", "Rows")))
            issue_txt = html.escape(str(a.get("issue", "")))
            rec_txt = html.escape(str(a.get("recommended", "")))
            story.append(Paragraph(f"• <b>{col_txt}</b>: {issue_txt} &rarr; Applied <i>{rec_txt}</i>", bullet_style))
    else:
        story.append(Paragraph("• No missing values or duplicates detected; dataset passed clean checks without alteration.", bullet_style))
    story.append(Spacer(1, 8))

    # 4. Exploratory Data Analysis & Statistical Findings
    story.append(Paragraph("4. Exploratory Data Analysis & Statistics", h2_style))
    for f in eda.get("findings", []):
        story.append(Paragraph(f"• {html.escape(f)}", bullet_style))
    story.append(Spacer(1, 8))

    # 5. Machine Learning Model Leaderboard
    models_list = ml.get("models", [])
    story.append(Paragraph(f"5. Machine Learning Leaderboard ({len(models_list)} Models Evaluated)", h2_style))
    story.append(Paragraph(
        f"<b>Task:</b> {html.escape(str(ml.get('task', 'N/A')))} &nbsp;|&nbsp; "
        f"<b>Engine:</b> {html.escape(str(ml.get('engine', 'Pure NumPy Multi-Model Suite')))} &nbsp;|&nbsp; "
        f"<b>Top Model:</b> {html.escape(str(ml.get('best_model', 'N/A')))} (Score: {html.escape(str(ml.get('best_score', '')))})",
        body_style,
    ))
    story.append(Spacer(1, 6))

    # Table layout: total width = 34 + 130 + 176 + 200 = 540 pt (matches exactly page margins)
    ml_table_data = [[
        Paragraph("Rank", th_center),
        Paragraph("Model Name", th_style),
        Paragraph("Architecture / Family", th_style),
        Paragraph("Evaluation Metrics", th_style),
    ]]

    for idx, m in enumerate(models_list, 1):
        m_name = html.escape(str(m.get("name", "")))
        m_desc = html.escape(str(m.get("description", "")))
        metrics_pairs = [f"<b>{html.escape(str(k))}:</b> {html.escape(str(v))}" for k, v in m.get("metrics", {}).items()]
        m_metrics_html = " &bull; ".join(metrics_pairs)

        rank_style = td_center if idx > 1 else ParagraphStyle("Rank1", parent=td_center, fontName="Helvetica-Bold", textColor=colors.HexColor("#0284c7"))
        name_style = td_style if idx > 1 else td_bold

        ml_table_data.append([
            Paragraph(f"#{idx}", rank_style),
            Paragraph(m_name, name_style),
            Paragraph(m_desc, td_style),
            Paragraph(m_metrics_html, td_metric),
        ])

    t_ml = Table(ml_table_data, colWidths=[34, 130, 176, 200])
    table_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3d6c")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
    ]
    if len(models_list) >= 1:
        # Highlight top-ranked model row
        table_styles.append(("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e0f2fe")))
    t_ml.setStyle(TableStyle(table_styles))
    story.append(t_ml)
    story.append(Spacer(1, 12))

    # 6. Analysed Conclusion at the End (Powered by Gemini AI)
    conclusion_text = gemini_conclusion.get("conclusion", "")
    conclusion_model = gemini_conclusion.get("model_used", "Automated AI Synthesis")
    story.append(KeepTogether([
        Paragraph("6. Analysed Conclusion & Strategic Recommendations (Gemini AI)", h2_style),
        Paragraph(f"<i>Synthesized using {html.escape(gemini_conclusion.get('engine', 'Google Gemini AI'))} ({html.escape(conclusion_model)})</i>", subtitle_style),
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#2563eb"), spaceAfter=8),
    ]))

    # Format markdown paragraphs into PDF flowables with safe auto-wrapping
    for block in conclusion_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("### "):
            story.append(Paragraph(html.escape(block.removeprefix("### ")), h3_style))
        elif block.startswith("## "):
            story.append(Paragraph(html.escape(block.removeprefix("## ")), h3_style))
        elif block.startswith("1. ") or block.startswith("2. ") or block.startswith("3. ") or block.startswith("4. ") or block.startswith("5. "):
            story.append(Paragraph(html.escape(block), bullet_style))
            story.append(Spacer(1, 2))
        else:
            story.append(Paragraph(html.escape(block), conclusion_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    return str(pdf_path)
