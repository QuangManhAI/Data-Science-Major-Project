#!/usr/bin/env python3
"""export_report_to_pdf.py

Chuyển đổi toàn diện tệp báo cáo Markdown
`docs/reports/comprehensive_data_metrics_and_inference_audit_report.md`
thành ấn phẩm PDF xuất bản chất lượng cao (A4 Portrait, đầy đủ hình ảnh,
công thức toán học MathML, bảng số liệu, biểu đồ và 20 ảnh suy luận thực tế).
"""

from __future__ import annotations

import re
from pathlib import Path
import markdown
import weasyprint
from latex2mathml.converter import convert as latex2mathml_convert

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "docs/reports/comprehensive_data_metrics_and_inference_audit_report.md"
PDF_PATH = REPO_ROOT / "docs/reports/comprehensive_data_metrics_and_inference_audit_report.pdf"


def convert_md_to_pdf() -> None:
    raw_md = MD_PATH.read_text(encoding="utf-8")

    # 1. Chuyển đổi công thức toán Display $$...$$
    def replace_display_math(match):
        code = match.group(1).strip()
        try:
            return f'<div class="math-block">{latex2mathml_convert(code, display="block")}</div>'
        except Exception as e:
            print(f"[Math Warning] Display math fallback: {e}")
            return f'<pre class="math-fallback">{code}</pre>'

    # 2. Chuyển đổi công thức toán Inline $...$
    def replace_inline_math(match):
        code = match.group(1).strip()
        try:
            return latex2mathml_convert(code, display="inline")
        except Exception as e:
            return f'<code class="math-fallback">{code}</code>'

    processed_md = re.sub(r'\$\$(.*?)\$\$', replace_display_math, raw_md, flags=re.DOTALL)
    processed_md = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', replace_inline_math, processed_md)

    # 3. Xử lý biểu đồ Mermaid thành các khối HTML/CSS sắc nét
    def replace_mermaid(match):
        content = match.group(1).strip()
        if content.startswith("pie"):
            return """
    <div class="callout callout-info" style="margin: 15px 0; padding: 12px 16px; background: #f0f7ff; border-left: 4px solid #0066cc; border-radius: 4px;">
        <strong>📊 Phân Bổ Nhãn Thực Thể Lâm Sàng:</strong> Dáng đứng tiêu chuẩn (3,150f - 26.4%), Bẫy đạo cụ rỗng (2,550f - 21.3%), Ngồi ghế băng (1,800f - 15.1%), Nhân viên y tế (1,500f - 12.6%), Đám đông (1,350f - 11.3%), Xe lăn (900f - 7.5%), Gù lưng (600f - 5.0%), Giường bệnh (150f - 1.3%).
    </div>
    """
        elif "flowchart" in content:
            return """
    <div class="pipeline-container">
        <div class="pipe-step"><span class="step-num">1</span> <strong>Lưới Điểm 3D</strong><br><small>Cylindrical Grid</small></div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-step"><span class="step-num">2</span> <strong>Chiếu Camera</strong><br><small>3D unproject 2D</small></div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-step"><span class="step-num">3</span> <strong>Bao Hộp 2D</strong><br><small>Min/Max Bounds</small></div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-step"><span class="step-num">4</span> <strong>Raycasting</strong><br><small>Đo che khuất V_vis</small></div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-step"><span class="step-num">5</span> <strong>Xuất Nhãn</strong><br><small>YOLO .txt & JSONL</small></div>
    </div>
    """
        return f"<pre>{content}</pre>"

    processed_md = re.sub(r'```mermaid\s*\n(.*?)\n```', replace_mermaid, processed_md, flags=re.DOTALL)

    # 4. Xử lý GitHub Alerts
    alert_types = {
        "NOTE": ("callout-info", "ℹ️ LƯU Ý KỸ THUẬT"),
        "IMPORTANT": ("callout-danger", "🚨 ĐÁNH GIÁ TRỌNG YẾU"),
        "WARNING": ("callout-warning", "⚠️ CẢNH BÁO AN TOÀN"),
        "CAUTION": ("callout-caution", "🛑 RỦI RO CẤP 1 (ISO 3691-4)"),
        "TIP": ("callout-success", "💡 KHUYẾN NGHỊ THỰC NGHIỆM")
    }
    for a_key, (css_cls, title) in alert_types.items():
        pattern = r'>\s*\[!' + a_key + r'\]\s*\n((?:>.*(?:\n|$))*)'
        def make_alert(m):
            lines = [l.lstrip('>').strip() for l in m.group(1).splitlines() if l.strip()]
            body = " ".join(lines)
            return f'<div class="callout {css_cls}"><div class="callout-title">{title}</div><div class="callout-body">{body}</div></div>\n\n'
        processed_md = re.sub(pattern, make_alert, processed_md)

    # 5. Biên dịch Markdown sang HTML
    html_body = markdown.markdown(
        processed_md,
        extensions=['tables', 'fenced_code', 'attr_list', 'def_list', 'nl2br']
    )

    # 6. Đóng gói các thẻ case card cho 20 ảnh suy luận
    pattern = r'(<h4>(\d+\.\s*Ca\s*[CN]\d+.*?)\s*</h4>\s*<p><img\s+(alt=".*?")?\s*src="(.*?)"\s*(?:alt=".*?")?.*?/?>\s*</p>\s*<blockquote>\s*<p>(.*?)</p>\s*</blockquote>)'
    def card_repl(m):
        title = m.group(2)
        img_src = m.group(4)
        analysis = m.group(5)
        cls = "case-card case-success" if "SUCCESS" in img_src or "BẮT ĐƯỢC" in title else "case-card case-failed"
        return f"""
    <div class="{cls}">
        <div class="case-header">{title}</div>
        <div class="case-img-box"><img src="{img_src}" class="case-img"></div>
        <div class="case-analysis-box">{analysis}</div>
    </div>
    """
    html_body = re.sub(pattern, card_repl, html_body, flags=re.DOTALL)

    # 7. CSS Hoàn Chỉnh Chuẩn In Ấn Xuất Bản A4
    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo Cáo Toàn Diện Thị Giác AI Robot Bệnh Viện</title>
<style>
    @page {{
        size: A4 portrait;
        margin: 16mm 14mm 16mm 14mm;
        @top-center {{
            content: "DỰ ÁN ROBOT BỆNH VIỆN — BÁO CÁO GIÁM ĐỊNH THỊ GIÁC TOÀN DIỆN & BỘ 20 ẢNH INFERENCE";
            font-family: "Noto Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            color: #7f8c8d;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 4px;
        }}
        @bottom-left {{
            content: "Data Science Major Project | Hospital Autonomous Assistant";
            font-family: "Noto Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            color: #7f8c8d;
        }}
        @bottom-right {{
            content: "Trang " counter(page) " / " counter(pages);
            font-family: "Noto Sans", "Liberation Sans", sans-serif;
            font-size: 7.5pt;
            font-weight: bold;
            color: #2c3e50;
        }}
    }}

    body {{
        font-family: "Noto Sans", "Liberation Sans", "DejaVu Sans", sans-serif;
        font-size: 9.5pt;
        line-height: 1.5;
        color: #2c3e50;
        margin: 0;
        padding: 0;
    }}

    h1 {{
        font-size: 16pt;
        color: #1a365d;
        border-bottom: 2.5px solid #2b6cb0;
        padding-bottom: 8px;
        margin-top: 0;
        margin-bottom: 8px;
        line-height: 1.3;
    }}

    h2 {{
        font-size: 12.5pt;
        color: #2b6cb0;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 18px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }}

    h3 {{
        font-size: 10.5pt;
        color: #2d3748;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}

    h4 {{
        font-size: 9.5pt;
        color: #4a5568;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}

    p {{
        margin-top: 0;
        margin-bottom: 8px;
        text-align: justify;
    }}

    ul, ol {{
        margin-top: 0;
        margin-bottom: 8px;
        padding-left: 20px;
    }}

    li {{
        margin-bottom: 3px;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        margin-bottom: 14px;
        font-size: 8pt;
        page-break-inside: auto;
    }}

    tr {{
        page-break-inside: avoid;
        page-break-after: auto;
    }}

    th {{
        background-color: #2b6cb0;
        color: #ffffff;
        font-weight: bold;
        text-align: left;
        padding: 6px 6px;
        border: 1px solid #cbd5e0;
    }}

    td {{
        padding: 5px 6px;
        border: 1px solid #e2e8f0;
        vertical-align: middle;
    }}

    tr:nth-child(even) td {{
        background-color: #f7fafc;
    }}

    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 6px auto;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }}

    .callout {{
        padding: 8px 12px;
        margin: 10px 0;
        border-left: 4px solid #3182ce;
        border-radius: 0 4px 4px 0;
        background-color: #ebf8ff;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}

    .callout-title {{
        font-weight: bold;
        margin-bottom: 4px;
        color: #2b6cb0;
    }}

    .callout-danger {{
        border-left-color: #e53e3e;
        background-color: #fff5f5;
    }}
    .callout-danger .callout-title {{
        color: #c53030;
    }}

    .callout-warning {{
        border-left-color: #dd6b20;
        background-color: #fffaf0;
    }}
    .callout-warning .callout-title {{
        color: #c05621;
    }}

    .callout-caution {{
        border-left-color: #9b2c2c;
        background-color: #fff5f5;
    }}
    .callout-caution .callout-title {{
        color: #742a2a;
    }}

    .callout-success {{
        border-left-color: #38a169;
        background-color: #f0fff4;
    }}
    .callout-success .callout-title {{
        color: #276749;
    }}

    .pipeline-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #edf2f7;
        padding: 10px 8px;
        border-radius: 6px;
        margin: 10px 0;
        page-break-inside: avoid;
    }}

    .pipe-step {{
        flex: 1;
        background: #ffffff;
        border: 1px solid #cbd5e0;
        border-radius: 4px;
        padding: 6px 4px;
        text-align: center;
        font-size: 7.5pt;
        line-height: 1.25;
    }}

    .step-num {{
        display: inline-block;
        background: #2b6cb0;
        color: #ffffff;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        font-size: 7pt;
        font-weight: bold;
        margin-bottom: 2px;
    }}

    .pipe-arrow {{
        color: #4a5568;
        font-weight: bold;
        padding: 0 4px;
        font-size: 10pt;
    }}

    .case-card {{
        border: 1px solid #e2e8f0;
        border-radius: 5px;
        background: #ffffff;
        padding: 8px 10px;
        margin-bottom: 12px;
        page-break-inside: avoid;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}

    .case-success {{
        border-left: 4px solid #38a169;
    }}

    .case-failed {{
        border-left: 4px solid #e53e3e;
    }}

    .case-header {{
        font-weight: bold;
        font-size: 9pt;
        color: #2d3748;
        margin-bottom: 4px;
    }}

    .case-img {{
        max-width: 100%;
        max-height: 220px;
        object-fit: contain;
        margin: 4px auto;
        display: block;
    }}

    .case-analysis-box {{
        background: #f7fafc;
        border-radius: 3px;
        padding: 4px 8px;
        font-size: 8pt;
        color: #4a5568;
        line-height: 1.35;
        margin-top: 4px;
    }}

    .math-block {{
        text-align: center;
        margin: 8px 0;
        font-size: 10pt;
    }}

    code {{
        background: #edf2f7;
        padding: 1px 4px;
        border-radius: 3px;
        font-family: "Liberation Mono", monospace;
        font-size: 8pt;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    print("Rendering PDF with WeasyPrint...")
    html_obj = weasyprint.HTML(string=full_html, base_url=str(MD_PATH.parent))
    html_obj.write_pdf(str(PDF_PATH))
    print(f"✅ Đã tạo PDF thành công: {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    convert_md_to_pdf()
