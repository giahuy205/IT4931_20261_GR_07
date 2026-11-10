# -*- coding: utf-8 -*-
"""
Export Markdown report to DOCX (Word) and PDF with perfect Vietnamese fonts (Times New Roman).
"""

import os
import re
import subprocess
from pathlib import Path

import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls


def set_cell_background(cell, fill_hex):
    """Đặt màu nền cho ô trong bảng docx"""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    """Đặt lề trong cho ô (dxf)"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color="D1D5DB", sz="4", val="single"):
    """Đặt viền mỏng tinh tế cho bảng docx"""
    tblPr = table._tbl.tblPr
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(tblBorders)


def add_formatted_text(p, text):
    """Phân tích các thẻ Markdown đơn giản như **bold**, *italic*, `code`, math"""
    # Xử lý bold và italic
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\$.*?\$)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = p.add_run(token[2:-2])
            run.bold = True
        elif token.startswith('*') and token.endswith('*'):
            run = p.add_run(token[1:-1])
            run.italic = True
        elif token.startswith('`') and token.endswith('`'):
            run = p.add_run(token[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(10.5)
            run.font.color.rgb = RGBColor(30, 41, 59)
        elif token.startswith('$') and token.endswith('$'):
            # Math
            math_text = token[1:-1].replace(r'\approx', '≈').replace(r'\ge', '≥').replace(r'\le', '≤').replace(r'\in', '∈').replace(r'\mathbb{R}', 'R').replace(r'\cdot', '·').replace(r'\times', '×').replace(r'\lambda', 'λ').replace(r'\sum', 'Σ').replace(r'\min', 'min')
            run = p.add_run(math_text)
            run.italic = True
            run.font.color.rgb = RGBColor(15, 23, 42)
        else:
            p.add_run(token)


def create_docx(md_path, docx_path):
    doc = docx.Document()

    # 1. Định dạng trang A4 theo chuẩn học thuật HUST
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(3.0)   # Lề trái 3cm đóng gáy
        section.right_margin = Cm(2.0)

    # 2. Đặt font mặc định toàn bộ tài liệu là Times New Roman
    style_normal = doc.styles['Normal']
    font_normal = style_normal.font
    font_normal.name = 'Times New Roman'
    font_normal.size = Pt(12)
    font_normal.color.rgb = RGBColor(30, 41, 59)

    # Đọc nội dung file markdown
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if not table_lines:
            return
        
        # Parse table markdown
        rows_data = []
        for tl in table_lines:
            tl = tl.strip()
            if not tl or re.match(r'^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?$', tl):
                continue
            cells = [c.strip() for c in tl.strip('|').split('|')]
            rows_data.append(cells)
        
        if rows_data:
            num_rows = len(rows_data)
            num_cols = max(len(r) for r in rows_data)
            table = doc.add_table(rows=num_rows, cols=num_cols)
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            set_table_borders(table)

            for r_idx, row in enumerate(rows_data):
                for c_idx, cell_text in enumerate(row):
                    if c_idx < num_cols:
                        cell = table.cell(r_idx, c_idx)
                        set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_before = Pt(0)
                        p.paragraph_format.space_after = Pt(0)
                        p.paragraph_format.line_spacing = 1.15
                        
                        if r_idx == 0:
                            # Header
                            set_cell_background(cell, "F1F5F9")
                            run = p.add_run(cell_text.replace('**', ''))
                            run.bold = True
                            run.font.name = 'Times New Roman'
                            run.font.size = Pt(11)
                            run.font.color.rgb = RGBColor(15, 23, 42)
                        else:
                            # Data cell
                            if r_idx % 2 == 1:
                                set_cell_background(cell, "FFFFFF")
                            else:
                                set_cell_background(cell, "F8FAFC")
                            add_formatted_text(p, cell_text)
                            for r in p.runs:
                                if r.font.name != 'Consolas':
                                    r.font.name = 'Times New Roman'
                                r.font.size = Pt(11)

            doc.add_paragraph().paragraph_format.space_after = Pt(6)

        table_lines = []
        in_table = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n')
        stripped = line.strip()

        # Kiểm tra bảng markdown
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            flush_table()

        if not stripped:
            i += 1
            continue

        # Tiêu đề tài liệu (# )
        if stripped.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(stripped[2:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(16)
            run.font.color.rgb = RGBColor(15, 23, 42)
        
        # Tiêu đề mục lớn (## )
        elif stripped.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[3:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(13.5)
            run.font.color.rgb = RGBColor(3, 105, 161) # Deep Blue HUST

        # Tiêu đề mục con (### )
        elif stripped.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[4:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12.5)
            run.font.color.rgb = RGBColor(15, 23, 42)

        # Tiêu đề cấp 4 (#### )
        elif stripped.startswith('#### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[5:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(51, 65, 85)

        # Đường kẻ ngang (---)
        elif stripped == '---':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            # Thêm viền dưới
            pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="CBD5E1"/></w:pBdr>')
            p._p.get_or_add_pPr().append(pBdr)

        # Công thức hiển thị ($$...$$)
        elif stripped.startswith('$$') and stripped.endswith('$$'):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            math_text = stripped[2:-2].strip().replace(r'\approx', '≈').replace(r'\ge', '≥').replace(r'\le', '≤').replace(r'\in', '∈').replace(r'\mathbb{R}', 'R').replace(r'\cdot', '·').replace(r'\times', '×').replace(r'\lambda', 'λ').replace(r'\sum', 'Σ').replace(r'\min', 'min')
            run = p.add_run(math_text)
            run.italic = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12.5)
            run.font.color.rgb = RGBColor(15, 23, 42)

        # Danh sách gạch đầu dòng (* hoặc -)
        elif stripped.startswith('* ') or stripped.startswith('- '):
            indent_level = (len(line) - len(line.lstrip())) // 2
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.left_indent = Inches(0.25 * (indent_level + 1))
            add_formatted_text(p, stripped[2:])
            for r in p.runs:
                if r.font.name != 'Consolas':
                    r.font.name = 'Times New Roman'

        # Danh sách số (1. 2. 3.)
        elif re.match(r'^\d+\.\s+', stripped):
            num_match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
            indent_level = (len(line) - len(line.lstrip())) // 2
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.left_indent = Inches(0.25 * (indent_level + 1))
            add_formatted_text(p, num_match.group(2))
            for r in p.runs:
                if r.font.name != 'Consolas':
                    r.font.name = 'Times New Roman'

        # Đoạn văn thông thường
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_text(p, stripped)
            for r in p.runs:
                if r.font.name != 'Consolas':
                    r.font.name = 'Times New Roman'

        i += 1

    if in_table:
        flush_table()

    doc.save(docx_path)
    print(f"[✓] Đã tạo thành công file Word (.docx): {docx_path}")


def create_pdf_from_html(md_path, pdf_path):
    """Tạo file PDF chuẩn academic bằng Edge headless"""
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Chuyển đổi Markdown sang HTML đơn giản
    # Xử lý headers
    html_lines = []
    in_table = False
    table_lines = []

    def flush_html_table():
        nonlocal in_table, table_lines
        if not table_lines:
            return
        rows_data = []
        for tl in table_lines:
            tl = tl.strip()
            if not tl or re.match(r'^\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)+\|?$', tl):
                continue
            cells = [c.strip() for c in tl.strip('|').split('|')]
            rows_data.append(cells)
        if rows_data:
            html_lines.append('<table class="data-table"><thead><tr>')
            for c in rows_data[0]:
                html_lines.append(f'<th>{c.replace("**", "")}</th>')
            html_lines.append('</tr></thead><tbody>')
            for r in rows_data[1:]:
                html_lines.append('<tr>')
                for c in r:
                    # format text in cell
                    c_fmt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', c)
                    c_fmt = re.sub(r'\*(.*?)\*', r'<i>\1</i>', c_fmt)
                    html_lines.append(f'<td>{c_fmt}</td>')
                html_lines.append('</tr>')
            html_lines.append('</tbody></table>')
        table_lines = []
        in_table = False

    for line in md_content.splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            flush_html_table()

        if not stripped:
            continue

        if stripped.startswith('# '):
            html_lines.append(f'<h1 class="doc-title">{stripped[2:]}</h1>')
        elif stripped.startswith('## '):
            html_lines.append(f'<h2 class="sec-title">{stripped[3:]}</h2>')
        elif stripped.startswith('### '):
            html_lines.append(f'<h3 class="sub-title">{stripped[4:]}</h3>')
        elif stripped.startswith('#### '):
            html_lines.append(f'<h4 class="sub-sub-title">{stripped[5:]}</h4>')
        elif stripped == '---':
            html_lines.append('<hr class="divider"/>')
        elif stripped.startswith('$$') and stripped.endswith('$$'):
            math = stripped[2:-2].strip().replace(r'\approx', '≈').replace(r'\ge', '≥').replace(r'\le', '≤').replace(r'\in', '∈').replace(r'\mathbb{R}', 'R').replace(r'\cdot', '·').replace(r'\times', '×').replace(r'\lambda', 'λ').replace(r'\sum', 'Σ').replace(r'\min', 'min')
            html_lines.append(f'<div class="math-block"><i>{math}</i></div>')
        elif stripped.startswith('* ') or stripped.startswith('- '):
            item = stripped[2:]
            item = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item)
            item = re.sub(r'\*(.*?)\*', r'<i>\1</i>', item)
            item = re.sub(r'`(.*?)`', r'<code>\1</code>', item)
            html_lines.append(f'<li class="bullet-item">{item}</li>')
        elif re.match(r'^\d+\.\s+', stripped):
            item = re.sub(r'^\d+\.\s+', '', stripped)
            item = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item)
            item = re.sub(r'\*(.*?)\*', r'<i>\1</i>', item)
            item = re.sub(r'`(.*?)`', r'<code>\1</code>', item)
            html_lines.append(f'<li class="num-item">{item}</li>')
        else:
            item = stripped
            item = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', item)
            item = re.sub(r'\*(.*?)\*', r'<i>\1</i>', item)
            item = re.sub(r'`(.*?)`', r'<code>\1</code>', item)
            html_lines.append(f'<p class="para">{item}</p>')

    if in_table:
        flush_html_table()

    html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo Tiến độ - IT4931 Nhóm 7</title>
<style>
  @page {{
    size: A4;
    margin: 20mm 20mm 20mm 25mm;
    @bottom-right {{
      content: counter(page);
      font-family: 'Times New Roman', serif;
      font-size: 10pt;
    }}
  }}
  body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.35;
    color: #1e293b;
    background: #ffffff;
    margin: 0;
    padding: 0;
  }}
  .doc-title {{
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    color: #0f172a;
    margin-top: 10px;
    margin-bottom: 8px;
    text-transform: uppercase;
  }}
  .sec-title {{
    font-size: 13.5pt;
    font-weight: bold;
    color: #0369a1;
    margin-top: 18px;
    margin-bottom: 8px;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 3px;
    page-break-after: avoid;
  }}
  .sub-title {{
    font-size: 12.5pt;
    font-weight: bold;
    color: #0f172a;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}
  .sub-sub-title {{
    font-size: 11.5pt;
    font-weight: bold;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}
  .para {{
    margin: 4px 0;
    text-align: justify;
  }}
  .bullet-item, .num-item {{
    margin: 3px 0 3px 18px;
    text-align: justify;
  }}
  .divider {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 12px 0;
  }}
  .math-block {{
    text-align: center;
    font-size: 12pt;
    margin: 8px 0;
    padding: 6px;
    background: #f8fafc;
    border-radius: 4px;
  }}
  code {{
    font-family: Consolas, monospace;
    font-size: 10.5pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 11pt;
    page-break-inside: avoid;
  }}
  .data-table th {{
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    font-weight: bold;
    text-align: left;
    color: #0f172a;
  }}
  .data-table td {{
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    color: #1e293b;
  }}
  .data-table tr:nth-child(even) {{
    background-color: #f8fafc;
  }}
</style>
</head>
<body>
{"".join(html_lines)}
</body>
</html>
"""
    temp_html = os.path.join(os.path.dirname(pdf_path), "report_temp_print.html")
    temp_pdf_out = os.path.join(os.path.dirname(pdf_path), "report_temp_print.pdf")
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(html_template)

    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(edge_exe):
        user_data_dir = os.path.join(os.path.dirname(pdf_path), ".edge_temp_profile")
        cmd = [
            edge_exe,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-data-dir={user_data_dir}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={temp_pdf_out}",
            temp_html
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as e:
            print(f"[!] Lỗi chạy Edge: {e}")
        
        if os.path.exists(temp_html):
            os.remove(temp_html)
        if os.path.exists(user_data_dir):
            import shutil
            try:
                shutil.rmtree(user_data_dir, ignore_errors=True)
            except Exception:
                pass

        if os.path.exists(temp_pdf_out):
            import shutil
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            shutil.move(temp_pdf_out, pdf_path)

        if os.path.exists(pdf_path):
            print(f"[✓] Đã tạo thành công file PDF (.pdf): {pdf_path}")
            return True
    return False


def main():
    root = Path(__file__).resolve().parent.parent.parent
    md_file = os.path.join(root, "[NHÓM 7] BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.md")
    docx_file = os.path.join(root, "[NHÓM 7] BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.docx")
    pdf_file = os.path.join(root, "[NHÓM 7] BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.pdf")

    print(f"[*] Bắt đầu xuất báo cáo từ: {md_file}")
    create_docx(md_file, docx_file)
    create_pdf_from_html(md_file, pdf_file)


if __name__ == "__main__":
    main()
