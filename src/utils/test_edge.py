# -*- coding: utf-8 -*-
import subprocess, os, tempfile

def test_edge_pdf():
    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    temp_dir = tempfile.mkdtemp()
    html_file = os.path.join(temp_dir, "test.html")
    pdf_file = os.path.join(temp_dir, "test.pdf")

    with open(html_file, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>Xin chào</h1></body></html>")

    user_data = os.path.join(temp_dir, "user_data")
    cmd = [
        edge_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--user-data-dir={user_data}",
        f"--print-to-pdf={pdf_file}",
        html_file
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    print("Exit code:", res.returncode)
    print("PDF exists:", os.path.exists(pdf_file))
    if os.path.exists(pdf_file):
        print("Size:", os.path.getsize(pdf_file))

if __name__ == "__main__":
    test_edge_pdf()
