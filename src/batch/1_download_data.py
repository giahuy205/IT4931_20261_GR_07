# -*- coding: utf-8 -*-
"""
Script tải và giải nén bộ dữ liệu MovieLens 25M vào thư mục Bronze Lake: data/bronze/
Sử dụng requests hỗ trợ bỏ qua kiểm tra SSL trên Windows nếu thiếu chứng chỉ root CA.
"""

import argparse
import os
import shutil
import sys
import time
import zipfile
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/bronze"))


def download_file(url, dest_path):
    print(f"[*] Bắt đầu tải MovieLens 25M từ: {url}")
    
    existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"[*] Tiếp tục tải từ byte {existing_size:,} ({existing_size / (1024*1024):.1f} MB)...")

    response = requests.get(url, headers=headers, stream=True, verify=False, timeout=120)

    # Nếu server hỗ trợ Range (status 206), lấy tổng size = existing + content_length
    if response.status_code == 206:
        total_size = existing_size + int(response.headers.get("content-length", 0))
        mode = "ab"
    else:
        total_size = int(response.headers.get("content-length", 0))
        existing_size = 0
        mode = "wb"

    chunk_size = 1024 * 1024  # 1 MB
    downloaded = existing_size
    start_time = time.time()

    with open(dest_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                duration = time.time() - start_time
                speed = ((downloaded - existing_size) / (1024 * 1024 * duration)) if duration > 0 else 0
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                sys.stdout.write(
                    f"\r[*] Đang tải: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB) - Tốc độ: {speed:.2f} MB/s"
                )
                sys.stdout.flush()

    print(f"\n[+] Tải hoàn tất: {dest_path}")


def extract_zip(zip_path, extract_to):
    print(f"[*] Đang giải nén {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    subfolder = os.path.join(extract_to, "ml-25m")
    if os.path.exists(subfolder):
        for filename in os.listdir(subfolder):
            src_file = os.path.join(subfolder, filename)
            dst_file = os.path.join(extract_to, filename)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.move(src_file, dst_file)
        os.rmdir(subfolder)
    print(f"[+] Giải nén thành công vào: {extract_to}")


def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu MovieLens 25M")
    parser.add_argument(
        "--url", default=MOVIELENS_25M_URL, help="URL tải dataset"
    )
    args = parser.parse_args()

    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_path = os.path.join(TARGET_DIR, "ml-25m.zip")

    # Kiểm tra xem các file CSV chính đã tồn tại chưa
    expected_files = ["ratings.csv", "movies.csv", "tags.csv"]
    if all(os.path.exists(os.path.join(TARGET_DIR, f)) for f in expected_files):
        print(f"[!] Dữ liệu MovieLens 25M đã tồn tại trong {TARGET_DIR}. Bỏ qua bước tải.")
        return

    # Nếu file zip chưa hoàn chỉnh, tiếp tục tải
    if not (os.path.exists(zip_path) and zipfile.is_zipfile(zip_path)):
        download_file(args.url, zip_path)

    extract_zip(zip_path, TARGET_DIR)


if __name__ == "__main__":
    main()
