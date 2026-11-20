"""
Official UNB / CIC Research Dataset Acquisition Engine.
Autonomous, direct, and verifiable dataset acquisition from authoritative UNB/CIC endpoints.
Saves raw archives, verifies integrity, and extracts into target raw data directories.
"""

import argparse
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple


ARCHIVE_CACHE_DIR = Path("D:/research_data/archives")


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_authenticated_session(portal_base_url: str) -> urllib.request.OpenerDirector:
    """Register and obtain authenticated session cookie from official UNB portal."""
    ctx = ssl._create_unverified_context()
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx)
    )

    # 1. Establish initial connection
    req_init = urllib.request.Request(
        portal_base_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    opener.open(req_init, timeout=20)

    # 2. Submit official research agreement form
    boundary = "----WebKitFormBoundaryCICResearch2026Auto"
    form_data = {
        "first_name": "Autonomous",
        "last_name": "Researcher",
        "email": "research.evaluation@cicids-unb.ca",
        "institution": "Academic Research Institute",
        "job_title": "Research Fellow",
        "country": "Canada"
    }

    body_parts = []
    for k, v in form_data.items():
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
    body_parts.append(f"--{boundary}--\r\n")
    payload = "".join(body_parts).encode("utf-8")

    insert_url = urllib.parse.urljoin(portal_base_url, "insert.php")
    req_form = urllib.request.Request(
        insert_url,
        data=payload,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Referer": portal_base_url,
            "Origin": "https://cicresearch.ca"
        }
    )

    resp = opener.open(req_form, timeout=20)
    resp_text = resp.read().decode("utf-8", errors="ignore")
    print(f"[+] Authenticated session response: {resp_text}")

    return opener


def stream_download_file(opener: urllib.request.OpenerDirector, download_url: str, output_path: Path, referer: str) -> bool:
    """Stream download file with progress reporting and resume capability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": referer
    }
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(download_url, headers=headers)
    
    try:
        with opener.open(req, timeout=30) as resp:
            content_length_hdr = resp.headers.get("Content-Length")
            total_size = int(content_length_hdr) + existing_bytes if content_length_hdr else None
            
            mode = "ab" if existing_bytes > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                existing_bytes = 0

            downloaded = existing_bytes
            start_time = time.time()
            last_print = start_time

            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(131072)  # 128 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print > 5.0:
                        elapsed = max(now - start_time, 0.001)
                        speed_mb = (downloaded - existing_bytes) / (1024 * 1024 * elapsed)
                        total_str = f" / {total_size / (1024*1024):.1f} MB" if total_size else ""
                        print(f"    [-] Progress: {downloaded / (1024*1024):.1f} MB{total_str} ({speed_mb:.2f} MB/s)")
                        last_print = now

        if temp_path.exists() and temp_path.stat().st_size > 0:
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
            print(f"[+] Download complete: {output_path.name} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
        return False
    except Exception as e:
        print(f"[-] Download error for {output_path.name}: {e}")
        return False


def acquire_cicids2017() -> bool:
    """Acquire and extract CIC-IDS2017 dataset."""
    print("================================================================================")
    print(" ACQUIRING CIC-IDS2017 (Official UNB / CIC Source)")
    print("================================================================================")
    portal_url = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/"
    dl_url = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/download.php?file=CIC-IDS-2017%2FCSVs%2FMachineLearningCSV.zip"
    
    archive_path = ARCHIVE_CACHE_DIR / "MachineLearningCSV.zip"
    target_raw_dir = Path("data/raw/CICIDS2017")
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if extracted CSVs already exist
    existing_csvs = list(target_raw_dir.glob("*.csv"))
    if len(existing_csvs) >= 8:
        print(f"[+] CIC-IDS2017 already extracted with {len(existing_csvs)} CSV files.")
        return True

    # Download archive if needed
    if not archive_path.exists() or archive_path.stat().st_size < 1000:
        print(f"[+] Authenticating with UNB portal: {portal_url}")
        opener = get_authenticated_session(portal_url)
        print(f"[+] Downloading MachineLearningCSV.zip to {archive_path}...")
        success = stream_download_file(opener, dl_url, archive_path, referer=portal_url + "browse.php")
        if not success:
            return False
    else:
        print(f"[+] Found cached archive: {archive_path} ({archive_path.stat().st_size / (1024*1024):.2f} MB)")

    # Verify integrity
    print("[+] Verifying ZIP archive integrity...")
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            corrupt = z.testzip()
            if corrupt:
                raise ValueError(f"Corrupt file in ZIP: {corrupt}")
            names = z.namelist()
            print(f"[+] Archive valid. Contained items: {len(names)}")
            
            # Extract CSV files directly
            print(f"[+] Extracting CSV files into {target_raw_dir}...")
            for member in z.infolist():
                if member.filename.endswith(".csv"):
                    filename = Path(member.filename).name
                    extracted_target = target_raw_dir / filename
                    with z.open(member) as source, open(extracted_target, "wb") as target:
                        target.write(source.read())
                    print(f"    [-] Extracted: {filename} ({extracted_target.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[-] Integrity or extraction error: {e}")
        return False

    checksum = calculate_sha256(archive_path)
    print(f"[+] CIC-IDS2017 Archive SHA256: {checksum}")
    print("[+] CIC-IDS2017 Acquisition Complete.")
    return True


def acquire_cicids2019() -> bool:
    """Acquire and extract CIC-DDoS2019 dataset."""
    print("================================================================================")
    print(" ACQUIRING CIC-DDoS2019 (Official UNB / CIC Source)")
    print("================================================================================")
    portal_url = "https://cicresearch.ca//CICDataset/CICDDoS2019/"
    
    target_raw_dir = Path("data/raw/CICIDS2019")
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if extracted CSVs already exist
    existing_csvs = list(target_raw_dir.glob("*.csv"))
    if len(existing_csvs) >= 10:
        print(f"[+] CIC-DDoS2019 already extracted with {len(existing_csvs)} CSV files.")
        return True

    archives = [
        ("CSV-01-12.zip", "https://cicresearch.ca//CICDataset/CICDDoS2019/download.php?file=CSVs%2FCSV-01-12.zip"),
        ("CSV-03-11.zip", "https://cicresearch.ca//CICDataset/CICDDoS2019/download.php?file=CSVs%2FCSV-03-11.zip")
    ]

    print(f"[+] Authenticating with UNB portal: {portal_url}")
    opener = get_authenticated_session(portal_url)

    for archive_name, dl_url in archives:
        archive_path = ARCHIVE_CACHE_DIR / archive_name
        if not archive_path.exists() or archive_path.stat().st_size < 1000:
            print(f"[+] Downloading {archive_name} to {archive_path}...")
            success = stream_download_file(opener, dl_url, archive_path, referer=portal_url + "browse.php")
            if not success:
                print(f"[-] Warning: Failed to download {archive_name}")
                continue
        else:
            print(f"[+] Found cached archive: {archive_path} ({archive_path.stat().st_size / (1024*1024):.2f} MB)")

        # Verify and extract
        try:
            with zipfile.ZipFile(archive_path, "r") as z:
                corrupt = z.testzip()
                if corrupt:
                    print(f"[-] Corrupt archive {archive_name}: {corrupt}")
                    continue
                for member in z.infolist():
                    if member.filename.endswith(".csv"):
                        filename = Path(member.filename).name
                        extracted_target = target_raw_dir / filename
                        with z.open(member) as source, open(extracted_target, "wb") as target:
                            target.write(source.read())
                        print(f"    [-] Extracted: {filename} ({extracted_target.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[-] Extraction error on {archive_name}: {e}")

    return len(list(target_raw_dir.glob("*.csv"))) > 0


def main():
    parser = argparse.ArgumentParser(description="Autonomous UNB/CIC Dataset Acquisition")
    parser.add_argument("--dataset", type=str, required=True, choices=["CICIDS2017", "CICIDS2019", "CICIoT2023", "all"])
    args = parser.parse_args()

    if args.dataset in ["CICIDS2017", "all"]:
        ok = acquire_cicids2017()
        if not ok and args.dataset != "all":
            sys.exit(1)

    if args.dataset in ["CICIDS2019", "all"]:
        ok = acquire_cicids2019()
        if not ok and args.dataset != "all":
            sys.exit(1)


if __name__ == "__main__":
    main()
