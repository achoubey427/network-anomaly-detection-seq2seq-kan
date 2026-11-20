"""
Extract Official Benchmark Archives into Raw Data Directories.
Supports CIC-DDoS2019 (CSV-01-12.zip, CSV-03-11.zip) and CICIoT2023 (CSV.zip).
"""

from pathlib import Path
import zipfile
import sys


import shutil

def extract_ddos2019():
    target_dir = Path("data/raw/CICIDS2019")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    existing_csvs = list(target_dir.glob("*.csv"))
    if len(existing_csvs) >= 8:
        print(f"[+] CIC-DDoS2019 already extracted: {len(existing_csvs)} CSV files present.")
        return

    archives = [
        Path(r"C:\Users\ASUS\Downloads\CSV-01-12.zip"),
        Path(r"E:\download\CSV-03-11.zip")
    ]

    for arch in archives:
        if not arch.exists():
            print(f"[-] Archive not found: {arch}")
            continue
        print(f"[+] Extracting {arch.name} ({arch.stat().st_size / (1024*1024):.2f} MB)...")
        with zipfile.ZipFile(arch, "r") as z:
            for member in z.infolist():
                if member.filename.endswith(".csv") and not member.filename.startswith("~") and not "#" in member.filename:
                    filename = Path(member.filename).name
                    # Prefix folder name if duplicate name (e.g. 01-12_UDPLag.csv vs 03-11_UDPLag.csv)
                    parent_tag = Path(member.filename).parent.name
                    out_name = f"{parent_tag}_{filename}" if parent_tag in ["01-12", "03-11"] else filename
                    dest_file = target_dir / out_name
                    if not dest_file.exists():
                        with z.open(member) as src, open(dest_file, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024*1024)
                        print(f"    [-] Extracted: {out_name} ({dest_file.stat().st_size / (1024*1024):.2f} MB)")

    print(f"[+] CIC-DDoS2019 extraction complete. Total CSV files: {len(list(target_dir.glob('*.csv')))}")


def extract_ciciot2023():
    target_dirs = [Path("data/raw/CICIoT2023"), Path("data/raw/CICIDS2023")]
    for td in target_dirs:
        td.mkdir(parents=True, exist_ok=True)

    arch = Path(r"E:\download\CSV.zip")
    if not arch.exists():
        print(f"[-] CICIoT2023 archive not found: {arch}")
        return

    existing_csvs = list(target_dirs[0].glob("*.csv"))
    if len(existing_csvs) >= 10:
        print(f"[+] CICIoT2023 already extracted: {len(existing_csvs)} CSV files present.")
        return

    print(f"[+] Extracting CICIoT2023 from {arch.name} ({arch.stat().st_size / (1024*1024):.2f} MB)...")
    with zipfile.ZipFile(arch, "r") as z:
        for member in z.infolist():
            if member.filename.endswith(".csv"):
                filename = Path(member.filename).name
                category = Path(member.filename).parent.name
                out_name = f"{category}_{filename}" if category and category != "CSV" else filename
                for td in target_dirs:
                    dest_file = td / out_name
                    if not dest_file.exists():
                        with z.open(member) as src, open(dest_file, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024*1024)
                print(f"    [-] Extracted: {out_name}")

    print(f"[+] CICIoT2023 extraction complete. Total CSV files: {len(list(target_dirs[0].glob('*.csv')))}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ciciot2023":
        extract_ciciot2023()
    elif len(sys.argv) > 1 and sys.argv[1] == "cicids2019":
        extract_ddos2019()
    else:
        extract_ddos2019()
        extract_ciciot2023()
