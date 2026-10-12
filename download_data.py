"""
Automated downloader and extractor for NEU Surface Defect Database.
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path

DATA_DIR = Path("data")
TARGET_DIR = DATA_DIR / "NEU-CLS"
GDRIVE_FILE_ID = "1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k"
ZIP_PATH = DATA_DIR / "neu_surface_defect_database.zip"

def download_with_gdown():
    print(">> Attempting download via gdown...")
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(ZIP_PATH), quiet=False)
        return ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1000
    except Exception as e:
        print(f"gdown download error: {e}")
        return False

def download_with_requests():
    print(">> Attempting download via requests stream...")
    try:
        import requests
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check for confirmation token if file is large
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                url = f"{url}&confirm={v}"
                response = session.get(url, stream=True)
                break
                
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    
        return ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1000
    except Exception as e:
        print(f"requests download error: {e}")
        return False

def verify_dataset(extract_path):
    print(">> Verifying extracted dataset structure...")
    expected_classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    # Check if files are directly inside or inside a subfolder
    subdirs = [p for p in extract_path.iterdir() if p.is_dir()]
    print(f"Found directories: {[p.name for p in subdirs]}")
    
    found_classes = {}
    for p in extract_path.rglob("*"):
        if p.is_dir():
            name_lower = p.name.lower().replace(" ", "_").replace("-", "_")
            for c in expected_classes:
                if c in name_lower:
                    images = list(p.glob("*.bmp")) + list(p.glob("*.jpg")) + list(p.glob("*.png"))
                    if len(images) > 0:
                        found_classes[c] = (p, len(images))
                        
    print(f">> Detected defect classes: {list(found_classes.keys())}")
    for c, (p, cnt) in found_classes.items():
        print(f"   - {c}: {cnt} images (in {p})")
        
    return len(found_classes) == 6

def main():
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    if TARGET_DIR.exists() and verify_dataset(TARGET_DIR):
        print(f">> Dataset is already verified and ready at {TARGET_DIR}!")
        return
        
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 1000:
        success = download_with_gdown()
        if not success:
            success = download_with_requests()
            
        if not success:
            print(">> Automatic download failed. You can manually download the zip from:")
            print(f"   https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
            print(f"   and place it at: {ZIP_PATH.resolve()}")
            sys.exit(1)

    print(f">> Extracting {ZIP_PATH} to {DATA_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    # Standardize directory name if needed
    verify_dataset(DATA_DIR)
    print(">> Dataset setup complete!")

if __name__ == "__main__":
    main()
