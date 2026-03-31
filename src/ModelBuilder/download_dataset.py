

import os, sys, shutil, zipfile

DATASET   = "kmader/parkinsons-drawings"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "spiral_data")
HEALTHY   = os.path.join(OUT_DIR, "Healthy")
PARKINSON = os.path.join(OUT_DIR, "Parkinson")

def check_kaggle_api():
    try:
        import kaggle
        return True
    except ImportError:
        print("[ERROR] kaggle package not found. Run: pip install kaggle")
        return False

def check_api_key():
    key_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(key_path):
        print("[ERROR] Kaggle API key not found!")
        print(f"\nExpected at: {key_path}")
        print("\nTo get your API key:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token' — downloads kaggle.json")
        print(f"  4. Move it to: {key_path}")
        return False
    print(f"[OK] Kaggle API key found at: {key_path}")
    return True

def download():
    import kaggle
    zip_path = os.path.join(OUT_DIR, "parkinsons-drawings.zip")
    print(f"\n[1/4] Downloading dataset '{DATASET}'...")
    os.makedirs(OUT_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(DATASET, path=OUT_DIR, unzip=False, quiet=False)

    zips = [f for f in os.listdir(OUT_DIR) if f.endswith(".zip")]
    if not zips:
        print("[ERROR] No zip file found after download.")
        sys.exit(1)
    zip_path = os.path.join(OUT_DIR, zips[0])
    print(f"[2/4] Extracting {zips[0]}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(OUT_DIR)
    os.remove(zip_path)
    print("[OK] Extracted.")

def organize():
    
    print("[3/4] Organizing images into Healthy/ and Parkinson/ folders...")
    os.makedirs(HEALTHY, exist_ok=True)
    os.makedirs(PARKINSON, exist_ok=True)

    copied_h = copied_p = 0
    for root, dirs, files in os.walk(OUT_DIR):

        if root.startswith(HEALTHY) or root.startswith(PARKINSON):
            continue
        folder_name = os.path.basename(root).lower()
        if folder_name in ("healthy", "health"):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(root, f), os.path.join(HEALTHY, f))
                    copied_h += 1
        elif folder_name in ("parkinson", "parkinsons", "patient"):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy2(os.path.join(root, f), os.path.join(PARKINSON, f))
                    copied_p += 1

    print(f"  Healthy   images: {copied_h}")
    print(f"  Parkinson images: {copied_p}")
    return copied_h, copied_p

def main():
    print("=" * 55)
    print("Parkinson's Drawing Dataset Setup")
    print("=" * 55)

    if not check_kaggle_api():
        sys.exit(1)
    if not check_api_key():
        sys.exit(1)

    download()
    h, p = organize()

    if h >= 10 and p >= 10:
        print("\n[4/4] Dataset ready!")
        print(f"  Healthy   -> {HEALTHY}")
        print(f"  Parkinson -> {PARKINSON}")
        print("\nNow retrain the model:")
        print("  python train_drawing_cnn.py")
    else:
        print(f"\n[WARNING] Not enough images found (Healthy={h}, Parkinson={p}).")
        print("Please check extraction manually in:", OUT_DIR)

if __name__ == "__main__":
    main()
