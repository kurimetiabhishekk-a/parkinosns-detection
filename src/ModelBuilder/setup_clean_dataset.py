import os, shutil

base  = os.path.join(os.path.dirname(__file__), "spiral_data")
clean = os.path.join(os.path.dirname(__file__), "spiral_data_clean")

# Create clean directory with only 2 class folders
for cls in ["Healthy", "Parkinson"]:
    d = os.path.join(clean, cls)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Copy ONLY spiral training+testing images (not wave or synthetic)
mapping = {"healthy": "Healthy", "parkinson": "Parkinson"}
counts = {"Healthy": 0, "Parkinson": 0}

for split in ["training", "testing"]:
    for src_cls, dst_cls in mapping.items():
        src = os.path.join(base, "spiral", split, src_cls)
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    dest_name = split + "_" + f
                    shutil.copy2(os.path.join(src, f), os.path.join(clean, dst_cls, dest_name))
                    counts[dst_cls] += 1

print("Clean spiral dataset ready (2 classes only):")
print("  Healthy  :", counts["Healthy"], "images")
print("  Parkinson:", counts["Parkinson"], "images")
print("  Location :", clean)
