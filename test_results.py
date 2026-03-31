import os
import sys
import numpy as np
from PIL import Image
from utils import predictImg

print("--- TESTING DRAWING ANALYSIS ---")

test_files = ['good.jpg', 'bad.jpg', 'test_spiral.jpg']

for f in test_files:
    if os.path.exists(f):
        print(f"\nAnalyzing {f}...")
        res = predictImg(f)
        print(f"Result: {res}")
    else:
        print(f"File {f} not found.")
