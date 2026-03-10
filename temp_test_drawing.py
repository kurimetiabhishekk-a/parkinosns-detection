import os
import sys
import glob
from PIL import Image
from utils import _geometric_spiral_analysis

for f in glob.glob('*.jpg') + glob.glob('*.png'):
    try:
        im = Image.open(f).convert('L')
        status, tremor_index, metrics = _geometric_spiral_analysis(im)
        print(f"{f}: status={status}, tremor={tremor_index:.2f}")
    except Exception as e:
        print(f"{f}: ERROR {e}")
