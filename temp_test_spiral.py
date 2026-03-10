
import sys
import os
sys.path.append(os.getcwd())
from utils import predictImg

test_img = "test_spiral.jpg" # From list_dir
if os.path.exists(test_img):
    print(f"Testing spiral with {test_img}...")
    result = predictImg(test_img)
    print(f"Result: {result}")
else:
    print(f"WARNING: {test_img} not found")
