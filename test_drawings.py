
import os
import sys
sys.path.append(os.getcwd())
from utils import predictImg

print("Testing good.jpg...")
res_good = predictImg('good.jpg')
print("Result for good.jpg:", res_good)

print("\nTesting bad.jpg...")
res_bad = predictImg('bad.jpg')
print("Result for bad.jpg:", res_bad)

print("\nChecking model files...")
print("drawing_model.pkl exists:", os.path.exists('drawing_model.pkl'))
print("drawing_scaler.pkl exists:", os.path.exists('drawing_scaler.pkl'))
print("keras_model.h5 exists:", os.path.exists('keras_model.h5'))
