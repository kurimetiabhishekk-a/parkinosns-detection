
import sys
import os

sys.path.append(os.getcwd())

from voiceTest import testVoice
import shutil

os.makedirs('upload', exist_ok=True)
shutil.copy('healthy_tone.wav', 'upload/test.wav')

print("Starting voice test...")
try:
    result = testVoice()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error during voice test: {e}")
    import traceback
    traceback.print_exc()
