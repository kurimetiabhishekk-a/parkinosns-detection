
import sys
import os
import traceback

# Force unbuffered output
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, name):
       return getattr(self.stream, name)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# Add current directory to path
sys.path.append(os.getcwd())

from voiceTest import testVoice
import shutil

# Copy healthy_tone.wav to upload/test.wav
os.makedirs('upload', exist_ok=True)
if os.path.exists('healthy_tone.wav'):
    shutil.copy('healthy_tone.wav', 'upload/test.wav')
else:
    print("WARNING: healthy_tone.wav not found")

print("Starting voice test...")
try:
    result = testVoice()
    print(f"SUCCESS_RESULT: {result}")
except Exception as e:
    print(f"CRASH_ERROR: {e}")
    traceback.print_exc()
print("Test script finished.")
