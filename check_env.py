
import os
import sys

print("Checking libraries...")

try:
    import librosa
    print("librosa version:", librosa.__version__)
except ImportError:
    print("librosa MISSING")

try:
    import parselmouth
    print("parselmouth AVAILABLE")
except ImportError:
    print("parselmouth MISSING")

try:
    import soundfile as sf
    print("soundfile AVAILABLE")
except ImportError:
    print("soundfile MISSING")

try:
    import joblib
    print("joblib version:", joblib.__version__)
except ImportError:
    print("joblib MISSING")

model_path = "src/trainedModel.sav"
if os.path.exists(model_path):
    print(f"Model file {model_path} EXISTS, size: {os.path.getsize(model_path)} bytes")
else:
    print(f"Model file {model_path} MISSING")

audio_path = "upload/test.wav"
if os.path.exists(audio_path):
     print(f"Audio file {audio_path} EXISTS, size: {os.path.getsize(audio_path)} bytes")
else:
     print(f"Audio file {audio_path} MISSING")
