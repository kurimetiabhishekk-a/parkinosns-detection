
import parselmouth
import os

try:
    print("Testing Parselmouth...")
    wav_path = "healthy_tone.wav" # Exist from previous list_dir
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
    else:
        sound = parselmouth.Sound(wav_path)
        print(f"Sound loaded: duration={sound.duration}")
        pitch = sound.to_pitch()
        print(f"Pitch extracted: mean={pitch.get_mean()}")
    print("Parselmouth test PASSED")
except Exception as e:
    print(f"Parselmouth test FAILED: {e}")
    import traceback
    traceback.print_exc()
