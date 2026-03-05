
import wave
import math
import struct

def generate_sine_wave(frequency=200, duration=1.0, volume=0.5, sample_rate=44100):
    n_samples = int(sample_rate * duration)
    # 16-bit audio
    max_amplitude = 32767 * volume
    
    with wave.open("healthy_tone.wav", "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        
        for i in range(n_samples):
            # t = i / sample_rate
            value = int(max_amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
    print("Generated healthy_tone.wav")

if __name__ == "__main__":
    generate_sine_wave()
