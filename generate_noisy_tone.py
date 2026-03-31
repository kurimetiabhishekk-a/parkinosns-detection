
import wave
import math
import struct
import random

def generate_noisy_sine_wave(frequency=200, duration=1.0, volume=0.5, sample_rate=44100, noise_level=0.01):
    n_samples = int(sample_rate * duration)
    max_amplitude = 32767 * volume
    
    with wave.open("noisy_tone.wav", "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        
        for i in range(n_samples):

            sine_val = math.sin(2 * math.pi * frequency * i / sample_rate)

            noise = (random.random() * 2 - 1) * noise_level
            
            value = int(max_amplitude * (sine_val + noise))

            value = max(-32767, min(32767, value))
            
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
    print("Generated noisy_tone.wav with noise level", noise_level)

if __name__ == "__main__":

    generate_noisy_sine_wave(noise_level=0.05)
