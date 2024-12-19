import numpy as np
import sounddevice as sd

def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * pow(np.sin(2 * np.pi * frequency * t),2)
    return wave

# 音階と対応する周波数
notes = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25,
}

duration = 0.125  # 各音の長さ
sample_rate = 44100

# 各音を順番に再生
for note, freq in notes.items():
    print(f"Playing {note} ({freq} Hz)")
    tone = generate_tone(freq, duration, sample_rate)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()

def play_note(note):
    tone = generate_tone(note, duration, sample_rate)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()

play_note("C4")
play_note("E4")
play_note("G4")
play_note("C5")


