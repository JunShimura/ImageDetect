import numpy as np
import sounddevice as sd

def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.5, fade_out=True):
    """
    トーンを生成する関数。
    fade_out: Trueの場合、最後に減衰効果を追加。
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    if fade_out:
        fade = np.linspace(1, 0, len(wave))  # 減衰するフェードアウト係数
        wave *= fade
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
sd.default.blocksize = 0  # 自動設定

# warm up
silent_tone = np.zeros(int(sample_rate * 2))  # 無音
sd.play(silent_tone, samplerate=sample_rate)
sd.wait()


# 各音を順番に再生
for note, freq in notes.items():
    print(f"Playing {note} ({freq} Hz)")
    tone = generate_tone(freq, duration, sample_rate)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()

def play_note(note):
    silent_tone = np.zeros(int(sample_rate * 0.0625))  # 無音
    sd.play(silent_tone, samplerate=sample_rate,latency='high')
    sd.wait()
    tone = generate_tone(notes[note], duration, sample_rate)
    sd.play(tone, samplerate=sample_rate,latency='high')
    sd.wait()

play_note("C4")
play_note("E4")
play_note("G4")
play_note("C5")