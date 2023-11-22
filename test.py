import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from matplotlib.colors import hsv_to_rgb
from scipy.fftpack import fft

# 参数设置
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 声道数
sample_rate = 44100  # 采样率
chunk_size = 1024  # 每次读取的帧数
duration = 0.1

plt.tight_layout()

while True:
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size)
    frames = []
    for i in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames)
    fft_result = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(fft_result), 1.0 / sample_rate)
    
    max_index = np.argmax(np.abs(fft_result))
    detected_frequency = np.abs(frequencies[max_index])

    rms = np.sqrt(np.mean(audio_data**2))

    t = np.arange(0, len(audio_data) / sample_rate, 1 / sample_rate)
    sine_wave = 0.5 * np.sin(2 * np.pi * detected_frequency * t)

    rms_sine_wave = np.sqrt(np.mean(sine_wave**2))
    sine_wave *= rms / rms_sine_wave
    
    color = hsv_to_rgb([detected_frequency / 350, 1, rms / 50])

    print(f"Detected frequency: {detected_frequency} Hz, rms: {rms}, color: {color}")

   # 显示颜色
    plt.imshow([[color]])

    # 显示图形
    plt.pause(0.1)  # 适当的延迟以观察颜色变化
    plt.draw()

    continue

