import cv2
import wave
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import dct

# 读取音频文件
def read_audio_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        frame_rate = wf.getframerate()
        audio_data = wf.readframes(-1)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
    return audio_data, frame_rate, sample_width, num_frames

# 分窗处理音频数据
def window_audio(audio_data, window_size):
    windows = []
    for i in range(0, len(audio_data), window_size):
        window = audio_data[i:i+window_size]
        if len(window) == window_size:
            windows.append(window)
    return windows

# 绘制音频波形图
def plot_audio_waveform(audio_data, framerate, title):
    time = np.arange(0, len(audio_data)) / framerate
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_data)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# 进行一维DFT处理
def apply_dft(audio_data):
    dft_result = np.fft.fft(audio_data)
    return dft_result

# 进行一维DCT处理
def apply_dct(audio_data):
    dct_result = dct(audio_data, norm='ortho')
    return dct_result

# 进行一维DWT处理
def apply_dwt(audio_data):
    coeffs = pywt.dwt(audio_data, 'haar')
    return coeffs

# 绘制频域图
def plot_frequency_domain(data, title):
    plt.figure(figsize=(12, 4))
    plt.plot(np.abs(data))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def plot_first_window_features(audio_file, window_size):
    # 读取音频文件
    audio_data, framerate, _, _ = read_audio_file(audio_file)

    # 分窗处理音频数据
    windows = window_audio(audio_data, window_size)

    # 获取第一个窗口
    first_window = windows[0]

    # 绘制原始音频波形图
    plt.figure(figsize=(12, 6))
    plt.subplot(4, 1, 1)
    plt.plot(first_window)
    plt.title('Original Audio Waveform')

    # 进行一维DFT处理
    dft_result = apply_dft(first_window)
    plt.subplot(4, 1, 2)
    plt.plot(np.abs(dft_result))
    plt.title('DFT')

    # 进行一维DCT处理
    dct_result = apply_dct(first_window)
    plt.subplot(4, 1, 3)
    plt.plot(dct_result)
    plt.title('DCT')

    # 进行一维DWT处理
    coeffs = apply_dwt(first_window)
    plt.subplot(4, 1, 4)
    plt.plot(coeffs[0])
    plt.title('DWT')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = './wav/shexiangfuren.wav'
    window_size = 1024

    # 读取音频文件
    audio_data, frame_rate, sample_width, num_frames = read_audio_file(audio_file)

    # 分窗处理音频数据
    windows = window_audio(audio_data, window_size)

    # 绘制原始音频波形图(前1s)
    plot_audio_waveform(audio_data[0:frame_rate*1], frame_rate, "Original Audio Waveform")

    # 打印音频信息
    print(f"采样宽度（字节）: {sample_width}")
    print(f"采样率（帧率）: {frame_rate}")
    print(f"音频帧数: {num_frames}")

    plot_first_window_features(audio_file, window_size) # 分窗处理并绘制处理后的波形

    # for i, window in enumerate(windows):
    #     # 进行一维DFT处理
    #     dft_result = apply_dft(window)
    #     plot_frequency_domain(dft_result, f"DFT for Window {i}")

    #     # 进行一维DCT处理
    #     dct_result = apply_dct(window)
    #     plot_frequency_domain(dct_result, f"DCT for Window {i}")

    #     # 进行一维DWT处理
    #     coeffs = apply_dwt(window)
    #     plot_frequency_domain(coeffs[0], f"DWT Approximation Coefficients for Window {i}")
    #     plot_frequency_domain(coeffs[1], f"DWT Detail Coefficients for Window {i}")
