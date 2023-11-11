from pydub import AudioSegment
import os
import time

def compress_and_decompress_audio(input_file, output_file):
    # 读取原始音频文件
    audio = AudioSegment.from_wav(input_file)

    # 计时开始
    start_time = time.time()

    # 压缩为MP3格式
    audio.export(output_file, format="mp3", codec="libmp3lame")

    # 计时结束
    end_time = time.time()

    # 计算文件大小
    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)

    # 计算压缩倍数
    compression_ratio = original_size / compressed_size

    # 计算压缩时间
    compression_time = end_time - start_time

    return original_size, compressed_size, compression_ratio, compression_time

def decompress_audio(input_file, output_file):
    # 读取压缩后的MP3音频文件
    compressed_audio = AudioSegment.from_mp3(input_file)

    # 将MP3音频转换为WAV格式
    decompressed_audio = compressed_audio.set_channels(1).set_frame_rate(44100)  # 设置通道数和采样率，可以根据需要进行调整

    # 导出解压缩后的WAV音频
    decompressed_audio.export(output_file, format="wav")

if __name__ == "__main__":
    input_audio_file = "./wav/shexiangfuren.wav"
    compressed_audio_file = "./wav/compressed_audio.mp3"
    decompressed_audio_file = "./wav/decompressed_audio.wav"

    original_size, compressed_size, compression_ratio, compression_time = compress_and_decompress_audio(input_audio_file, compressed_audio_file)
    decompress_audio(compressed_audio_file, decompressed_audio_file)

    print(f"原始文件大小: {original_size} 字节")
    print(f"压缩后文件大小: {compressed_size} 字节")
    print(f"压缩率: {compression_ratio*100:.2f} %")
    print(f"压缩时间: {compression_time:.2f} 秒")

