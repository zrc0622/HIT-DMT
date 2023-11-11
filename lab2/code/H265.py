import ffmpeg
import os

def compress_video(input_video, output_video):
    # 获取原始视频文件的大小
    original_size = os.path.getsize(input_video) / (1024 * 1024)  # 将文件大小从字节转换为兆字节（MB）

    # 使用FFmpeg进行H.264压缩
    ffmpeg.input(input_video).output(output_video, vcodec='libx265').run()

    # 获取压缩后的视频文件大小
    compressed_size = os.path.getsize(output_video) / (1024 * 1024)  # 将文件大小从字节转换为兆字节（MB）

    # 计算压缩率
    compression_ratio = original_size / compressed_size

    return original_size, compressed_size, compression_ratio

if __name__ == "__main__":
    input_video = './mp4/video.mp4'
    output_video = './mp4/compressed.mp4'

    original_size, compressed_size, compression_ratio = compress_video(input_video, output_video)

    print(f'原始视频大小: {original_size:.2f} MB')
    print(f'压缩后视频大小: {compressed_size:.2f} MB')
    print(f'压缩率: {compression_ratio*100:.2f} %')
