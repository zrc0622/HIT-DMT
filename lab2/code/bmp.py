from PIL import Image
import fitz
import os

# 读取BMP图像
input_image_path = "./bmp/samoye.bmp"
output_jpeg_path = "./bmp/compressed.jpeg"
output_jpeg_decompressed_path = "./bmp/decompressed.bmp"

# 打开BMP图像
image = Image.open(input_image_path)

# 获取原始图像大小
original_size = os.path.getsize(input_image_path) / (1024.0 * 1024.0)  # 原始大小以MB为单位

# 压缩并保存为JPEG格式
image.save(output_jpeg_path, "JPEG", quality=95)  # 调整quality参数以改变压缩质量

# 解压缩JPEG并保存为BMP格式
compressed_image = Image.open(output_jpeg_path)
compressed_image.save(output_jpeg_decompressed_path, "BMP")

# 获取压缩后图像大小
compressed_size = os.path.getsize(output_jpeg_path) / (1024.0 * 1024.0)  # 压缩后大小以MB为单位

# 计算压缩率
compression_ratio = original_size / compressed_size

# 输出结果
print(f"原始图像大小: {original_size:.2f} MB")
print(f"压缩后图像大小: {compressed_size:.2f} MB")
print(f"压缩率: {compression_ratio*100:.2f} %")
