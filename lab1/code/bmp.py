import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.fftpack import dct, idct
import math
from skimage.metrics import structural_similarity as ssim
import pywt

image_dir = "./image/samoye.bmp"

def main():
    class bmp:
        def __init__(self, image_dir):
            self.image_dir = image_dir
            self.image = cv2.imread(image_dir) # image为一个numpy数组，储存了图片的各个位置的rgb
            self.gray_image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
            self.height, self.width, self.channels = self.image.shape
            self.hist = None
        
        def getPixel(self,x,y):
            if 0<=x<self.width and 0<=y<self.height:
                pixel = self.image[y,x] # opencv中y代表行，其范围为高
                print(pixel)
                return pixel.tolist()
            else:
                return None 
        
        def drawImage(self, image):
            cv2.imshow('image', image)
            cv2.waitKey(0) # wait for input, if input, next code
            cv2.destroyAllWindows() # close all windows
        
        def drawRow(self, row):
            if 0<=row<self.height:
                row_image = self.image[row, :]
                self.drawImage(row_image)

        def drawCol(self, col):
            if 0<=col<self.width:
                col_image = self.image[:, col]
                self.drawImage(col_image)
    
        def getHist(self):
            if self.channels==3:
                image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) # 转为灰度图像
            hist = cv2.calcHist([image], [0], None, [256], [0,256]) # [256]表示横坐标每格表现256/256=1个像素 (hist是一个一维的numpy数组，返回的是各灰度值的出现次数)
            plt.plot(hist)
            plt.title("Pixel Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.xlim([0, 256])
            plt.show()
            self.hist = hist

        def getEntropy(self):
            if len(self.hist) == 0:
                self.getHist()

            # 计算每个像素值的概率
            total_pixels = self.width * self.height
            prob = self.hist / total_pixels

            # 计算信息熵
            entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))

            return entropy

        def PermutationFun(self, blockwidth, blockheight, seed):
            # 设置随机数种子以保证每次运行的结果一致性
            if seed is not None:
                random.seed(seed)

            # 创建一个副本图像，用于置乱图片
            permuted_image = self.image.copy()

            # 计算块的行数和列数
            num_rows = self.height // blockheight
            num_cols = self.width // blockwidth

            # 创建块的位置列表
            block_positions = [(x, y) for x in range(num_cols) for y in range(num_rows)]

            # 随机置乱块的位置
            random.shuffle(block_positions)

            # 遍历每个块并将其复制到副本图像中
            for i, (block_x, block_y) in enumerate(block_positions): # numerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据
                # 计算块的像素坐标
                x1 = block_x * blockwidth
                y1 = block_y * blockheight
                x2 = x1 + blockwidth
                y2 = y1 + blockheight

                # 提取原始图像中的块
                block = self.image[y1:y2, x1:x2]

                # 计算副本图像中的块位置
                dest_x1 = i % num_cols * blockwidth
                dest_y1 = i // num_cols * blockheight
                dest_x2 = dest_x1 + blockwidth
                dest_y2 = dest_y1 + blockheight

                # 将块复制到副本图像中的新位置
                permuted_image[dest_y1:dest_y2, dest_x1:dest_x2] = block

            # 显示置乱后的图像
            cv2.imshow('Permuted Image', permuted_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def screenshot(self, x1, x2, y1, y2, if_save):
            cropped_image = self.image[y1:y2, x1:x2] # 先宽（列）后高（行）
            if if_save:
                cv2.imwrite("./image/cropped_image.jpg", cropped_image)
            cv2.imshow("Cropped Image", cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def DFT(self):
            # 读取灰度图像
            image = self.gray_image

            # 执行二维DFT
            dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

            # 将零频率（低频）分量移动到频谱的中心
            dft_shift = np.fft.fftshift(dft)

            # dft_shift[:, :, 0] 和 dft_shift[:, :, 1] 分别提取了 DFT 变换后的图像中的实部和虚部
            # dft_shift[:, :, 0] 中的前两维代表频域中的坐标
            # cv2.magnitude 函数用于计算复数的幅度（模），它将实部和虚部作为参数传递，并返回相应位置的幅度值
            # np.log 函数对幅度值进行自然对数运算，以便在对数尺度下表示幅度。这有助于突出幅度的差异，特别是在频域中存在大范围的值时
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) # 幅度谱
            phase_spectrum = np.angle(dft_shift[:, :, 0] + 1j*dft_shift[:, :, 1])  # 相位谱

            # 121 表示1行2列中的第一个
            plt.subplot(131), plt.imshow(image, cmap='gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title('DFT Magnitude Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(133), plt.imshow(phase_spectrum, cmap='gray')
            plt.title('DFT Phase Image'), plt.xticks([]), plt.yticks([])

            plt.show()
        
        def DCT(self):
            # 读取灰度图像
            image = self.gray_image

            # 进行二维DCT变换
            dct = cv2.dct(np.float32(image))

            # np.abs(dct)计算DCT系数矩阵的绝对值
            dct_coefficient =np.power(np.log(np.abs(dct) + 1), 0.5)

            # 显示DCT变换结果
            plt.subplot(121), plt.imshow(image, cmap='gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(122), plt.imshow(dct_coefficient, cmap='gray')# 默认进行缩放（0-255），如果添加vmin、vmax，则先进行截断再进行缩放
            plt.title('DCT Image'), plt.xticks([]), plt.yticks([])

            plt.show()

            # cv2.imshow('DCT', dct_coefficient)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        def psnr(self, original, compressed):
            mse = np.mean((original - compressed) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            return psnr

        def dct2(self, block):
            return dct(dct(block.T, norm='ortho').T, norm='ortho')

        def idct2(self, block):
            return idct(idct(block.T, norm='ortho').T, norm='ortho')

        def DCT2(self):
            # 读取原始图像
            original_image = self.gray_image[:936, :936]
            
            # 执行8x8的DCT变换
            dct_blocks = np.zeros_like(original_image, dtype=float)
            for i in range(0, original_image.shape[0], 8):
                for j in range(0, original_image.shape[1], 8):
                    dct_blocks[i:i+8, j:j+8] = self.dct2(original_image[i:i+8, j:j+8])
            
            # 截断DCT系数并执行逆DCT变换以恢复图像
            k = 8  # 保留的系数数量
            compressed_blocks = np.zeros_like(dct_blocks)
            for i in range(0, dct_blocks.shape[0], 8):
                for j in range(0, dct_blocks.shape[1], 8):
                    sorted_indices = np.argsort(np.abs(dct_blocks[i:i+8, j:j+8]).ravel())[::-1]
                    for idx in range(k):
                        row, col = np.unravel_index(sorted_indices[idx], (8, 8))
                        compressed_blocks[i+row, j+col] = dct_blocks[i+row, j+col]
            
            # 执行逆DCT变换以恢复图像
            reconstructed_image = np.zeros_like(original_image, dtype=float)
            for i in range(0, original_image.shape[0], 8):
                for j in range(0, original_image.shape[1], 8):
                    reconstructed_image[i:i+8, j:j+8] = self.idct2(compressed_blocks[i:i+8, j:j+8])
            
            reconstructed_image = np.clip(reconstructed_image, 0, 255)
            reconstructed_image = np.round(reconstructed_image).astype(np.uint8)
            
            # 计算PSNR
            psnr_value = self.psnr(original_image, reconstructed_image)
            print("PSNR:", psnr_value)

            # 计算SSIM
            ssim_value = ssim(original_image, reconstructed_image)
            print("SSIM:", ssim_value)

            # 绘制原始图像、恢复图像和DCT图像
            plt.subplot(121)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(122)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])

            plt.show()

        def DWT(self):
            # 小波基函数和层级
            wavelet = 'haar'  # 小波基函数，例如'haar'、'db1'等
            level = 3  # 变换的层级

            # 执行DWT
            coeffs = pywt.wavedec2(self.gray_image, wavelet, level=level)

            # 获取逼近系数和细节系数
            approximation = coeffs[0]
            details = coeffs[1:]

            # 可视化
            plt.figure(figsize=(12, 4))
            plt.subplot(141)
            plt.imshow(approximation, cmap='gray')
            plt.title('Approximation (A%d)' % level)

            for i, detail in enumerate(details):
                plt.subplot(142 + i)
                plt.imshow(detail[00], cmap='gray')
                plt.title('Detail (D%d)' % (level - i))
            plt.show()



    bmp = bmp(image_dir)
    bmp.getPixel(100, 100) # 获取任意一点像素值
    bmp.drawRow(100) # 画出任意一行
    bmp.drawCol(885) # 画出任意一列
    bmp.getHist() # 统计像素直方图
    entropy = bmp.getEntropy()
    print(f'the entropy of picture is: {entropy}') # 获取图像信息熵
    bmp.PermutationFun(20,20,2) # 图像分块并打乱
    bmp.screenshot(100, 800, 100, 400, if_save=True) # 图像裁剪
    
    # FFT是DFT的一种快速实现算法
    bmp.DFT() # 二维DFT变换
    bmp.DCT() # 二维DCT变换
    bmp.DCT2() # 二维DCT变换并恢复，同时输出图像的PSNR和SSIM值
    bmp.DWT() # 二维DWT变换
    

if __name__ == '__main__':
    main()
    


