from pathlib import Path
import numpy as np
import csv
import re
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_point_cloud(depth_map):
    # 获取深度图的尺寸
    rows, cols = depth_map.shape[:2]

    # 创建网格坐标
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))

    # 构建点云数据
    # 在此处，利用深度信息直接计算点云的三维坐标
    # 根据具体情况和深度图的表示方法，可能需要适当的缩放或转换
    points = np.zeros((rows, cols, 3))
    # 这里仅是一个示例，根据深度信息直接计算三维坐标
    points[:, :, 0] = c
    points[:, :, 1] = r
    points[:, :, 2] = depth_map

    # 重塑为点云数组
    point_cloud = points.reshape(-1, 3)

    return point_cloud

def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c=point_cloud[:, 2], cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization')

    plt.show()



# 读取相机参数
def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)
    return calib


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')

    img = np.reshape(dispariy, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    # show(img, "disparity")
    return dispariy, [(height, width, channels), scale]


def create_depth_map(pfm_file_path, calib=None):
    dispariy, [shape, scale] = read_pfm(pfm_file_path)

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])
        # scale factor is used here
        # d = bf/(d+ doffs)         doffs就是(x_or-x_ol) 两个相机主点在各自图像坐标系x方向上的坐标差
        depth_map = fx * base_line / (dispariy / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).astype('uint8')
        return depth_map


def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)



def read_depth():
    pfm_file_dir = Path(r'../my_data/Mask-perfect')
    calib_file_path = pfm_file_dir.joinpath('calib.txt')
    disp_left = pfm_file_dir.joinpath('disp0.pfm')
    # calibration information
    calib = read_calib(calib_file_path)
    # create depth map
    depth_map_left = create_depth_map(disp_left, calib)
    return depth_map_left


if __name__ == '__main__':
    depth = read_depth()
    print(depth.shape)
    point_cloud = generate_point_cloud(depth)
    visualize_point_cloud(point_cloud)