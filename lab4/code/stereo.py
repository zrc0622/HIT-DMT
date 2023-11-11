import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class Camera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[4844.97,0.,1332.834], [0., 4844.97, 979.162],[0., 0., 1.]])
        # 右相机内参
        self.cam_matrix_right = np.array([[4844.97, 0., 1495.13],[0., 4844.97, 979.162],[0., 0., 1.]])
        # 主点列坐标的差
        self.doffs = 162.296
        # 畸变系数
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        # 旋转矩阵
        self.R = np.identity(3, dtype=np.float64)
        # 平移矩阵
        self.T = np.array([[-174.945], [0.0], [0.0]])
        self.isRectified = True

def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2

# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 10
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right

def getDepthMapWithConfig(disparityMap: np.ndarray, config: Camera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)


if __name__ == '__main__':
    # 读取MiddleBurry数据集的图片
    iml = cv2.imread('../my_data/im0.png', 1)  # 左图
    imr = cv2.imread('../my_data/im1.png', 1)  # 右图

    height, width = iml.shape[0:2]

    # 读取相机内参和外参
    config = Camera()

    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

    # 立体匹配
    disp, _ = stereoMatchSGBM(iml, imr, True) 

    # 计算深度图
    depthMap = getDepthMapWithConfig(disp, config)
    minDepth = np.min(depthMap)
    maxDepth = np.max(depthMap)
    depthMapVis = (255.0 * (depthMap - minDepth)) / (maxDepth - minDepth)
    depthMapVis = depthMapVis.astype(np.uint8)

    stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=15)

    # 计算视差图
    disparity = stereo.compute(iml, imr)

    # 根据视差图计算深度信息
    # 深度 = 焦距 * 双目距离 / 视差
    # Z = baseline * f / (d + doffs)
    depth = 4844.97 * 170.458 / (disparity + 162.296)
    depth_disparity = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 显示深度图像
    plt.imshow(depth_disparity, cmap='gray')
    plt.show()

    # 使用open3d库绘制点云
    iml = cv2.cvtColor(iml, cv2.COLOR_BGR2RGB)
    colorImage = o3d.geometry.Image(iml)
    depthImage = o3d.geometry.Image(depthMap)
    rgbdImage = o3d.geometry.RGBDImage.create_from_color_and_depth(colorImage, depthImage, depth_scale=1000.0,
                                                                     depth_trunc=np.inf,convert_rgb_to_intensity=False)
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    fx = config.cam_matrix_left[0, 0]
    fy = fx
    cx = config.cam_matrix_left[0, 2]
    cy = config.cam_matrix_left[1, 2]
    intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsics = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)


    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 参数中的Q就是由getRectifyTransform()函数得到的重投影矩阵

    # 构建点云--Point_XYZRGBA格式
    o3d.visualization.draw_geometries([pointcloud], width=720, height=480)

    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.005)  # 调整体素大小来降低点云密度 0.008(越小越密集)

    # 从点云生成网格
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pointcloud, alpha=0.03) # 0.02（越小越精细）

    # 清理无效部分
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # 显示网格
    o3d.visualization.draw_geometries([mesh], width=720, height=480)