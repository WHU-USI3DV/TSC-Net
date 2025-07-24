import numpy as np
import torch
import cv2
from scipy.spatial import distance
from skimage.measure import regionprops, label

import numpy as np

def merge_masks_union(masks):
    """
    将多个mask合并为它们的并集（逻辑或操作）

    参数:
        masks: mask列表，每个mask是HxW的numpy数组，值为0或1

    返回:
        numpy.ndarray: 合并后的mask (HxW), 值为0或1
    """
    if not masks:
        raise ValueError("mask列表不能为空")

    # 初始化结果mask为全0，与第一个mask形状相同
    result = np.zeros_like(masks[0])

    # 对所有mask进行逻辑或操作
    for mask in masks:
        result = np.logical_or(result, mask)

    return result.astype(np.uint8)


def calculate_vegetation_area_around_trees(depth_map, vegetation_mask, tree_masks, camera_matrix, radius=10.0):
    """
    计算每棵树中心周围指定半径内的植被面积

    参数:
        depth_map: 深度图 (H x W) torch.Tensor, 单位是米
        vegetation_mask: 植被掩膜 (H x W) numpy数组或torch.Tensor, 值为1表示植被区域
        tree_masks: 单棵树掩膜列表 [ (tree_id, mask), ... ] 或 多通道树掩膜 (numpy或torch.Tensor)
        camera_matrix: 相机内参矩阵 (3x3) numpy数组或torch.Tensor
        radius: 考虑的范围半径 (米), 默认为10米

    返回:
        dict: {tree_id: 周围植被面积} 单位平方米
    """
    # 转换输入为numpy数组（为了兼容性）
    if torch.is_tensor(depth_map):
        depth_map_np = depth_map.cpu().numpy()
    else:
        depth_map_np = np.array(depth_map)

    if torch.is_tensor(vegetation_mask):
        vegetation_mask_np = vegetation_mask.cpu().numpy()
    else:
        vegetation_mask_np = np.array(vegetation_mask)

    if torch.is_tensor(camera_matrix):
        camera_matrix_np = camera_matrix.cpu().numpy()
    else:
        camera_matrix_np = np.array(camera_matrix)

    # 处理树掩膜输入格式
    if isinstance(tree_masks, (np.ndarray, torch.Tensor)) and (len(tree_masks.shape) == 3 or torch.is_tensor(tree_masks) and tree_masks.ndim == 3):
        # 多通道树掩膜情况
        if torch.is_tensor(tree_masks):
            tree_masks_np = tree_masks.cpu().numpy()
        else:
            tree_masks_np = tree_masks
        tree_masks_list = [(i, tree_masks_np[:, :, i]) for i in range(tree_masks_np.shape[2])]
    else:
        # 假设已经是列表格式，转换每个mask为numpy
        tree_masks_list = []
        for tree_id, mask in enumerate(tree_masks):
            if torch.is_tensor(mask):
                tree_masks_list.append((tree_id, mask.cpu().numpy()))
            else:
                tree_masks_list.append((tree_id, np.array(mask)))

    results = {}

    # 获取所有植被点的3D坐标
    all_veg_points = get_3d_points_from_mask(depth_map_np, vegetation_mask_np, camera_matrix_np)
    np.savetxt('all_veg_points.txt', all_veg_points, fmt='%.4f')

    for tree_id, tree_mask in tree_masks_list:
        # 计算树的中心点(2D像素坐标)
        tree_center_2d = calculate_mask_center(tree_mask)

        # 获取树的中心点3D坐标
        tree_center_3d = get_3d_point_from_pixel(depth_map_np, tree_center_2d, camera_matrix_np)

        if tree_center_3d is None:
            results[tree_id] = 0.0
            continue

        # 筛选半径10米内的植被点
        distances = distance.cdist([tree_center_3d[:, :2]], all_veg_points[:, :2])[0]
        nearby_mask = distances <= radius
        nearby_veg_points = all_veg_points[nearby_mask]

        # 计算面积
        if len(nearby_veg_points) > 0:
            # 方法1: 简单投影法 (更快)
            pixel_area = estimate_pixel_area(nearby_veg_points, camera_matrix_np[0, 0], camera_matrix_np[1, 1])
            area = len(nearby_veg_points) * pixel_area

            # 方法2: 三角剖分法 (更精确但更慢)
            # area = estimate_surface_area_with_triangulation(nearby_veg_points)
        else:
            area = 0.0

        results[tree_id] = area

    return results

def get_3d_points_from_mask(depth_map, mask, camera_matrix, min_depth=0.1, max_depth=80.0):
    """
    从掩膜区域获取3D点云

    参数:
        depth_map: 深度图 numpy数组
        mask: 感兴趣区域掩膜 numpy数组
        camera_matrix: 相机内参 numpy数组
        min_depth: 最小有效深度
        max_depth: 最大有效深度

    返回:
        numpy.ndarray: (N, 3) 3D点云数组
    """
    height, width = depth_map.shape

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # 筛选有效点
    valid_mask = (mask > 0) & (depth_map > min_depth) & (depth_map < max_depth)
    u = u[valid_mask]
    v = v[valid_mask]
    depths = depth_map[valid_mask]

    if len(depths) == 0:
        return np.zeros((0, 3))

    # 转换为3D坐标
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) * depths / fx
    y = (v - cy) * depths / fy
    z = depths

    return np.column_stack((x, y, z))

def calculate_mask_center(mask):
    """
    计算掩膜的中心点坐标

    参数:
        mask: 二值掩膜 numpy数组

    返回:
        tuple: (x, y) 中心点坐标
    """
    # 使用regionprops计算质心
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return (mask.shape[1]//2, mask.shape[0]//2)  # 默认中心

    # 返回最大连通区域的质心
    largest_region = max(props, key=lambda x: x.area)
    return (int(largest_region.centroid[1]), int(largest_region.centroid[0]))

def get_3d_point_from_pixel(depth_map, pixel, camera_matrix, min_depth=0.1, max_depth=80.0):
    """
    从单个像素点获取3D坐标

    参数:
        depth_map: 深度图 numpy数组
        pixel: (x, y) 像素坐标
        camera_matrix: 相机内参 numpy数组
        min_depth: 最小有效深度
        max_depth: 最大有效深度

    返回:
        numpy.ndarray: (3,) 3D坐标 或 None(如果无效)
    """
    x, y = pixel
    if x < 0 or y < 0 or x >= depth_map.shape[1] or y >= depth_map.shape[0]:
        return None

    depth = depth_map[y, x]
    if depth <= min_depth or depth >= max_depth:
        return None

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x_3d = (x - cx) * depth / fx
    y_3d = (y - cy) * depth / fy
    z_3d = depth

    return np.array([x_3d, y_3d, z_3d])

def estimate_pixel_area(points_3d, fx, fy):
    """
    估算单个像素对应的实际面积
    """
    if len(points_3d) < 2:
        return 0.0

    # 计算平均深度
    avg_depth = np.mean(points_3d[:, 2])

    # 计算单个像素对应的实际尺寸
    pixel_width = avg_depth / fx
    pixel_height = avg_depth / fy

    return pixel_width * pixel_height

# 示例使用方式
if __name__ == "__main__":
    # 创建示例数据 (PyTorch Tensor)
    height, width = 720, 1280
    depth_map = torch.rand((height, width)) * 19.0 + 1.0  # 1-20米深度
    vegetation_mask = torch.randint(0, 2, (height, width), dtype=torch.uint8)

    # 创建3棵树的掩膜示例 (PyTorch Tensor)
    tree_masks = torch.zeros((height, width, 3), dtype=torch.uint8)
    tree_masks[200:300, 300:400, 0] = 1  # 树1
    tree_masks[400:500, 600:700, 1] = 1  # 树2
    tree_masks[500:600, 200:300, 2] = 1  # 树3

    # 相机内参示例 (numpy数组)
    camera_matrix = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ])

    # 计算每棵树周围10米内的植被面积
    results = calculate_vegetation_area_around_trees(
        depth_map,
        vegetation_mask,
        tree_masks,
        camera_matrix,
        radius=10.0
    )

    # 打印结果
    for tree_id, area in results.items():
        print(f"树 {tree_id} 周围10米内的植被面积: {area:.2f} 平方米")
