"""
工具函数模块
提供IoU计算等辅助功能
"""
import numpy as np


def iou_batch(bb_test, bb_gt):
    """
    计算两个边界框矩阵之间的IoU交并比矩阵
    
    Args:
        bb_test: numpy数组，形状为 (N, 4)，格式为 [x, y, w, h]
        bb_gt: numpy数组，形状为 (M, 4)，格式为 [x, y, w, h]
    
    Returns:
        numpy数组，形状为 (N, M)，表示bb_test中每个框与bb_gt中每个框的IoU值
    """
    # 将 [x, y, w, h] 转换为 [x1, y1, x2, y2] 格式
    bb_test = np.copy(bb_test)
    bb_gt = np.copy(bb_gt)
    
    # 转换bb_test: [x, y, w, h] -> [x1, y1, x2, y2]
    bb_test[:, 2] = bb_test[:, 0] + bb_test[:, 2]  # x2 = x + w
    bb_test[:, 3] = bb_test[:, 1] + bb_test[:, 3]  # y2 = y + h
    
    # 转换bb_gt: [x, y, w, h] -> [x1, y1, x2, y2]
    bb_gt[:, 2] = bb_gt[:, 0] + bb_gt[:, 2]  # x2 = x + w
    bb_gt[:, 3] = bb_gt[:, 1] + bb_gt[:, 3]  # y2 = y + h
    
    # 计算交集区域
    xx1 = np.maximum(bb_test[:, 0][:, np.newaxis], bb_gt[:, 0][np.newaxis, :])
    yy1 = np.maximum(bb_test[:, 1][:, np.newaxis], bb_gt[:, 1][np.newaxis, :])
    xx2 = np.minimum(bb_test[:, 2][:, np.newaxis], bb_gt[:, 2][np.newaxis, :])
    yy2 = np.minimum(bb_test[:, 3][:, np.newaxis], bb_gt[:, 3][np.newaxis, :])
    
    # 计算交集面积
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # 计算并集面积
    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_test[:, np.newaxis] + area_gt[np.newaxis, :] - intersection
    
    # 计算IoU
    iou = intersection / union
    return iou

