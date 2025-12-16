"""
Kalman滤波器跟踪器模块
实现单个目标的Kalman滤波跟踪
"""
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    使用Kalman滤波器跟踪单个目标边界框的类
    
    状态向量: [u, v, s, r, u_dot, v_dot, s_dot]
    - u, v: 边界框中心点坐标
    - s: 边界框面积 (w * h)
    - r: 边界框宽高比 (w / h)
    - u_dot, v_dot, s_dot: 对应的速度分量
    """
    
    count = 0  # 静态变量，用于分配唯一ID
    
    def __init__(self, bbox):
        """
        初始化Kalman滤波器跟踪器
        
        Args:
            bbox: 初始边界框，格式为 [x, y, w, h] 或 [x, y, w, h, score]
        """
        # 初始化Kalman滤波器，状态维度7，观测维度4
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # 状态转移矩阵 F (7x7)
        # 假设匀速运动模型
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # u' = u + u_dot
            [0, 1, 0, 0, 0, 1, 0],  # v' = v + v_dot
            [0, 0, 1, 0, 0, 0, 1],  # s' = s + s_dot
            [0, 0, 0, 1, 0, 0, 0],  # r' = r (常数)
            [0, 0, 0, 0, 1, 0, 0],  # u_dot' = u_dot
            [0, 0, 0, 0, 0, 1, 0],  # v_dot' = v_dot
            [0, 0, 0, 0, 0, 0, 1]   # s_dot' = s_dot
        ])
        
        # 观测矩阵 H (4x7)
        # 只能观测到 [u, v, s, r]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # 观测 u
            [0, 1, 0, 0, 0, 0, 0],  # 观测 v
            [0, 0, 1, 0, 0, 0, 0],  # 观测 s
            [0, 0, 0, 1, 0, 0, 0]   # 观测 r
        ])
        
        # 过程噪声协方差矩阵 Q (7x7)
        # 对角矩阵，表示状态转移的不确定性
        self.kf.Q[4:, 4:] *= 0.01  # 速度的不确定性较小
        self.kf.Q *= 0.1
        
        # 观测噪声协方差矩阵 R (4x4)
        # 对角矩阵，表示观测的不确定性
        self.kf.R[2:, 2:] *= 10.  # 面积和宽高比的观测不确定性较大
        
        # 状态协方差矩阵 P (7x7)
        # 初始不确定性
        self.kf.P[4:, 4:] *= 1000.  # 速度的初始不确定性较大
        self.kf.P *= 10.
        
        # 初始化状态
        x, y, w, h = bbox[:4]  # 只取前4个元素
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        s = w * h  # 面积
        r = w / float(h)  # 宽高比
        
        # 确保形状匹配：kf.x是列向量，需要正确赋值
        self.kf.x[0] = center_x
        self.kf.x[1] = center_y
        self.kf.x[2] = s
        self.kf.x[3] = r
        
        # 存储置信度score（如果提供）
        if len(bbox) > 4:
            self.score = bbox[4]
        else:
            self.score = 1.0  # 默认值
        
        # 分配唯一ID
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        # 跟踪状态
        self.time_since_update = 0  # 自上次更新以来的帧数
        self.hit_streak = 0  # 连续命中次数
        self.history = []  # 历史状态（存储中心点坐标用于绘制轨迹）
        self.age = 0  # 跟踪器年龄（总帧数）
    
    def update(self, bbox):
        """
        使用观测边界框更新Kalman滤波器状态
        
        Args:
            bbox: 观测到的边界框，格式为 [x, y, w, h] 或 [x, y, w, h, score]
        """
        self.time_since_update = 0
        self.hit_streak += 1
        
        # 存储置信度score
        if len(bbox) > 4:
            self.score = bbox[4]
        
        # 将边界框转换为观测向量 [u, v, s, r]
        x, y, w, h = bbox[:4]
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        s = w * h
        r = w / float(h)
        
        z = np.array([center_x, center_y, s, r])
        self.kf.update(z)
        
        # 注意：历史记录在predict()中累积，这里不重复添加
    
    def predict(self):
        """
        推进Kalman滤波器状态一步（预测下一帧的状态）
        
        Returns:
            预测的边界框，格式为 [x, y, w, h]
        """
        # 如果面积小于等于0，则不再预测
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        
        # 获取当前预测状态
        current_state = self.get_state()
        
        # 将中心点坐标添加到历史记录中（用于绘制轨迹）
        center_x = current_state[0] + current_state[2] / 2.0
        center_y = current_state[1] + current_state[3] / 2.0
        self.history.append((center_x, center_y))
        
        return current_state
    
    def get_state(self):
        """
        获取当前预测的边界框状态
        
        Returns:
            边界框，格式为 [x, y, w, h]（一维数组）
        """
        # 从状态向量 [u, v, s, r] 转换为 [x, y, w, h]
        # 确保获取的是一维数组
        state_vec = self.kf.x[:4].flatten()
        center_x, center_y, s, r = state_vec
        
        # 从面积和宽高比恢复宽度和高度
        w = np.sqrt(s * r)
        h = s / w
        
        # 从中心点恢复左上角坐标
        x = center_x - w / 2.0
        y = center_y - h / 2.0
        
        return np.array([x, y, w, h])
