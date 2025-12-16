"""
SORT (Simple Online and Realtime Tracking) 算法实现
使用Kalman滤波器和匈牙利算法进行多目标跟踪
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_tracker import KalmanBoxTracker
from utils import iou_batch


class Sort:
    """
    SORT多目标跟踪器
    
    使用Kalman滤波器进行状态预测，使用匈牙利算法进行数据关联
    """
    
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, tracker_class=None):
        """
        初始化SORT跟踪器
        
        Args:
            max_age: 目标丢失多少帧后删除跟踪器
            min_hits: 目标需要连续命中多少次才显示（用于过滤初始不稳定跟踪）
            iou_threshold: IoU匹配阈值，低于此值不进行匹配
            tracker_class: 跟踪器类，默认为 KalmanBoxTracker（支持依赖注入）
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # 当前活跃的跟踪器列表
        self.frame_count = 0  # 帧计数器
        
        # 支持依赖注入：如果没有传入 tracker_class，则使用默认的 KalmanBoxTracker
        if tracker_class is None:
            self.tracker_class = KalmanBoxTracker
        else:
            self.tracker_class = tracker_class
    
    def update(self, dets):
        """
        更新跟踪器状态
        
        Args:
            dets: numpy数组，形状为 (N, 5)，格式为 [x, y, w, h, score]
        
        Returns:
            numpy数组，形状为 (M, 5)，格式为 [x, y, w, h, id]，表示当前帧的跟踪结果
        """
        self.frame_count += 1
        
        # 确保输入格式正确
        if len(dets) > 0 and dets.shape[1] < 5:
            # 如果输入是 (N, 4)，补充score为1.0
            dets = np.column_stack([dets, np.ones(len(dets))])
        
        # 第一步：对所有现有跟踪器进行预测
        trks = np.zeros((len(self.trackers), 5))  # [x, y, w, h, id]
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # 预测下一帧位置
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            
            # 如果预测框无效（NaN或无穷大），标记为删除
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                to_del.append(t)
        
        # 删除无效的跟踪器
        if len(trks) > 0:
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        else:
            trks = np.empty((0, 5))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # 如果没有检测框，直接返回空结果
        if len(dets) == 0:
            return np.empty((0, 5))
        
        # 如果还有跟踪器，进行匹配
        matched, unmatched_dets, unmatched_trks = [], [], []
        
        if len(trks) > 0:
            # 第二步：计算预测框和检测框的IoU矩阵
            iou_matrix = iou_batch(dets[:, :4], trks[:, :4])
            
            # 第三步：使用匈牙利算法进行匹配
            # linear_sum_assignment 求解最小代价匹配（我们使用1-IoU作为代价）
            if min(iou_matrix.shape) > 0:
                # 将IoU转换为代价（1 - IoU），IoU越大，代价越小
                cost_matrix = 1 - iou_matrix
                
                # 只考虑IoU大于阈值的匹配
                cost_matrix[cost_matrix > (1 - self.iou_threshold)] = 1e5
                
                # 匈牙利算法求解最优匹配
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # 第四步：处理匹配结果
                for r, c in zip(row_ind, col_ind):
                    # 如果IoU小于阈值，视为未匹配
                    if iou_matrix[r, c] < self.iou_threshold:
                        unmatched_dets.append(r)
                        unmatched_trks.append(c)
                    else:
                        matched.append((r, c))
                
                # 找出未匹配的检测框
                unmatched_dets = [d for d in range(len(dets)) if d not in [m[0] for m in matched]]
                # 找出未匹配的跟踪器
                unmatched_trks = [t for t in range(len(trks)) if t not in [m[1] for m in matched]]
            else:
                # 如果IoU矩阵为空，所有检测框和跟踪器都未匹配
                unmatched_dets = list(range(len(dets)))
                unmatched_trks = list(range(len(trks)))
        else:
            # 如果没有跟踪器，所有检测框都未匹配
            unmatched_dets = list(range(len(dets)))
        
        # 第五步：处理匹配上的跟踪器 - 更新状态（传递完整的 [x, y, w, h, score]）
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(dets[det_idx])  # 传递完整的检测框，包含score
        
        # 处理未匹配的检测框 - 创建新跟踪器（传递完整的 [x, y, w, h, score]）
        for i in unmatched_dets:
            trk = self.tracker_class(dets[i])  # 使用注入的跟踪器类创建新跟踪器
            self.trackers.append(trk)
        
        # 处理未匹配的跟踪器 - 增加丢失计数
        # 收集跟踪结果
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # 确保d是一维数组
            d = np.array(d).flatten()
            
            # 只返回满足条件的跟踪结果：
            # 1. 连续命中次数 >= min_hits
            # 2. 或者跟踪器年龄足够大（避免初始不稳定）
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # 确保维度匹配：d是一维数组，[trk.id + 1]也是一维数组
                track_result = np.concatenate((d, [trk.id + 1])).reshape(1, -1)  # +1 使ID从1开始
                ret.append(track_result)
            
            i -= 1
            
            # 删除丢失时间过长的跟踪器
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        # 返回跟踪结果
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def get_trackers(self):
        """
        获取当前所有活跃的跟踪器列表
        
        Returns:
            跟踪器列表
        """
        return self.trackers
