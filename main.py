"""
SORT多目标跟踪系统主程序
读取MOT16格式数据，进行目标跟踪，并可视化结果
"""
import os
import cv2
import numpy as np
from datetime import datetime
from sort import Sort
from kalman_tracker import KalmanBoxTracker, NSAKalmanTracker, IOUTracker


def load_detections(det_file, min_score=0.05):
    """
    加载检测结果文件（归一化后的分数，范围0~1）
    
    Args:
        det_file: 检测文件路径，格式为 MOT16 det_norm.txt（已归一化的检测文件）
        min_score: 最小置信度阈值，低于此值的检测框将被过滤
    
    Returns:
        dict: 键为帧号，值为该帧的检测框列表，每个检测框为 [x, y, w, h, score]
    """
    detections = {}
    
    if not os.path.exists(det_file):
        print(f"错误：检测文件不存在: {det_file}")
        return detections
    
    filtered_count = 0
    total_count = 0
    
    with open(det_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 处理MOT16格式的数据: frame, id, x, y, w, h, score, ...
            parts = line.split(',')
            if len(parts) < 7:
                continue
            
            total_count += 1
            frame_id = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])  # 第7列（索引6）是归一化后的置信度score（范围0~1）
            
            # 过滤低置信度检测框，减少噪音干扰
            if score < min_score:
                filtered_count += 1
                continue
            
            if frame_id not in detections:
                detections[frame_id] = []
            
            detections[frame_id].append([x, y, w, h, score])
    
    print(f"检测框统计: 总计 {total_count} 个，过滤 {filtered_count} 个低置信度框，保留 {total_count - filtered_count} 个")
    
    # 转换为numpy数组，确保形状为 (N, 5)
    for frame_id in detections:
        detections[frame_id] = np.array(detections[frame_id])
    
    return detections


def draw_boxes(img, detections, tracks, trackers, frame_id):
    """
    在图像上绘制检测框、跟踪框和轨迹
    
    Args:
        img: 输入图像
        detections: 当前帧的检测框，格式为 [x, y, w, h, score]
        tracks: 当前帧的跟踪结果，格式为 [x, y, w, h, id]
        trackers: 跟踪器列表，用于获取历史轨迹
        frame_id: 当前帧号
    
    Returns:
        绘制后的图像
    """
    img_draw = img.copy()
    
    # 创建ID到跟踪器的映射
    id_to_tracker = {}
    if trackers is not None:
        for trk in trackers:
            if trk.time_since_update < 1:  # 只处理当前帧活跃的跟踪器
                id_to_tracker[trk.id + 1] = trk  # +1 使ID从1开始
    
    # 绘制检测框（黑色细线）
    if detections is not None and len(detections) > 0:
        for det in detections:
            x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 0), 1)
    
    # 为每个ID生成不同颜色
    colors = {}
    if tracks is not None and len(tracks) > 0:
        for track in tracks:
            track_id = int(track[4])
            if track_id not in colors:
                # 使用HSV颜色空间生成不同颜色
                hue = int(180 * track_id / 100) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                colors[track_id] = tuple(map(int, color))
    
    # 绘制轨迹（在绘制跟踪框之前，这样轨迹在底层）
    # 已注释：不显示轨迹
    # if trackers is not None:
    #     for trk in trackers:
    #         if trk.time_since_update < 1 and len(trk.history) > 1:  # 只绘制活跃跟踪器的轨迹
    #             track_id = trk.id + 1
    #             if track_id in colors:
    #                 color = colors[track_id]
    #                 
    #                 # 将历史中心点转换为整数坐标
    #                 points = np.array(trk.history, dtype=np.int32)
    #                 
    #                 # 限制轨迹长度，只显示最近的部分（例如最近50个点，显示更长轨迹）
    #                 max_trail_length = 150
    #                 if len(points) > max_trail_length:
    #                     points = points[-max_trail_length:]
    #                 
    #                 # 使用polylines绘制轨迹，使用更粗的线条和更鲜艳的颜色
    #                 if len(points) >= 2:
    #                     # 将点转换为 (N, 1, 2) 格式
    #                     points = points.reshape((-1, 1, 2))
    #                     # 增加线条粗细，使轨迹更明显（thickness=3）
    #                     cv2.polylines(img_draw, [points], False, color, 3)
    
    # 绘制跟踪框（彩色粗线 + ID）
    if tracks is not None and len(tracks) > 0:
        for track in tracks:
            track_id = int(track[4])
            color = colors[track_id]
            
            x, y, w, h = int(track[0]), int(track[1]), int(track[2]), int(track[3])
            
            # 绘制跟踪框（粗线）
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 3)
            
            # 绘制ID文本
            label = f"ID: {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_draw, (x, y - text_height - baseline - 5), 
                         (x + text_width, y), color, -1)
            cv2.putText(img_draw, label, (x, y - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 绘制帧号
    cv2.putText(img_draw, f"Frame: {frame_id}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img_draw


def save_results(output_file, results):
    """
    将跟踪结果保存为标准MOT格式的txt文件
    
    Args:
        output_file: 输出文件路径
        results: 跟踪结果列表，每个元素为 [frame_id, track_id, x, y, w, h]
    
    输出格式（标准MOT格式）:
        frame_id, track_id, x, y, w, h, 1, -1, -1, -1
    """
    # 按帧号和track_id排序
    results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
    
    with open(output_file, 'w') as f:
        for result in results_sorted:
            frame_id, track_id, x, y, w, h = result
            # 标准MOT格式: frame_id, track_id, x, y, w, h, 1, -1, -1, -1
            line = f"{int(frame_id)},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
            f.write(line)
    
    print(f"跟踪结果已保存到: {output_file}")
    print(f"共保存 {len(results_sorted)} 条跟踪记录")


def main():
    """主函数"""
    # ========== 实验配置：选择跟踪模式 ==========
    # 可选值: 'Baseline', 'Standard', 'NSA'
    # EXPERIMENT_MODE = 'Baseline'
    # EXPERIMENT_MODE = 'Standard'
    EXPERIMENT_MODE = 'NSA'
    # ============================================
    
    # 硬编码数据路径
    sequence_dir = "./MOT16/train/MOT16-04"
    
    # 检查序列目录是否存在
    if not os.path.exists(sequence_dir):
        print(f"错误：序列目录不存在: {sequence_dir}")
        print("请确保数据路径正确：./MOT16/train/MOT16-04/")
        return
    
    # 硬编码序列信息（MOT16-04的标准信息）
    seq_info = {
        'name': 'MOT16-04',
        'imDir': 'img1',
        'frameRate': 30.0,
        'seqLength': 1050,
        'imWidth': 1920,
        'imHeight': 1080,
        'imExt': '.jpg'
    }
    
    print(f"序列信息: {seq_info['name']}")
    print(f"总帧数: {seq_info['seqLength']}")
    print(f"图像尺寸: {seq_info['imWidth']}x{seq_info['imHeight']}")
    
    # 加载检测结果（使用归一化后的检测文件）
    det_file = os.path.join(sequence_dir, "det", "det_norm.txt")
    detections = load_detections(det_file)
    print(f"加载了 {len(detections)} 帧的检测结果")
    
    # 图像目录
    img_dir = os.path.join(sequence_dir, seq_info['imDir'])
    
    # 根据实验模式选择跟踪器类型
    if EXPERIMENT_MODE == 'Baseline':
        tracker_class = IOUTracker
        tracker_name = "Baseline"
        print("使用 IOUTracker (Baseline: 无卡尔曼滤波)")
    elif EXPERIMENT_MODE == 'NSA':
        tracker_class = NSAKalmanTracker
        tracker_name = "NSA"
        print("使用 NSAKalmanTracker (自适应噪声跟踪器)")
    else:  # 'Standard'
        tracker_class = KalmanBoxTracker
        tracker_name = "Standard"
        print("使用 KalmanBoxTracker (标准跟踪器)")
    
    # 初始化SORT跟踪器（传入跟踪器类）
    mot_tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3, tracker_class=tracker_class)
    
    # 创建输出目录
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据跟踪器类型生成带后缀的文件名
    output_video_path = os.path.join(output_dir, f"MOT16-04_result_{tracker_name}.mp4")
    output_result_path = os.path.join(output_dir, f"res_{tracker_name}.txt")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    out = None
    
    # 用于收集跟踪结果的列表
    tracking_results = []
    
    # 主循环：按帧处理
    print("\n开始处理视频序列...")
    for frame_id in range(1, seq_info['seqLength'] + 1):
        # 读取图像
        img_filename = f"{frame_id:06d}{seq_info['imExt']}"
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"警告：图像文件不存在: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图像: {img_path}")
            continue
        
        # 获取当前帧的检测框（形状为 (N, 5)，包含score）
        frame_dets = detections.get(frame_id, np.empty((0, 5)))
        
        # 更新跟踪器（传递完整的检测框，包含score）
        if len(frame_dets) > 0:
            tracks = mot_tracker.update(frame_dets)  # 传递完整的 (N, 5) 数组
        else:
            tracks = mot_tracker.update(np.empty((0, 5)))
        
        # 收集跟踪结果（用于保存到txt文件）
        if len(tracks) > 0:
            for track in tracks:
                x, y, w, h, track_id = track[0], track[1], track[2], track[3], track[4]
                tracking_results.append([frame_id, track_id, x, y, w, h])
        
        # 获取跟踪器列表用于绘制轨迹
        trackers = mot_tracker.get_trackers()
        
        # 绘制结果（包括轨迹）
        img_result = draw_boxes(img, frame_dets, tracks, trackers, frame_id)
        
        # 初始化视频写入器（在第一帧时）
        if out is None:
            h, w = img_result.shape[:2]
            out = cv2.VideoWriter(output_video_path, fourcc, seq_info['frameRate'], (w, h))
            if not out.isOpened():
                print(f"错误：无法创建视频文件: {output_video_path}")
                print("尝试使用其他编码器...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_video_path = os.path.join(output_dir, "MOT16-04_result.avi")
                out = cv2.VideoWriter(output_video_path, fourcc, seq_info['frameRate'], (w, h))
        
        # 写入视频
        out.write(img_result)
        
        # 打印进度
        if frame_id % 50 == 0:
            print(f"处理进度: {frame_id}/{seq_info['seqLength']} 帧")
    
    # 释放资源
    if out is not None:
        out.release()
        print(f"\n视频已保存到: {output_video_path}")
    
    # 保存跟踪结果到txt文件（标准MOT格式）
    save_results(output_result_path, tracking_results)
    
    print("处理完成！")


if __name__ == "__main__":
    main()
