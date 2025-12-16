# SORT多目标跟踪系统

基于SORT (Simple Online and Realtime Tracking) 算法的多目标跟踪系统实现。

## 项目结构

```
MOT/
├── utils.py              # 工具函数（IoU计算）
├── kalman_tracker.py     # Kalman滤波器跟踪器
├── sort.py               # SORT算法核心实现
├── main.py               # 主程序
├── requirements.txt      # 依赖包列表
└── README.md            # 项目说明
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保MOT16数据集已放置在 `D:\CodeField\PyCode\MOT\MOT16` 目录下

2. 运行主程序：
```bash
python main.py
```

3. 在 `main.py` 中可以修改要处理的序列：
```python
sequence_name = "MOT16-02"  # 改为其他序列名称
```

## 算法说明

### SORT算法流程

1. **状态预测**：使用Kalman滤波器预测每个跟踪目标在下一帧的位置
2. **数据关联**：计算预测框与检测框的IoU矩阵，使用匈牙利算法进行最优匹配
3. **状态更新**：
   - 匹配成功的检测框更新对应的跟踪器
   - 未匹配的检测框创建新的跟踪器
   - 未匹配的跟踪器增加丢失计数，超过阈值则删除

### Kalman滤波器状态向量

- 状态维度：7维 `[u, v, s, r, u_dot, v_dot, s_dot]`
  - `u, v`: 边界框中心点坐标
  - `s`: 边界框面积 (w * h)
  - `r`: 边界框宽高比 (w / h)
  - `u_dot, v_dot, s_dot`: 对应的速度分量

- 观测维度：4维 `[u, v, s, r]`

## 参数说明

在 `main.py` 中初始化SORT跟踪器时可以调整以下参数：

- `max_age`: 目标丢失多少帧后删除跟踪器（默认：30）
- `min_hits`: 目标需要连续命中多少次才显示（默认：3）
- `iou_threshold`: IoU匹配阈值（默认：0.3）

## 输出结果

- 处理后的视频将保存为 `{序列名}_tracking.avi`
- 检测框用黑色细线绘制
- 跟踪框用彩色粗线绘制，并显示目标ID

## 注意事项

- 确保MOT16数据集格式正确
- 检测文件格式：`frame, id, x, y, w, h, score, ...`
- 图像文件命名格式：`{帧号:06d}.jpg`

