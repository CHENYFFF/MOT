"""
MOT跟踪结果评估脚本
使用motmetrics库计算跟踪指标
"""
import motmetrics as mm
import numpy as np
import os


def evaluate(gt_path, res_path):
    """
    评估跟踪结果
    
    Args:
        gt_path: Ground Truth文件路径
        res_path: 跟踪结果文件路径
    
    Returns:
        dict: 包含评估指标的字典
    """
    # 检查文件是否存在
    if not os.path.exists(gt_path):
        print(f"错误：Ground Truth文件不存在: {gt_path}")
        return None
    
    if not os.path.exists(res_path):
        print(f"错误：跟踪结果文件不存在: {res_path}")
        return None
    
    # 加载Ground Truth和跟踪结果
    gt = mm.io.loadtxt(gt_path, fmt='mot16')
    ts = mm.io.loadtxt(res_path, fmt='mot16')
    
    # 计算累积器
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    
    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    
    # 格式化输出
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    # 提取关键指标
    # summary是DataFrame，需要正确访问
    mota = 0.0
    idf1 = 0.0
    idsw = 0.0
    
    # 获取行索引（通常是'acc'）
    row_idx = summary.index[0] if len(summary.index) > 0 else 'acc'
    
    # 尝试提取MOTA（检查所有可能的列名）
    for col_name in summary.columns:
        if col_name.lower() == 'mota':
            try:
                value = summary.loc[row_idx, col_name]
                # 处理百分比格式或直接数值
                if isinstance(value, str):
                    if '%' in value:
                        mota = float(value.replace('%', '').strip()) / 100.0
                    else:
                        mota = float(value)
                else:
                    mota = float(value)
                break
            except (KeyError, TypeError, ValueError, IndexError) as e:
                continue
    
    # 尝试提取IDF1
    for col_name in summary.columns:
        if col_name.lower() == 'idf1':
            try:
                value = summary.loc[row_idx, col_name]
                if isinstance(value, str):
                    if '%' in value:
                        idf1 = float(value.replace('%', '').strip()) / 100.0
                    else:
                        idf1 = float(value)
                else:
                    idf1 = float(value)
                break
            except (KeyError, TypeError, ValueError, IndexError):
                continue
    
    # 尝试提取ID Sw (列名是'IDs'，表示ID切换次数)
    # 从输出表格看，列名是'IDs'（大写I和D，小写s）
    # 先尝试精确匹配
    if 'IDs' in summary.columns:
        try:
            idsw = float(summary.loc[row_idx, 'IDs'])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    
    # 如果没找到，尝试小写
    if idsw == 0.0 and 'ids' in summary.columns:
        try:
            idsw = float(summary.loc[row_idx, 'ids'])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    
    # 如果还是没找到，遍历所有列名匹配
    if idsw == 0.0:
        for col_name in summary.columns:
            # 匹配'ids'（不区分大小写）
            if col_name.lower() == 'ids':
                try:
                    idsw = float(summary.loc[row_idx, col_name])
                    break
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
    
    return {
        'summary': summary,
        'strsummary': strsummary,
        'mota': mota,
        'idf1': idf1,
        'idsw': idsw
    }


def main():
    """主函数：批量评估多个跟踪结果"""
    # Ground Truth文件路径
    gt_file = "./MOT16/train/MOT16-04/gt/gt.txt"
    
    # 跟踪结果文件列表（3种模式）
    result_files = [
        "./output/res_Baseline.txt",
        "./output/res_Standard.txt",
        "./output/res_NSA.txt"
    ]
    
    # 检查GT文件是否存在
    if not os.path.exists(gt_file):
        print(f"错误：Ground Truth文件不存在: {gt_file}")
        print("请确保MOT16数据集已正确放置")
        return
    
    print("=" * 80)
    print("MOT跟踪结果评估")
    print("=" * 80)
    print(f"Ground Truth: {gt_file}\n")
    
    # 循环评估每个结果文件
    for res_file in result_files:
        # 从文件路径提取模式名
        mode_name = os.path.basename(res_file).replace('res_', '').replace('.txt', '')
        
        print(f"\n评估模式: {mode_name}")
        print(f"结果文件: {res_file}")
        print("-" * 80)
        
        # 执行评估
        result = evaluate(gt_file, res_file)
        
        if result is not None:
            # 打印详细指标表格
            print(result['strsummary'])
        else:
            print(f"评估失败: {res_file}")
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()

