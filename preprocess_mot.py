"""
MOT16检测数据预处理脚本
对检测结果中的置信度分数进行Min-Max归一化
"""
import os
import numpy as np


def preprocess_mot_detections(input_file, output_file):
    """
    对MOT16检测文件进行Min-Max归一化处理
    
    Args:
        input_file: 输入文件路径，格式为 MOT16 det.txt
        output_file: 输出文件路径，保存归一化后的结果
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    # 第一步：遍历文件，找出所有score的最大值和最小值
    print("第一步：扫描文件，计算分数范围...")
    scores = []
    total_lines = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 7:
                continue
            
            total_lines += 1
            try:
                score = float(parts[6])  # 第7列（索引6）是score
                scores.append(score)
            except (ValueError, IndexError):
                continue
    
    if len(scores) == 0:
        print("错误：未找到有效的分数数据")
        return
    
    min_score = min(scores)
    max_score = max(scores)
    
    print(f"原始分数范围: Min = {min_score:.6f}, Max = {max_score:.6f}")
    print(f"总计 {total_lines} 行数据，其中 {len(scores)} 行包含有效分数")
    
    # 处理边界情况：如果max_score == min_score，所有分数归一化为0.5
    if max_score == min_score:
        print("警告：所有分数相同，归一化后所有分数将设为 0.5")
        norm_value = 0.5
    else:
        score_range = max_score - min_score
        print(f"分数范围: {score_range:.6f}")
    
    # 第二步：重新遍历文件，对每一行的score进行归一化
    print("\n第二步：归一化处理并写入新文件...")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            original_line = line.strip()
            if not original_line:
                f_out.write(line)  # 保留空行
                continue
            
            parts = original_line.split(',')
            if len(parts) < 7:
                f_out.write(line)  # 保留格式不正确的行
                skipped_count += 1
                continue
            
            try:
                # 获取原始分数
                original_score = float(parts[6])
                
                # 计算归一化分数
                if max_score == min_score:
                    norm_score = 0.5
                else:
                    norm_score = (original_score - min_score) / (max_score - min_score)
                
                # 替换第7列（索引6）为归一化后的分数
                parts[6] = f"{norm_score:.6f}"
                
                # 写入新行
                new_line = ','.join(parts) + '\n'
                f_out.write(new_line)
                processed_count += 1
                
            except (ValueError, IndexError) as e:
                # 如果无法解析分数，保留原行
                f_out.write(line)
                skipped_count += 1
                continue
    
    # 第三步：打印处理结果
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 行")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 行（格式错误或无法解析）")
    print(f"输出文件: {output_file}")
    
    # 验证输出文件
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"输出文件大小: {file_size / 1024:.2f} KB")


def main():
    """主函数"""
    # 设置输入和输出文件路径
    input_file = "./MOT16/train/MOT16-04/det/det.txt"
    output_file = "./MOT16/train/MOT16-04/det/det_norm.txt"
    
    print("=" * 60)
    print("MOT16 检测数据预处理脚本")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 60)
    
    # 执行预处理
    preprocess_mot_detections(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

