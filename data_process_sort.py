import json
import jsonlines
import random # 导入 random 模块
def filter_top50_by_avg_quality(input_file, output_file, rate=0.5):
    """
    过滤对话数据，只保留平均质量分前50%的对话
    """
    # 读取清洗后的数据
    conversations = []
    with jsonlines.open(input_file, 'r') as reader:
        for conv_data in reader:
            conversations.append(conv_data)

    # 计算每个对话的平均质量分
    conv_with_avg = []
    for conv in conversations:
        qualities = [turn['quality'] for turn in conv['conversation']]
        avg_quality = sum(qualities) / len(qualities)
        conv_with_avg.append( (avg_quality, conv) )

    # 按平均分降序排序
    sorted_convs = sorted(conv_with_avg, key=lambda x: x[0], reverse=True)

    # 计算需要保留的数量 (四舍五入取整)
    total = len(sorted_convs)
    top50_count = int(round(total * rate))

    # 提取前50%的对话
    top50_convs = [item[1] for item in sorted_convs[:top50_count]]

    # 保存结果
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(top50_convs)
        
    print(f"\n筛选完成！前{rate*100}%高质量对话已保存至: {output_file}")

cleaned_output_path = '/root/cleaned_session-02_complex_quality.jsonl'
filtered_output_path = '/root/cleaned_session-02_complex_quality_0_01.jsonl' # 新输出文件路径

# 过滤前50%数据
filter_top50_by_avg_quality(cleaned_output_path, filtered_output_path,rate=0.01) #取前30%的数据训练

