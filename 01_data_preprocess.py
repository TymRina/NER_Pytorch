import json
import re
import random
import os

# 设置随机种子，确保结果可重现
random.seed(42)

# 数据路径
DATA_PATH = "e:\\TymRina\\课程资料\\四阶段\\NER命名实体识别\\NER命名实体识别数据集\\data.json"
OUTPUT_DIR = "e:\\TymRina\\课程资料\\四阶段\\NER命名实体识别\\dataset"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
def load_data(file_path):
    """加载JSON格式的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# 文本清洗
def clean_text(text):
    """简单的文本清洗"""
    # 去除多余的空格和换行符
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    return text.strip()

# 转换为BIO标注格式
def convert_to_bio(text, mention_data):
    """将文本和实体信息转换为BIO标注格式"""
    # 将文本转换为字符列表
    text_chars = list(text)
    # 初始化标签序列为'O'
    labels = ['O'] * len(text_chars)
    
    # 按照实体长度排序，优先处理长实体
    mention_data_sorted = sorted(mention_data, key=lambda x: len(x['mention']), reverse=True)
    
    # 处理每个实体
    for entity in mention_data_sorted:
        mention = entity['mention']
        offset = int(entity['offset'])
        entity_type = entity['type']
        mention_len = len(mention)
        
        # 确保实体在文本范围内
        if offset < 0 or offset + mention_len > len(text_chars):
            continue
            
        # 检查实体文本是否完全匹配
        if ''.join(text_chars[offset:offset+mention_len]) == mention:
            # 设置开始标签
            labels[offset] = f'B-{entity_type}'
            # 设置内部标签
            for i in range(1, mention_len):
                if offset + i < len(text_chars):
                    labels[offset + i] = f'I-{entity_type}'
    
    return text_chars, labels

# 调整实体偏移量
def adjust_offsets(original_text, cleaned_text, mention_data):
    """调整实体的偏移量，使其与清洗后的文本匹配"""
    adjusted_mentions = []
    
    # 构建更健壮的原始文本到清洗后文本的映射
    original_pos = 0
    cleaned_pos = 0
    pos_map = {}  # 原始位置 -> 清洗后位置
    
    while original_pos < len(original_text) and cleaned_pos < len(cleaned_text):
        original_char = original_text[original_pos]
        
        # 检查当前原始字符是否需要保留
        if original_char.isspace() or not re.match(r'[\u4e00-\u9fa5a-zA-Z0-9]', original_char):
            # 跳过原始文本中的空格和特殊字符，不建立映射
            original_pos += 1
        elif cleaned_pos < len(cleaned_text) and original_char == cleaned_text[cleaned_pos]:
            # 字符匹配，建立映射
            pos_map[original_pos] = cleaned_pos
            original_pos += 1
            cleaned_pos += 1
        else:
            # 原始字符在清洗后文本中不存在，跳过
            original_pos += 1
    
    # 处理每个实体
    for entity in mention_data:
        original_offset = int(entity['offset'])
        original_mention = entity['mention']
        entity_type = entity['type']
        
        # 清洗实体提及文本
        cleaned_mention = clean_text(original_mention)
        if not cleaned_mention:
            continue
        
        # 优先使用位置映射
        if original_offset in pos_map:
            adjusted_offset = pos_map[original_offset]
            # 检查实体是否在清洗后的文本中存在且匹配
            if adjusted_offset + len(cleaned_mention) <= len(cleaned_text):
                if cleaned_text[adjusted_offset:adjusted_offset+len(cleaned_mention)] == cleaned_mention:
                    adjusted_mentions.append({
                        'mention': cleaned_mention,
                        'offset': adjusted_offset,
                        'type': entity_type
                    })
                    continue
        
        # 如果位置映射失败，尝试直接在清洗后的文本中查找
        start_idx = cleaned_text.find(cleaned_mention)
        if start_idx != -1:
            adjusted_mentions.append({
                'mention': cleaned_mention,
                'offset': start_idx,
                'type': entity_type
            })
    
    return adjusted_mentions

# 主函数
def main():
    print("加载数据...")
    data = load_data(DATA_PATH)
    print(f"共加载 {len(data)} 条数据")
    
    # 处理数据
    processed_data = []
    for item in data:
        original_text = item['text']
        cleaned_text = clean_text(original_text)
        # 如果清洗后的文本为空，跳过
        if not cleaned_text:
            continue
        
        # 调整实体偏移量
        adjusted_mentions = adjust_offsets(original_text, cleaned_text, item['mention_data'])
        
        tokens, labels = convert_to_bio(cleaned_text, adjusted_mentions)
        processed_data.append((tokens, labels))
    
    print(f"处理后剩余 {len(processed_data)} 条数据")
    
    # 划分训练集、验证集和测试集
    print("划分数据集...")
    # 手动实现数据集划分
    random.shuffle(processed_data)
    total_size = len(processed_data)
    test_size = int(total_size * 0.2)
    val_size = int(total_size * 0.2)
    train_size = total_size - test_size - val_size
    
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size+val_size]
    test_data = processed_data[train_size+val_size:]
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    # 保存数据
    def save_data(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for tokens, labels in data:
                # 确保tokens和labels长度一致
                if len(tokens) != len(labels):
                    continue
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # 句子之间用空行分隔
    
    save_data(train_data, os.path.join(OUTPUT_DIR, 'train.txt'))
    save_data(val_data, os.path.join(OUTPUT_DIR, 'val.txt'))
    save_data(test_data, os.path.join(OUTPUT_DIR, 'test.txt'))
    
    print(f"数据已保存到 {OUTPUT_DIR}")
    print("数据预处理完成！")

if __name__ == "__main__":
    main()