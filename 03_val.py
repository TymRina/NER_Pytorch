import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from model import BiLSTM_CRF
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 数据加载器
class NERDataset:
    def __init__(self, data_file, word_to_ix, tag_to_ix, max_len=128):
        self.data = self.load_data(data_file)
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v: k for k, v in tag_to_ix.items()}
        self.max_len = max_len

    def load_data(self, data_file):
        """加载BIO格式的数据"""
        data = []
        sentence = []
        tags = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # 句子结束
                    if sentence and tags:
                        data.append((sentence, tags))
                        sentence = []
                        tags = []
                    continue
                
                # 处理每行数据，格式为：字符 标签
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[1]
                    sentence.append(word)
                    tags.append(tag)
        
        # 处理最后一个句子
        if sentence and tags:
            data.append((sentence, tags))
            
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取一个样本，转换为张量格式"""
        sentence, tags = self.data[idx]
        
        # 将句子转换为索引
        word_ids = [self.word_to_ix.get(word, self.word_to_ix['<unk>']) for word in sentence]
        # 将标签转换为索引
        tag_ids = [self.tag_to_ix[tag] for tag in tags]
        
        # 处理变长序列，进行填充或截断
        if len(word_ids) > self.max_len:
            word_ids = word_ids[:self.max_len]
            tag_ids = tag_ids[:self.max_len]
        else:
            pad_length = self.max_len - len(word_ids)
            word_ids += [self.word_to_ix['<pad>']] * pad_length
            tag_ids += [self.tag_to_ix['O']] * pad_length  # 使用'O'标签作为填充
        
        # 转换为张量
        word_ids = torch.tensor(word_ids, dtype=torch.long)
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)
        
        return word_ids, tag_ids

# NER评估函数
def evaluate_ner(predictions, labels, tag_to_ix):
    """评估NER模型的性能
    Args:
        predictions: 模型预测的标签序列
        labels: 真实标签序列
        tag_to_ix: 标签到索引的映射
    Returns:
        report: 分类报告
        cm: 混淆矩阵
    """
    # 将标签索引转换为标签名称
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # 扁平化预测和标签
    flat_predictions = []
    flat_labels = []
    
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            # 只考虑非填充位置（标签不是'O'）
            if labels[i][j] != tag_to_ix['O'] or predictions[i][j] != tag_to_ix['O']:
                flat_predictions.append(ix_to_tag[predictions[i][j]])
                flat_labels.append(ix_to_tag[labels[i][j]])
    
    # 计算分类报告
    report = classification_report(flat_labels, flat_predictions, zero_division=0)
    
    # 计算混淆矩阵
    cm = confusion_matrix(flat_labels, flat_predictions, labels=list(tag_to_ix.keys()))
    
    return report, cm

# 实体级别评估
def evaluate_entity_level(predictions, labels, tag_to_ix):
    """在实体级别评估模型性能
    Args:
        predictions: 模型预测的标签序列
        labels: 真实标签序列
        tag_to_ix: 标签到索引的映射
    Returns:
        entity_metrics: 实体级别的评估指标
    """
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # 提取实体
    def extract_entities(tags):
        entities = []
        current_entity = None
        
        for i, tag in enumerate(tags):
            tag_name = ix_to_tag[tag]
            
            if tag_name.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': tag_name[2:],
                    'start': i,
                    'end': i + 1
                }
            elif tag_name.startswith('I-') and current_entity:
                # 继续当前实体
                entity_type = tag_name[2:]
                if entity_type == current_entity['type']:
                    current_entity['end'] = i + 1
                else:
                    # 实体类型不匹配，开始新实体
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'start': i,
                        'end': i + 1
                    }
            else:
                # 非实体或实体结束
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    # 计算每个实体类型的TP、FP、FN
    entity_types = set()
    for tag in tag_to_ix.keys():
        if tag.startswith('B-'):
            entity_types.add(tag[2:])
    
    metrics = {}
    for entity_type in entity_types:
        metrics[entity_type] = {'TP': 0, 'FP': 0, 'FN': 0}
    metrics['total'] = {'TP': 0, 'FP': 0, 'FN': 0}
    
    for pred, gold in zip(predictions, labels):
        pred_entities = extract_entities(pred)
        gold_entities = extract_entities(gold)
        
        # 标记已匹配的实体
        matched = set()
        
        # 计算TP
        for p_ent in pred_entities:
            for g_ent in gold_entities:
                if (p_ent['type'] == g_ent['type'] and 
                    p_ent['start'] == g_ent['start'] and 
                    p_ent['end'] == g_ent['end']):
                    # 实体匹配
                    metrics[p_ent['type']]['TP'] += 1
                    metrics['total']['TP'] += 1
                    matched.add(g_ent)
                    break
            else:
                # 没有匹配的实体，FP
                metrics[p_ent['type']]['FP'] += 1
                metrics['total']['FP'] += 1
        
        # 计算FN
        for g_ent in gold_entities:
            if g_ent not in matched:
                metrics[g_ent['type']]['FN'] += 1
                metrics['total']['FN'] += 1
    
    # 计算精确率、召回率、F1值
    entity_metrics = {}
    for entity_type, counts in metrics.items():
        if counts['TP'] + counts['FP'] == 0:
            precision = 0.0
        else:
            precision = counts['TP'] / (counts['TP'] + counts['FP'])
        
        if counts['TP'] + counts['FN'] == 0:
            recall = 0.0
        else:
            recall = counts['TP'] / (counts['TP'] + counts['FN'])
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        entity_metrics[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': counts['TP'] + counts['FN']
        }
    
    return entity_metrics

# 主函数
def main():
    # 超参数设置
    embedding_dim = 128
    hidden_dim = 256
    batch_size = 64
    max_len = 128
    
    # 数据集路径
    test_file = os.path.join('dataset', 'test.txt')
    
    # 检查数据集是否存在
    if not os.path.exists(test_file):
        print("测试数据集不存在，请先运行数据预处理脚本！")
        return
    
    # 加载词汇表和标签映射
    vocab_path = os.path.join('runs', 'vocab.json')
    tag_map_path = os.path.join('runs', 'tag_map.json')
    
    if not os.path.exists(vocab_path) or not os.path.exists(tag_map_path):
        print("词汇表或标签映射不存在，请先运行训练脚本！")
        return
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        word_to_ix = json.load(f)
    
    with open(tag_map_path, 'r', encoding='utf-8') as f:
        tag_to_ix = json.load(f)
    
    # 加载测试数据集
    print("加载测试数据集...")
    test_dataset = NERDataset(test_file, word_to_ix, tag_to_ix, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型参数文件（如果存在）
    model_params_path = os.path.join('runs', 'model_params.json')
    if os.path.exists(model_params_path):
        try:
            with open(model_params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            if 'embedding_dim' in params:
                embedding_dim = params['embedding_dim']
            if 'hidden_dim' in params:
                hidden_dim = params['hidden_dim']
        except Exception as e:
            print(f"加载模型参数文件失败: {e}")
    
    # 初始化模型
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练好的模型 - 查找最新的匹配参数的模型文件
    runs_dir = 'runs'
    
    # 查找所有best_model相关的文件，包括传统的best_model.pth和新的参数化格式
    model_files = []
    for f in os.listdir(runs_dir):
        if (f.startswith('best_model_') and f.endswith('.pth')) or f == 'best_model.pth':
            model_files.append(f)
    
    if not model_files:
        print("模型文件不存在，请先运行训练脚本！")
        return
    
    # 解析文件名，找到与当前参数匹配的模型
    matched_model = None
    for model_file in model_files:
        try:
            # 解析文件名格式：best_model_<mode>_emb<embedding_dim>_hid<hidden_dim>.pth
            parts = model_file.split('_')
            if len(parts) < 5:
                continue
            
            # 提取参数
            emb_dim = int(parts[-3].replace('emb', ''))
            hid_dim = int(parts[-2].replace('hid', ''))
            
            # 检查参数是否匹配
            if emb_dim == embedding_dim and hid_dim == hidden_dim:
                matched_model = model_file
                break
        except (ValueError, IndexError):
            continue
    
    # 如果没有找到匹配的模型，使用最新的模型
    if not matched_model:
        # 按修改时间排序，选择最新的
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)), reverse=True)
        matched_model = model_files[0]
    
    model_path = os.path.join(runs_dir, matched_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"加载模型成功：{model_path}")
    
    model = model.to(device)
    model.eval()
    
    # 进行预测
    print("开始评估...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            word_ids, tag_ids = batch
            word_ids = word_ids.to(device)
            
            # 获取预测结果
            batch_predictions = []
            for i in range(word_ids.size(0)):
                # 对每个样本进行单独预测
                pred = model(word_ids[i].unsqueeze(0))
                batch_predictions.append(pred)
            
            all_predictions.extend(batch_predictions)
            all_labels.extend(tag_ids.tolist())
    
    # 标签级别评估
    print("\n=== 标签级别评估 ===")
    report, cm = evaluate_ner(all_predictions, all_labels, tag_to_ix)
    print(report)
    
    # 实体级别评估
    print("\n=== 实体级别评估 ===")
    entity_metrics = evaluate_entity_level(all_predictions, all_labels, tag_to_ix)
    
    # 打印实体级别评估结果
    print(f"{'实体类型':<15} {'精确率':<10} {'召回率':<10} {'F1值':<10} {'支持度':<10}")
    print("-" * 55)
    
    for entity_type, metrics in sorted(entity_metrics.items()):
        if entity_type != 'total':
            print(f"{entity_type:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}")
    
    print("-" * 55)
    total = entity_metrics['total']
    print(f"{'总体':<15} {total['precision']:<10.4f} {total['recall']:<10.4f} {total['f1']:<10.4f} {total['support']:<10}")
    
    print("\n评估完成！")

if __name__ == '__main__':
    main()
