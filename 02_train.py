import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from model import BiLSTM_CRF
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置随机种子，保证实验结果的可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 数据加载器
class NERDataset(Dataset):
    def __init__(self, data_file, word_to_ix=None, tag_to_ix=None, max_len=128):
        self.data = self.load_data(data_file)
        self.max_len = max_len
        
        # 如果没有提供词汇表，则构建词汇表
        if word_to_ix is None:
            self.word_to_ix = self.build_vocab()
        else:
            self.word_to_ix = word_to_ix
        
        # 如果没有提供标签映射，则构建标签映射
        if tag_to_ix is None:
            self.tag_to_ix = self.build_tag_map()
        else:
            self.tag_to_ix = tag_to_ix

    def load_data(self, data_file):
        """加载BIO格式的数据，确保字符和标签对应，增强对格式问题的处理"""
        data = []
        sentence = []
        tags = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 移除所有不可见字符，只保留可见字符
                line = ''.join(char for char in line if char.isprintable() or char == '\t')
                line = line.strip()
                
                if not line:
                    # 句子结束
                    if sentence and tags:
                        # 确保句子和标签长度一致
                        if len(sentence) != len(tags):
                            print(f"警告：第{line_num}行附近的句子中字符和标签数量不匹配，已跳过")
                        else:
                            data.append((sentence, tags))
                        sentence = []
                        tags = []
                    continue
                
                # 使用split()分割，它能处理任意数量的空格和制表符
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    tag = parts[1]
                    sentence.append(word)
                    tags.append(tag)
        
        # 处理最后一个句子
        if sentence and tags:
            if len(sentence) != len(tags):
                print(f"警告：文件末尾的句子中字符和标签数量不匹配，已跳过")
            else:
                data.append((sentence, tags))
            
        return data

    def build_vocab(self):
        """构建词汇表"""
        word_to_ix = {
            '<pad>': 0,  # 填充字符
            '<unk>': 1   # 未知字符
        }
        
        for sentence, tags in self.data:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        
        return word_to_ix

    def build_tag_map(self):
        """构建标签映射"""
        tag_to_ix = {}
        
        for sentence, tags in self.data:
            for tag in tags:
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
        
        return tag_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """获取一个样本，转换为张量格式"""
        sentence, tags = self.data[idx]
        
        # 将句子转换为索引
        word_ids = [self.word_to_ix.get(word, self.word_to_ix['<unk>']) for word in sentence]
        # 将标签转换为索引，对于未见过的标签，使用'O'标签
        tag_ids = [self.tag_to_ix.get(tag, self.tag_to_ix.get('O', 0)) for tag in tags]
        
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

# 计算准确率
def calculate_accuracy(predictions, labels, mask=None):
    """计算模型的准确率"""
    correct = 0
    total = 0
    
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            if mask is None or (isinstance(mask[i][j], torch.Tensor) and mask[i][j].item() == 1) or mask[i][j] == 1:
                if isinstance(predictions[i][j], torch.Tensor):
                    pred_val = predictions[i][j].item()
                else:
                    pred_val = predictions[i][j]
                    
                if isinstance(labels[i][j], torch.Tensor):
                    label_val = labels[i][j].item()
                else:
                    label_val = labels[i][j]
                    
                if pred_val == label_val:
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0

# 验证函数
def evaluate(model, val_loader, device):
    """评估模型在验证集上的性能"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            word_ids, tag_ids = batch
            word_ids = word_ids.to(device)
            tag_ids = tag_ids.to(device)
            
            # 计算损失
            loss = model(word_ids, tag_ids)
            total_loss += loss.item()
            
            # 获取预测结果
            # 对整个批次进行预测，得到最佳路径
            best_paths = model(word_ids)
            # 将预测结果转换为列表并添加到all_predictions中
            predictions = best_paths.tolist()
            
            all_predictions.extend(predictions)
            all_labels.extend(tag_ids.tolist())
    
    # 计算准确率
    accuracy = calculate_accuracy(all_predictions, all_labels)
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='NER模型训练脚本')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'quick'],
                        help='训练模式: standard(标准训练), quick(快速训练)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮次')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')
    
    args = parser.parse_args()
    
    # 根据训练模式设置超参数
    if args.mode == 'quick':
        # 快速训练参数
        embedding_dim = 64
        hidden_dim = 128
        batch_size = 128
        epochs = 5
        learning_rate = args.learning_rate
        max_len = args.max_len
        print("使用快速训练模式")
    else:
        # 标准训练参数
        embedding_dim = args.embedding_dim
        hidden_dim = args.hidden_dim
        batch_size = args.batch_size
        epochs = args.epochs
        learning_rate = args.learning_rate
        max_len = args.max_len
        print("使用标准训练模式")
    
    # 数据集路径
    train_file = os.path.join('dataset', 'train.txt')
    val_file = os.path.join('dataset', 'val.txt')
    
    # 检查数据集是否存在
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print("数据集不存在，请先运行数据预处理脚本！")
        return
    
    # 检查模型保存目录
    runs_dir = 'runs'
    os.makedirs(runs_dir, exist_ok=True)
    
    # 尝试加载已有词汇表和标签映射（增量训练模式或用户指定）
    vocab_path = os.path.join(runs_dir, 'vocab.json')
    tag_map_path = os.path.join(runs_dir, 'tag_map.json')
    
    use_existing_vocab = args.mode == 'incremental' and os.path.exists(vocab_path) and os.path.exists(tag_map_path)
    
    if use_existing_vocab:
        print("加载已有词汇表和标签映射...")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            word_to_ix = json.load(f)
        
        with open(tag_map_path, 'r', encoding='utf-8') as f:
            tag_to_ix = json.load(f)
        
        # 加载数据集，使用已有词汇表和标签映射
        train_dataset = NERDataset(train_file, word_to_ix=word_to_ix, tag_to_ix=tag_to_ix, max_len=max_len)
        val_dataset = NERDataset(val_file, word_to_ix=word_to_ix, tag_to_ix=tag_to_ix, max_len=max_len)
    else:
        print("加载数据集...")
        # 加载数据集，构建新的词汇表和标签映射
        train_dataset = NERDataset(train_file, max_len=max_len)
        val_dataset = NERDataset(val_file, word_to_ix=train_dataset.word_to_ix, 
                                tag_to_ix=train_dataset.tag_to_ix, max_len=max_len)
        
        # 保存词汇表和标签映射
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset.word_to_ix, f, ensure_ascii=False, indent=2)
        
        with open(tag_map_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset.tag_to_ix, f, ensure_ascii=False, indent=2)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"词汇表大小: {len(train_dataset.word_to_ix)}")
    print(f"标签数量: {len(train_dataset.tag_to_ix)}")
    
    # 初始化模型
    vocab_size = len(train_dataset.word_to_ix)
    tag_to_ix = train_dataset.tag_to_ix
    
    model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"使用设备: {device}")
    
    # 尝试加载已有模型进行增量训练
    start_epoch = 0
    
    # 尝试查找对应参数配置的模型（无论之前的训练模式是什么）
    print("\n寻找可用于增量训练的模型...")
    
    # 遍历runs目录下所有模型文件，寻找匹配当前参数配置的模型
    found_model = False
    for filename in os.listdir(runs_dir):
        if filename.startswith('best_model_') and filename.endswith('.pth'):
            # 提取模型文件名中的参数信息
            parts = filename.replace('best_model_', '').replace('.pth', '').split('_')
            if len(parts) >= 4 and parts[-2].startswith('emb') and parts[-1].startswith('hid'):
                try:
                    # 解析嵌入维度和隐藏层维度
                    emb_dim = int(parts[-2].replace('emb', ''))
                    hid_dim = int(parts[-1].replace('hid', ''))
                    
                    # 检查参数是否匹配
                    if emb_dim == embedding_dim and hid_dim == hidden_dim:
                        # 找到匹配的模型
                        model_path = os.path.join(runs_dir, filename)
                        # 对应的参数文件
                        params_filename = filename.replace('best_model_', 'model_params_').replace('.pth', '.json')
                        model_params_path = os.path.join(runs_dir, params_filename)
                        
                        print(f"找到匹配的模型文件: {filename}")
                        found_model = True
                        break
                except (ValueError, IndexError):
                    continue
    
    if not found_model:
        print("未发现匹配参数的模型，将从头开始训练...")
        # 设置默认的模型保存路径
        model_filename = f'best_model_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.pth'
        model_path = os.path.join(runs_dir, model_filename)
        model_params_filename = f'model_params_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.json'
        model_params_path = os.path.join(runs_dir, model_params_filename)
    else:
        # 加载找到的模型进行增量训练
        try:
            # 检查模型参数是否匹配
            if os.path.exists(model_params_path):
                with open(model_params_path, 'r', encoding='utf-8') as f:
                    saved_params = json.load(f)
                
                # 验证关键参数是否匹配
                if (saved_params['embedding_dim'] != embedding_dim or 
                    saved_params['hidden_dim'] != hidden_dim or
                    saved_params['vocab_size'] != vocab_size or
                    saved_params['tagset_size'] != model.tagset_size):
                    print("警告：加载的模型参数与当前配置不匹配！")
                    print(f"已保存模型: embedding_dim={saved_params['embedding_dim']}, hidden_dim={saved_params['hidden_dim']}")
                    print(f"当前配置: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
                    print("将从头开始训练")
                    model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
                    model = model.to(device)
                    # 设置默认的模型保存路径
                    model_filename = f'best_model_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.pth'
                    model_path = os.path.join(runs_dir, model_filename)
                    model_params_filename = f'model_params_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.json'
                    model_params_path = os.path.join(runs_dir, model_params_filename)
                else:
                    # 加载模型
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"成功加载模型: {model_path}")
                    print("将在已有模型基础上继续训练")
                    # 加载起始epoch（如果有保存）
                    if 'last_epoch' in saved_params:
                        start_epoch = saved_params['last_epoch']
                        print(f"从epoch {start_epoch + 1}开始继续训练")
            else:
                # 没有参数文件，直接尝试加载模型
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"成功加载模型: {model_path}")
                print("将在已有模型基础上继续训练")
                print("注意：没有找到模型参数文件，无法验证参数匹配性")
                # 设置默认的模型保存路径
                model_filename = f'best_model_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.pth'
                model_path = os.path.join(runs_dir, model_filename)
                model_params_filename = f'model_params_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.json'
                model_params_path = os.path.join(runs_dir, model_params_filename)
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将从头开始训练")
            # 设置默认的模型保存路径
            model_filename = f'best_model_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.pth'
            model_path = os.path.join(runs_dir, model_filename)
            model_params_filename = f'model_params_{args.mode}_emb{embedding_dim}_hid{hidden_dim}.json'
            model_params_path = os.path.join(runs_dir, model_params_filename)

    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_val_loss = float('inf')
    
    # 创建损失值记录列表
    loss_history = []  # 记录所有批次的损失值
    batch_interval = 10  # 每隔10个批次记录一次损失值
    
    print(f"\n开始训练...")
    print(f"训练配置: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, batch_size={batch_size}, epochs={epochs}")
    
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch + 1
        total_epochs = start_epoch + epochs
        
        start_time = time.time()
        model.train()
        train_loss = 0
        
        # 使用tqdm显示训练进度
        with tqdm(total=len(train_loader), desc=f"Epoch {current_epoch}/{total_epochs}", unit="batch") as pbar:
            for batch in train_loader:
                word_ids, tag_ids = batch
                word_ids = word_ids.to(device)
                tag_ids = tag_ids.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 计算损失
                loss = model(word_ids, tag_ids)
                train_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                # 记录损失值（每隔batch_interval个批次）
                if (pbar.n % batch_interval == 0) or (pbar.n == len(train_loader)):
                    loss_history.append((current_epoch, pbar.n, loss.item()))
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证模型
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        # 保存最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            
            # 保存模型参数和训练状态
            model_params = {
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'vocab_size': vocab_size,
                'tagset_size': model.tagset_size,
                'last_epoch': current_epoch,
                'best_val_loss': best_val_loss
            }
            with open(model_params_path, 'w', encoding='utf-8') as f:
                json.dump(model_params, f, ensure_ascii=False, indent=2)
            
            print(f"保存最佳模型到: {model_path}")
        
        # 打印训练信息
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {current_epoch}/{total_epochs}")
        print(f"训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"耗时: {epoch_time:.2f}秒")
        print("-" * 50)
    
    print("训练完成！")
    print(f"最终模型已保存到: {model_path}")
    
    # 绘制损失值变化曲线
    if loss_history:
        # 提取数据
        epochs, batch_nums, losses = zip(*loss_history)
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(losses)), losses, 'b-', label='Training Loss')
        
        # 设置标题和标签
        plt.title('Training Loss Curve')
        plt.xlabel(f'Batch Number (interval: {batch_interval})')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        loss_curve_path = os.path.join(runs_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()
        
        print(f"\n损失值变化曲线已保存到: {loss_curve_path}")
    
    if args.mode == 'quick':
        print("注意：由于使用了快速训练参数，模型性能可能会有所下降。如需更好的性能，请使用标准训练模式。")

if __name__ == '__main__':
    main()
