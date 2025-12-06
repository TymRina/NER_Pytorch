# 提取的损失值相关核心代码
# 1. CRF层的损失计算核心逻辑（来自model.py）

import torch
import torch.nn as nn
from torch.autograd import Variable

def log_sum_exp(vec):
    """计算向量的log-sum-exp，保证数值稳定性"""
    if vec.dim() == 1:
        max_score = torch.max(vec)
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score)))
    elif vec.dim() == 2:
        max_score, _ = torch.max(vec, dim=1, keepdim=True)
        return max_score.squeeze(1) + torch.log(torch.sum(torch.exp(vec - max_score), dim=1))
    else:
        raise ValueError("输入向量的维度必须是1或2")

class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        # 转移矩阵，transitions[i][j]表示从标签i转移到标签j的分数
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        # 填充位置的标签，避免影响损失计算
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        # 初始化转移矩阵的参数，确保合理性
        # 这里设置了-10000来禁止某些转移，这就是用户看到大负损失值的来源之一
        self.transitions.data[self.end_tag, :] = -10000  # 结束标签不能转移到任何标签
        self.transitions.data[:, self.start_tag] = -10000  # 任何标签不能转移到开始标签

    def _forward_alg(self, feats):
        """前向算法计算所有路径的分数总和"""
        seq_len, batch_size, num_tags = feats.shape
        device = feats.device
        
        # 初始化为负无穷
        init_alphas = torch.full((batch_size, num_tags), -10000., device=device)
        forward_var = init_alphas
        
        for i in range(seq_len):
            feat = feats[i]  # 形状: [batch_size, num_tags]
            alphas_t = torch.zeros((batch_size, num_tags), device=device)
            
            for next_tag in range(num_tags):
                emit_score = feat[:, next_tag]
                trans_score = self.transitions[:num_tags, next_tag]
                trans_score = trans_score.unsqueeze(0).expand(batch_size, -1)
                
                # 当前路径的总分数
                next_tag_var = forward_var + trans_score + emit_score.unsqueeze(1)
                alphas_t[:, next_tag] = log_sum_exp(next_tag_var)
            
            forward_var = alphas_t
        
        # 计算结束位置的分数
        trans_score = self.transitions[:num_tags, self.end_tag]
        trans_score = trans_score.unsqueeze(0).expand(batch_size, -1)
        terminal_var = forward_var + trans_score
        
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        """计算给定标签序列的分数"""
        seq_len, batch_size, num_tags = feats.shape
        device = feats.device
        
        start_tags = torch.full((batch_size,), self.start_tag, dtype=torch.long, device=device)
        
        # 计算起始标签到第一个标签的转移分数
        trans_score = self.transitions[start_tags, tags[0]]
        emit_score = feats[0, range(batch_size), tags[0]]
        
        # 初始化总分数
        score = trans_score + emit_score
        
        # 计算后续标签的转移和发射分数
        for i in range(1, seq_len):
            trans_score = self.transitions[tags[i-1], tags[i]]
            emit_score = feats[i, range(batch_size), tags[i]]
            score += trans_score + emit_score
        
        # 添加结束标签的转移分数
        end_tags = torch.full((batch_size,), self.end_tag, dtype=torch.long, device=device)
        score += self.transitions[tags[-1], end_tags]
        
        return score

    def forward(self, feats, tags=None, mask=None):
        """前向传播
        如果提供了tags，返回负对数似然损失；否则返回viterbi解码结果
        """
        if tags is not None:
            # 计算所有可能路径的分数总和
            forward_score = self._forward_alg(feats)
            # 计算真实路径的分数
            gold_score = self._score_sentence(feats, tags)
            # 返回负对数似然损失（批次平均）
            # 这里的损失计算是导致大负值的核心：forward_score - gold_score
            return torch.mean(forward_score - gold_score)
        else:
            # 预测时使用viterbi解码
            return self._viterbi_decode(feats)

# 2. BiLSTM_CRF模型中的损失计算（来自model.py）

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=128, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # BiLSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        # 线性层将BiLSTM的输出映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # CRF层
        self.crf = CRF(self.tagset_size)
        
    def forward(self, sentence, tags=None):
        """前向传播
        Args:
            sentence: 输入句子，形状为[batch_size, seq_len]
            tags: 真实标签，形状为[batch_size, seq_len]
        Returns:
            如果提供了tags，返回损失；否则返回预测标签
        """
        # 词嵌入
        embeds = self.word_embeds(sentence)
        # BiLSTM输出
        lstm_out, _ = self.lstm(embeds)
        # 映射到标签空间
        lstm_feats = self.hidden2tag(lstm_out)
        
        # 转换为CRF期望的形状: [seq_len, batch_size, num_tags]
        lstm_feats = lstm_feats.permute(1, 0, 2)
        
        if tags is not None:
            # 转换标签形状为CRF期望的形状: [seq_len, batch_size]
            tags = tags.permute(1, 0)
            # 计算CRF损失
            loss = self.crf(lstm_feats, tags)
            return loss
        else:
            # 预测标签
            score, tag_seq = self.crf._viterbi_decode(lstm_feats)
            return tag_seq

# 3. 训练循环中的损失处理（来自02_train.py）

"""
# 在训练循环中的损失计算和处理
for batch in train_loader:
    word_ids, tag_ids = batch
    word_ids = word_ids.to(device)
    tag_ids = tag_ids.to(device)
    
    # 梯度清零
    optimizer.zero_grad()
    
    # 计算损失：调用model的forward方法，返回CRF损失
    loss = model(word_ids, tag_ids)
    train_loss += loss.item()
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    optimizer.step()
    
    # 更新进度条显示当前损失值
    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
"""

# 关键说明：
# 1. 损失值为负的原因：CRF的损失计算为 forward_score - gold_score，而forward_score通常大于gold_score
# 2. 损失值绝对值大的原因：
#    - 转移矩阵初始化时使用了-10000来禁止无效转移
#    - 序列长度较长，得分会随序列长度累积
#    - 训练初期参数随机初始化，得分差异较大
# 3. 训练过程中损失值会逐渐向0靠近，绝对值减小