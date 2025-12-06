import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# CRF层实现
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        # 转移矩阵，transitions[i][j]表示从标签i转移到标签j的分数
        # 使用较小的初始值，提高数值稳定性
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2) * 0.1)
        # 填充位置的标签，避免影响损失计算
        self.start_tag = num_tags
        self.end_tag = num_tags + 1
        # 初始化转移矩阵的参数，确保合理性
        # 使用合理的负值禁止无效转移，避免数值溢出
        self.transitions.data[self.end_tag, :] = -10  # 结束标签不能转移到任何标签
        self.transitions.data[:, self.start_tag] = -10  # 任何标签不能转移到开始标签

    def _forward_alg(self, feats):
        """前向算法计算所有路径的分数总和
        Args:
            feats: 模型的输出特征，形状为[seq_len, batch_size, num_tags]
        Returns:
            alpha: 所有路径的分数总和
        """
        seq_len, batch_size, num_tags = feats.shape
        
        # 确保使用与输入特征相同的设备
        device = feats.device
        
        # 初始化为合理的负值，与转移矩阵的禁止值保持一致，避免数值问题
        init_alphas = torch.full((batch_size, num_tags), -10., device=device)
        
        # 使用动态规划计算前向传播
        forward_var = init_alphas
        
        for i in range(seq_len):
            feat = feats[i]  # 形状: [batch_size, num_tags]
            alphas_t = torch.zeros((batch_size, num_tags), device=device)
            
            for next_tag in range(num_tags):
                # 当前标签的发射分数，形状: [batch_size]
                emit_score = feat[:, next_tag]
                
                # 从所有可能的当前标签到下一个标签的转移分数，形状: [num_tags]
                trans_score = self.transitions[:num_tags, next_tag]
                
                # 广播转移分数到batch_size，形状: [batch_size, num_tags]
                trans_score = trans_score.unsqueeze(0).expand(batch_size, -1)
                
                # 当前路径的总分数，形状: [batch_size, num_tags]
                next_tag_var = forward_var + trans_score + emit_score.unsqueeze(1)
                
                # 计算log-sum-exp，形状: [batch_size]
                alphas_t[:, next_tag] = log_sum_exp(next_tag_var)
            
            forward_var = alphas_t
        
        # 计算结束位置的分数
        trans_score = self.transitions[:num_tags, self.end_tag]
        trans_score = trans_score.unsqueeze(0).expand(batch_size, -1)
        terminal_var = forward_var + trans_score
        
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        """计算给定标签序列的分数
        Args:
            feats: 模型的输出特征，形状为[seq_len, batch_size, num_tags]
            tags: 真实标签，形状为[seq_len, batch_size]
        Returns:
            score: 真实路径的分数
        """
        seq_len, batch_size, num_tags = feats.shape
        
        # 确保使用与输入特征相同的设备
        device = feats.device
        
        # 创建起始标签张量
        start_tags = torch.full((batch_size,), self.start_tag, dtype=torch.long, device=device)
        
        # 计算起始标签到第一个标签的转移分数
        trans_score = self.transitions[start_tags, tags[0]]
        
        # 计算第一个标签的发射分数
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

    def _viterbi_decode(self, feats):
        """使用Viterbi算法找到最优路径
        Args:
            feats: 模型的输出特征，形状为[seq_len, batch_size, num_tags]
        Returns:
            path_score: 最优路径的分数
            best_path: 最优路径，形状为[batch_size, seq_len]
        """
        seq_len, batch_size, num_tags = feats.shape
        
        # 确保使用与输入特征相同的设备
        device = feats.device
        
        # 初始化viterbi变量和回溯指针
        backpointers = []
        
        # 初始化viterbi变量，形状: [batch_size, num_tags]
        init_vvars = torch.full((batch_size, num_tags), -10000., device=device)
        
        forward_var = init_vvars
        
        for i in range(seq_len):
            feat = feats[i]  # 形状: [batch_size, num_tags]
            bptrs_t = torch.zeros((batch_size, num_tags), dtype=torch.long, device=device)
            viterbivars_t = torch.zeros((batch_size, num_tags), device=device)
            
            for next_tag in range(num_tags):
                # 从所有可能的当前标签到下一个标签的转移分数，形状: [num_tags]
                trans_score = self.transitions[:num_tags, next_tag]
                
                # 广播转移分数到batch_size，形状: [batch_size, num_tags]
                trans_score = trans_score.unsqueeze(0).expand(batch_size, -1)
                
                # 当前路径的总分数，形状: [batch_size, num_tags]
                next_tag_var = forward_var + trans_score
                
                # 找到最优的前一个标签，形状: [batch_size]
                best_tag_ids = torch.argmax(next_tag_var, dim=1)
                bptrs_t[:, next_tag] = best_tag_ids
                
                # 保存最优路径的分数
                viterbivars_t[:, next_tag] = next_tag_var[range(batch_size), best_tag_ids]
            
            # 添加当前时间步的发射分数
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)
        
        # 计算结束位置的分数
        trans_score = self.transitions[:num_tags, self.end_tag]
        trans_score = trans_score.unsqueeze(0).expand(batch_size, -1)
        terminal_var = forward_var + trans_score
        
        # 找到最后一个时间步的最优标签
        best_tag_ids = torch.argmax(terminal_var, dim=1)
        path_scores = terminal_var[range(batch_size), best_tag_ids]
        
        # 回溯找到最优路径
        best_paths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        best_paths[:, -1] = best_tag_ids
        
        for i in range(seq_len-2, -1, -1):
            bptrs_t = backpointers[i]
            best_tag_ids = bptrs_t[range(batch_size), best_tag_ids]
            best_paths[:, i] = best_tag_ids
        
        return path_scores, best_paths

    def forward(self, feats, tags=None, mask=None):
        """前向传播
        Args:
            feats: 模型的输出特征，形状为[seq_len, batch_size, num_tags]
            tags: 真实标签，形状为[seq_len, batch_size]
            mask: 掩码，形状为[seq_len, batch_size]，用于处理变长序列
        Returns:
            如果提供了tags，返回负对数似然损失；否则返回viterbi解码结果
        """
        if tags is not None:
            # 计算所有可能路径的分数总和
            forward_score = self._forward_alg(feats)
            # 计算真实路径的分数
            gold_score = self._score_sentence(feats, tags)
            # 返回负对数似然损失（批次平均）
            return torch.mean(forward_score - gold_score)
        else:
            # 预测时使用viterbi解码
            return self._viterbi_decode(feats)

# 辅助函数
def log_sum_exp(vec):
    """计算向量的log-sum-exp
    Args:
        vec: 输入向量，可以是1D或2D张量
            - 1D: [num_tags]
            - 2D: [batch_size, num_tags]
    Returns:
        log_sum_exp结果：
            - 1D输入返回标量
            - 2D输入返回[batch_size]
    """
    if vec.dim() == 1:
        max_score = torch.max(vec)
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score)))
    elif vec.dim() == 2:
        max_score, _ = torch.max(vec, dim=1, keepdim=True)
        return max_score.squeeze(1) + torch.log(torch.sum(torch.exp(vec - max_score), dim=1))
    else:
        raise ValueError("输入向量的维度必须是1或2")

# BiLSTM-CRF模型
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

    def init_hidden(self):
        """初始化LSTM的隐藏状态"""
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
