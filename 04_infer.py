import torch
import os
import json
import re
from model import BiLSTM_CRF

class NERInference:
    def __init__(self, model_path=None, vocab_path=None, tag_map_path=None):
        """初始化NER推理器
        Args:
            model_path: 模型文件路径
            vocab_path: 词汇表路径
            tag_map_path: 标签映射路径
        """
        # 获取当前脚本所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置默认路径
        if vocab_path is None:
            vocab_path = os.path.join(current_dir, 'runs', 'vocab.json')
        if tag_map_path is None:
            tag_map_path = os.path.join(current_dir, 'runs', 'tag_map.json')
        
        # 如果没有指定模型路径，自动查找最新的或匹配参数的模型文件
        if model_path is None:
            runs_dir = os.path.join(current_dir, 'runs')
            
            # 查找所有best_model相关的文件，包括传统的best_model.pth和新的参数化格式
            model_files = []
            for f in os.listdir(runs_dir):
                if (f.startswith('best_model_') and f.endswith('.pth')) or f == 'best_model.pth':
                    model_files.append(f)
            
            if not model_files:
                raise FileNotFoundError("模型文件不存在，请先运行训练脚本！")
            
            # 按修改时间排序，选择最新的
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)), reverse=True)
            model_path = os.path.join(runs_dir, model_files[0])
        
        # 加载词汇表和标签映射
        self.word_to_ix = self.load_vocab(vocab_path)
        self.tag_to_ix = self.load_tag_map(tag_map_path)
        self.ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}
        
        # 初始化模型
        self.embedding_dim = 128  # 默认值
        self.hidden_dim = 256     # 默认值
        
        # 尝试从模型参数文件加载参数
        model_params_path = os.path.join(current_dir, 'runs', 'model_params.json')
        if os.path.exists(model_params_path):
            try:
                with open(model_params_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                if 'embedding_dim' in params:
                    self.embedding_dim = params['embedding_dim']
                if 'hidden_dim' in params:
                    self.hidden_dim = params['hidden_dim']
            except Exception as e:
                print(f"加载模型参数文件失败: {e}")
        
        self.model = BiLSTM_CRF(len(self.word_to_ix), self.tag_to_ix, 
                               embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        print(f"NER推理器初始化完成")
        print(f"使用设备: {self.device}")
        print(f"模型路径: {model_path}")
        print(f"词汇表大小: {len(self.word_to_ix)}")
        print(f"标签数量: {len(self.tag_to_ix)}")
    
    def load_vocab(self, vocab_path):
        """加载词汇表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            word_to_ix = json.load(f)
        return word_to_ix
    
    def load_tag_map(self, tag_map_path):
        """加载标签映射"""
        with open(tag_map_path, 'r', encoding='utf-8') as f:
            tag_to_ix = json.load(f)
        return tag_to_ix
    
    def preprocess_text(self, text):
        """文本预处理
        Args:
            text: 输入文本
        Returns:
            processed_text: 处理后的文本
        """
        # 只去除多余的空格，保留换行符等其他空白字符
        # 将连续的空格替换为单个空格，但保留换行符
        import re
        # 处理连续空格，但保留换行符
        text = re.sub(r'[ ]+', ' ', text)
        # 去除文本首尾的空格
        return text.strip()
    
    def split_long_text(self, text, max_len=128):
        """将长文本分割成多个短文本
        Args:
            text: 输入文本
            max_len: 最大长度
        Returns:
            chunks: 分割后的文本列表
        """
        chunks = []
        for i in range(0, len(text), max_len):
            chunks.append(text[i:i+max_len])
        return chunks
    
    def text_to_ids(self, text):
        """将文本转换为索引序列
        Args:
            text: 输入文本
        Returns:
            ids: 索引序列
        """
        return [self.word_to_ix.get(char, self.word_to_ix['<unk>']) for char in text]
    
    def predict(self, text):
        """预测文本中的实体
        Args:
            text: 输入文本
        Returns:
            result: 包含原始文本和实体列表的字典
        """
        # 使用原始文本进行预测
        original_text = text
        
        # 长文本分割
        # 注意：这里我们不使用预处理函数，直接使用原始文本进行分割
        # 只在分割后对每个chunk进行必要的处理
        text_chunks = self.split_long_text(text)
        
        all_entities = []
        original_offset = 0  # 原始文本的偏移量
        
        for chunk in text_chunks:
            # 解决标签错位和末尾字符标签丢失问题：
            # 在文本末尾添加一个占位符字符（数字0），这样当我们移动标签时，原始文本的最后一个字符的标签就不会丢失
            chunk_with_placeholder = chunk + '0'
            
            # 转换为索引序列
            word_ids = self.text_to_ids(chunk_with_placeholder)
            word_ids_tensor = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # 进行预测
            with torch.no_grad():
                pred_tags = self.model(word_ids_tensor)
            
            # 将预测结果转换为1D张量并移到CPU
            pred_tags = pred_tags[0].cpu().numpy()
            
            # 转换为标签名称
            pred_tag_names = [self.ix_to_tag[tag_id] for tag_id in pred_tags]
            
            # 确保标签序列与原始文本长度一致并解决错位问题
            # 用户反映标签整体错位一位，我们需要将标签序列向前移动一位
            text_len = len(chunk)  # 原始文本长度（不含占位符）
            tag_len = len(pred_tag_names)
            
            if tag_len >= text_len + 1:
                # 由于添加了占位符，标签序列应该至少比原始文本长1
                # 将标签序列向前移动一位，然后截取到原始文本的长度
                pred_tag_names = pred_tag_names[1:text_len+1]
            elif text_len == tag_len:
                # 意外情况：标签序列与原始文本长度相同（没有占位符的效果）
                # 将整个序列向前移动一位，并用'O'填充最后一位
                pred_tag_names = pred_tag_names[1:] + ['O']
            else:
                # 标签序列比文本短，补充'O'标签
                pred_tag_names = pred_tag_names[1:] + ['O'] * (text_len - tag_len + 1)
            
            # 提取实体（基于原始文本的chunk）
            chunk_entities = self.extract_entities(chunk, pred_tag_names)
            
            # 调整实体位置偏移（基于原始文本）
            for entity in chunk_entities:
                entity['start'] += original_offset
                entity['end'] += original_offset
            
            all_entities.extend(chunk_entities)
            original_offset += len(chunk)
        
        return {
            'text': text,  # 返回原始文本，保留标点符号
            'entities': all_entities
        }
    
    def extract_entities(self, text, tags):
        """从标签序列中提取实体
        Args:
            text: 输入文本
            tags: 标签序列
        Returns:
            entities: 实体列表
        """
        entities = []
        current_entity = None
        
        for i, (char, tag) in enumerate(zip(text, tags)):
            if tag.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = {
                    'text': char,
                    'type': entity_type,
                    'start': i,
                    'end': i + 1
                }
            elif tag.startswith('I-'):
                # 继续当前实体
                if current_entity:
                    entity_type = tag[2:]
                    if entity_type == current_entity['type']:
                        # 继续当前实体
                        current_entity['text'] += char
                        current_entity['end'] = i + 1
                    else:
                        # I-标签与当前实体类型不一致，按BIO规范应该忽略或作为新实体
                        # 这里选择忽略，继续当前实体
                        pass
                else:
                    # 没有前导B-标签的I-标签，忽略
                    pass
            else:
                # 非实体或实体结束
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        # 后处理：移除实体末尾的标点符号
        # 同时处理中英文标点符号
        punctuation = '.,!?;。，！？；'
        for entity in entities:
            # 检查实体文本末尾是否为标点符号
            while entity['text'] and entity['text'][-1] in punctuation:
                entity['text'] = entity['text'][:-1]
                entity['end'] -= 1  # 调整实体结束位置
            # 如果实体文本被完全移除（不太可能，但为了安全）
            if not entity['text']:
                entities.remove(entity)
        
        return entities
    
    def format_result(self, result):
        """格式化输出结果
        Args:
            result: 实体识别结果
        Returns:
            formatted_result: 格式化后的结果字符串
        """
        text = result['text']
        entities = result['entities']
        
        # 如果没有识别到实体
        if not entities:
            return f"文本: {text}\n实体: 无"
        
        # 构建高亮文本
        highlight_text = []
        last_end = 0
        
        # 按实体开始位置排序
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            
            # 添加实体前的文本
            highlight_text.append(text[last_end:start])
            # 添加实体（带标签）
            entity_text = text[start:end]
            highlight_text.append(f"[{entity_text}]({entity['type']})")
            
            last_end = end
        
        # 添加最后一个实体后的文本
        highlight_text.append(text[last_end:])
        
        # 构建结果字符串
        result_str = f"文本: {''.join(highlight_text)}\n"
        result_str += "实体列表:\n"
        
        for entity in sorted_entities:
            result_str += f"- {entity['text']}: {entity['type']} (位置: {entity['start']}-{entity['end']})\n"
        
        return result_str

# 命令行测试
def main():
    # 创建推理器
    ner_infer = NERInference()
    
    print("\n=== 命名实体识别系统 ===")
    print("输入'quit'退出程序")
    
    while True:
        try:
            # 获取用户输入
            text = input("\n请输入要识别的文本: ")
            
            if text.lower() == 'quit':
                print("退出程序")
                break
            
            # 进行预测
            result = ner_infer.predict(text)
            
            # 输出结果
            formatted_result = ner_infer.format_result(result)
            print(formatted_result)
            
        except Exception as e:
            print(f"错误: {e}")
            continue

if __name__ == '__main__':
    main()
