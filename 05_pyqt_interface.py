import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QFrame, QSplitter, QScrollArea,
    QMessageBox
)
from PyQt5.QtGui import QFont, QColor, QTextCharFormat, QTextCursor
from PyQt5.QtCore import Qt
import torch

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入以数字开头的模块
import importlib
infer_module = importlib.import_module('04_infer')
NERInference = infer_module.NERInference

class NERInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化NER推理器
        try:
            self.ner_infer = NERInference()
            self.is_model_loaded = True
        except Exception as e:
            self.is_model_loaded = False
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
        
        # 定义实体类型对应的颜色
        self.entity_colors = {
            'Person': QColor(255, 182, 193),  # 浅粉色 - 人物
            'Location': QColor(173, 216, 230),  # 浅蓝色 - 地点
            'Organization': QColor(144, 238, 144),  # 浅绿色 - 组织
            'Work': QColor(255, 215, 0),  # 金色 - 作品
            'Culture': QColor(238, 130, 238),  # 紫罗兰色 - 文化
            'VirtualThings': QColor(255, 165, 0),  # 橙色 - 虚拟事物
            'Other': QColor(220, 220, 220),  # 灰色 - 其他
        }
        
        # 初始化UI
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口标题和大小
        self.setWindowTitle("命名实体识别系统")
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建中心组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标题
        title_label = QLabel("命名实体识别系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建分隔符
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        
        # 创建输入区域
        input_layout = QVBoxLayout()
        input_label = QLabel("输入文本")
        input_label.setFont(QFont("Arial", 12, QFont.Bold))
        input_layout.addWidget(input_label)
        
        # 创建输入文本框
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请输入要识别的文本...")
        self.input_text.setFont(QFont("Arial", 12))
        self.input_text.setMinimumHeight(200)
        input_layout.addWidget(self.input_text)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 创建识别按钮
        self.recognize_btn = QPushButton("识别实体")
        self.recognize_btn.setFont(QFont("Arial", 12))
        self.recognize_btn.setFixedSize(120, 40)
        self.recognize_btn.clicked.connect(self.recognize_entities)
        button_layout.addWidget(self.recognize_btn)
        
        # 创建清空按钮
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setFont(QFont("Arial", 12))
        self.clear_btn.setFixedSize(120, 40)
        self.clear_btn.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_btn)
        
        button_layout.addStretch()
        input_layout.addLayout(button_layout)
        
        main_layout.addLayout(input_layout)
        
        # 创建分隔符
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator2)
        
        # 创建结果区域
        result_layout = QVBoxLayout()
        result_label = QLabel("识别结果")
        result_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout.addWidget(result_label)
        
        # 创建结果文本框
        self.result_text = QTextEdit()
        self.result_text.setFont(QFont("Arial", 12))
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(200)
        result_layout.addWidget(self.result_text)
        
        main_layout.addLayout(result_layout)
        
        # 创建实体类型图例
        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.Box)
        legend_frame.setFrameShadow(QFrame.Sunken)
        legend_layout = QHBoxLayout(legend_frame)
        
        legend_title = QLabel("实体类型:")
        legend_title.setFont(QFont("Arial", 10, QFont.Bold))
        legend_layout.addWidget(legend_title)
        
        for entity_type, color in self.entity_colors.items():
            entity_label = QLabel(f"{entity_type}")
            entity_label.setFont(QFont("Arial", 10))
            
            # 创建颜色标签
            color_label = QLabel()
            color_label.setFixedSize(20, 20)
            color_label.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")
            
            # 创建水平布局
            entity_layout = QHBoxLayout()
            entity_layout.addWidget(color_label)
            entity_layout.addWidget(entity_label)
            entity_layout.setAlignment(Qt.AlignCenter)
            
            # 创建容器
            entity_widget = QWidget()
            entity_widget.setLayout(entity_layout)
            
            legend_layout.addWidget(entity_widget)
        
        legend_layout.addStretch()
        main_layout.addWidget(legend_frame)
        
        # 设置样式
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
            }
            QLabel {
                color: #333;
            }
        """)
    
    def recognize_entities(self):
        """识别实体"""
        if not self.is_model_loaded:
            QMessageBox.critical(self, "错误", "模型未加载，请检查模型文件")
            return
        
        # 获取输入文本
        text = self.input_text.toPlainText()
        
        if not text.strip():
            QMessageBox.warning(self, "警告", "请输入要识别的文本")
            return
        
        # 显示加载状态
        self.recognize_btn.setEnabled(False)
        self.recognize_btn.setText("识别中...")
        QApplication.processEvents()
        
        try:
            # 调用NER推理器进行识别
            result = self.ner_infer.predict(text)
            
            # 显示结果
            self.display_result(result)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别失败: {e}")
        finally:
            # 恢复按钮状态
            self.recognize_btn.setEnabled(True)
            self.recognize_btn.setText("识别实体")
    
    def display_result(self, result):
        """显示识别结果"""
        # 清空结果文本框
        self.result_text.clear()
        
        text = result['text']
        entities = result['entities']
        
        # 创建文本格式
        default_format = QTextCharFormat()
        default_format.setFont(QFont("Arial", 12))
        
        # 将文本添加到结果文本框
        cursor = self.result_text.textCursor()
        cursor.insertText(text, default_format)
        
        # 按实体结束位置降序排序，避免插入高亮时位置偏移
        sorted_entities = sorted(entities, key=lambda x: x['end'], reverse=True)
        
        # 为实体添加高亮
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['type']
            
            # 创建实体高亮格式
            entity_format = QTextCharFormat()
            entity_format.setBackground(self.entity_colors.get(entity_type, QColor(220, 220, 220)))
            entity_format.setFontWeight(QFont.Bold)
            
            # 设置光标位置并应用格式
            cursor = self.result_text.textCursor()
            cursor.setPosition(start)
            cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, end - start)
            cursor.mergeCharFormat(entity_format)
    
    def clear_text(self):
        """清空输入和输出文本"""
        self.input_text.clear()
        self.result_text.clear()

# 主函数
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion样式
    
    # 设置应用程序信息
    app.setApplicationName("命名实体识别系统")
    app.setApplicationVersion("1.0")
    
    window = NERInterface()
    window.show()
    
    sys.exit(app.exec_())
