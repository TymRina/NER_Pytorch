import os
import subprocess
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 测试脚本列表
test_scripts = [
    '01_data_preprocess.py',
    '04_infer.py'
]

def run_test(script_path):
    """运行测试脚本并返回结果"""
    print(f"\n测试脚本: {script_path}")
    print("=" * 50)
    
    try:
        if script_path == '04_infer.py':
            # 创建一个简单的测试脚本，直接测试路径存在性
            test_content = '''
import os
import sys

try:
    # 直接测试04_infer.py中使用的相对路径是否存在
    required_paths = [
        os.path.join('runs', 'vocab.json'),
        os.path.join('runs', 'tag_map.json'),
        os.path.join('runs', 'model_params.json')
    ]
    
    all_exist = True
    for path in required_paths:
        if not os.path.exists(path):
            print(f"路径不存在: {path}")
            all_exist = False
    
    # 检查模型文件
    model_files = [f for f in os.listdir('runs') if (f.startswith('best_model_') and f.endswith('.pth')) or f == 'best_model.pth']
    if not model_files:
        print("未找到模型文件")
        all_exist = False
    else:
        print(f"找到模型文件: {model_files[0]}")
    
    if all_exist:
        print("路径配置正确")
        sys.exit(0)
    else:
        print("路径配置存在问题")
        sys.exit(1)

except Exception as e:
    print(f"测试失败: {e}")
    sys.exit(1)
'''
            
            test_script_path = os.path.join(current_dir, "test_infer_paths.py")
            
            # 写入临时测试脚本
            with open(test_script_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # 运行临时测试脚本
            result = subprocess.run(
                [sys.executable, "test_infer_paths.py"],
                cwd=current_dir,
                timeout=10,
                capture_output=True,
                text=True
            )
            
            # 清理临时文件
            if os.path.exists(test_script_path):
                os.remove(test_script_path)
        else:
            # 正常运行其他测试脚本
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=current_dir,
                timeout=60,  # 数据预处理需要较长时间
                capture_output=True,
                text=True
            )
        
        # 检查测试结果
        if result.returncode == 0:
            print("✅ 测试通过")
            print(f"输出: {result.stdout[:200]}...")
            return True
        else:
            print("❌ 测试失败")
            print(f"错误信息: {result.stderr[:200]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

# 主函数
def main():
    print("测试相对路径配置")
    print(f"当前脚本目录: {current_dir}")
    print(f"当前工作目录: {os.getcwd()}")
    
    success_count = 0
    total_count = len(test_scripts)
    
    for script in test_scripts:
        if run_test(script):
            success_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"测试结果: {success_count}/{total_count} 脚本通过测试")
    
    if success_count == total_count:
        print("✅ 所有测试通过，相对路径配置正确！")
    else:
        print("❌ 部分测试失败，请检查路径配置！")

if __name__ == "__main__":
    main()