import os
import sys

# 1. 生成高比例基线演示数据集

def generate_data():
    print('生成高比例基线演示数据集...')
    # 可根据需要自定义参数
    os.environ['BASELINE_FRAC'] = '1.0'  # 100%基线演示
    os.environ['NUM_STEPS'] = '100000'   # 10万步
    os.environ['OUTPUT_DIR'] = 'train_data'  # 输出目录
    ret = os.system(f'{sys.executable} generate_training_data_entry.py')
    if ret != 0:
        print('数据生成失败')
        sys.exit(1)

# 2. 训练SAC模型
def train_model():
    print('开始训练SAC模型...')
    ret = os.system(f'{sys.executable} train_sac_ball_entry.py')
    if ret != 0:
        print('训练失败')
        sys.exit(2)

# 3. 测试SAC模型
def test_model():
    print('测试SAC模型轨迹...')
    ret = os.system(f'{sys.executable} test_sac_ball_entry.py')
    if ret != 0:
        print('测试失败')
        sys.exit(3)

if __name__ == '__main__':
    generate_data()
    train_model()
    test_model()
    print('全部流程完成！')
