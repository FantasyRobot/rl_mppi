import os
import sys

# 设置默认参数
MODEL_PATH = os.environ.get('MODEL_PATH', 'sac_ball_model_final.pth')
TARGET_X = float(os.environ.get('TARGET_X', '8.0'))
TARGET_Y = float(os.environ.get('TARGET_Y', '8.0'))
NUM_TESTS = int(os.environ.get('NUM_TESTS', '10'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '4000'))

# 构建命令行参数
args = [
    f'--model_path', MODEL_PATH,
    f'--target_x', str(TARGET_X),
    f'--target_y', str(TARGET_Y),
    f'--num_tests', str(NUM_TESTS),
    f'--max_steps', str(MAX_STEPS)
]

# 调用test_sac_ball.py
if __name__ == '__main__':
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), 'test_sac_ball.py')
    result = subprocess.run([sys.executable, script_path] + args)
    sys.exit(result.returncode)
