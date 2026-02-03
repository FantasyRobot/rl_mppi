import os
import sys

# 设置默认参数

# 参数名修正，确保与主脚本一致
SAVE_PATH = os.environ.get('SAVE_PATH', 'sac_ball_model_final.pth')
EPOCHS = int(os.environ.get('EPOCHS', '200'))
DATA_DIR = os.environ.get('DATA_DIR', 'train_data')

args = [
    '--save_path', SAVE_PATH,
    '--epochs', str(EPOCHS),
    '--data_dir', DATA_DIR
]

if __name__ == '__main__':
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), 'train_sac_ball.py')
    result = subprocess.run([sys.executable, script_path] + args)
    sys.exit(result.returncode)
