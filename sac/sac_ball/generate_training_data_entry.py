import os
import sys

# 设置默认参数

# 参数名修正，确保与主脚本一致
BASELINE_FRAC = float(os.environ.get('BASELINE_FRAC', '0.6'))
NUM_STEPS = int(os.environ.get('NUM_STEPS', '30000'))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'train_data')

args = [
    '--baseline_frac', str(BASELINE_FRAC),
    '--num_steps', str(NUM_STEPS),
    '--output_dir', OUTPUT_DIR
]

if __name__ == '__main__':
    import subprocess
    script_path = os.path.join(os.path.dirname(__file__), 'generate_training_data.py')
    result = subprocess.run([sys.executable, script_path] + args)
    sys.exit(result.returncode)
