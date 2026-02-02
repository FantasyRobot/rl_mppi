# PowerShell script for Ball Environment SAC Pipeline
# This script runs the complete pipeline from data generation to model testing

#powershell.exe -ExecutionPolicy Bypass -File "d:\VirtualSpace\rl_mppi\sac\sac_ball\shell.ps1"

# Change to the script directory
Set-Location "$PSScriptRoot"

# Configuration with enhanced settings for improved consistency
$TARGET_X = 2.0
$TARGET_Y = -2.0
$NUM_STEPS = 100000  # Increased data for more diverse training scenarios
$EPOCHS = 3000       # Extended training for better convergence and robustness
$BATCH_SIZE = 256
$DATA_DIR = "train_data"  # New data directory with enhanced scenarios
$MODEL_PATH = "sac_ball_model.pth"  # Model optimized for consistent target reaching

# Step 1: Generate training data
Write-Host "====================================="
Write-Host "Step 1: Generating Training Data"
Write-Host "====================================="
python generate_training_data.py `
    --num_steps $NUM_STEPS `
    --output_dir $DATA_DIR `
    --target_x $TARGET_X `
    --target_y $TARGET_Y

# Step 2: Train SAC model
Write-Host "`n====================================="
Write-Host "Step 2: Training SAC Model"
Write-Host "====================================="
python train_sac_ball.py `
    --data_dir $DATA_DIR `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --save_path $MODEL_PATH `
    --target_x $TARGET_X `
    --target_y $TARGET_Y

# Step 3: Test SAC model
Write-Host "`n====================================="
Write-Host "Step 3: Testing SAC Model"
Write-Host "====================================="
python test_sac_ball.py  --model_path $MODEL_PATH  --target_x $TARGET_X --target_y $TARGET_Y --num_tests 10 --max_steps 2000

Write-Host "`n====================================="
Write-Host "Pipeline Complete!"
Write-Host "====================================="