#!/bin/bash

# ==========================================
# TRANSFORMER WITH DSA TRAINING LAUNCH SCRIPT
# DSA优化Transformer训练启动脚本
# ==========================================

# 只需修改下面的变量来配置训练参数！
# ==========================================

# 训练基本配置
NUM_EPOCHS=200                   # 训练轮数（减少用于测试）
BATCH_SIZE=16                  # 批大小（减小以减少GPU内存使用）
LEARNING_RATE=0.001            # 学习率
SAVE_EVERY=20                    # 每隔多少epoch保存模型
RESUME_TRAINING=true          # 是否从上次训练继续 (true/false)

# 模型配置
INPUT_DIM=5                    # 输入特征维度
EMBED_DIM=32                   # 嵌入维度（减少内存）
NUM_HEADS=4                    # 注意力头数（减少内存）
NUM_LAYERS=2                   # Transformer层数（减少内存）
HIDDEN_DIM=64                  # 前馈网络隐藏层维度（减少内存）
OUTPUT_DIM=6                   # 输出维度（入射点3个+出射点3个坐标）
DROPOUT=0.1                    # Dropout率

# DSA配置
DSA_ENABLED=true                # 是否启用DSA优化 (true/false)
SPARSITY_RATIO=0.1             # 初始稀疏度比例
TARGET_SPARSITY=0.05           # 目标稀疏度
SPARSITY_WEIGHT=0.001          # 稀疏度正则化权重
ENTROPY_WEIGHT=0.0001          # 熵正则化权重

# 数据配置
# DATASET_PATH="/data/juno/lin/JUNO/transformer/muon_track_reco_transformer/sample/dataset"
DATASET_PATH="/scratchfs/juno/fanliangqianjin/muonRec/TRANSFORMER_FOR_TTinput/muon_track_reco_transformer/sample/TTdataset_small/"
MAX_FILES="1000"                      # 最大加载文件数（可选，留空表示加载所有文件）
TRAIN_RATIO=0.8                # 训练集比例
NORMALIZE=true                 # 是否归一化数据 (true/false)

# 训练策略配置
SCHEDULER="plateau"             # 学习率调度器: "plateau", "cosine", "none"
WEIGHT_DECAY=1e-4              # 权重衰减
EARLY_STOPPING_PATIENCE=30     # 早停耐心值
MIN_DELTA=1e-6                 # 早停最小改善阈值

# GPU配置
NUM_GPUS=1                      # GPU数量，0表示CPU，>0表示使用GPU
GPU_IDS="1"                     # 指定使用的GPU ID，例如"0,1,2,3"，留空则自动选择

# 系统配置
NUM_WORKERS=16                  # 数据加载进程数
LOG_DIR="./log"                 # 日志保存目录
MODEL_DIR="./model"             # 模型保存目录

# ==========================================
# 脚本内容 - 通常不需要修改
# ==========================================

# 设置Python环境（与原始transformer脚本相同）
export ROOTPATH="/publicfs/juno/software/J24.1.x/setup.sh"
export RUNENVPATH="/datafs/users/wujxy/py_venv/juno_cvmfs_env/bin/activate"

# 加载环境
source $ROOTPATH
source $RUNENVPATH

# 设置CUDA内存管理以避免内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA操作，更容易调试
export TORCH_CUDA_ARCH_LIST="8.6"  # 指定GPU架构

# 禁用Python输出缓冲
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# 设置DDP参数以避免buffer同步问题
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN

# 创建必要的目录
mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# 创建日志文件并记录系统信息（在log文件最开头）
echo "========================================" > "$LOG_FILE"
echo "TRAINING LOG STARTED AT: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "SYSTEM & PROCESS INFORMATION:" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "Process ID (PID): $$" >> "$LOG_FILE"
echo "Parent PID: $PPID" >> "$LOG_FILE"
echo "Script name: $0" >> "$LOG_FILE"
echo "User: $(whoami)" >> "$LOG_FILE"
echo "Working directory: $(pwd)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# System information
echo "SYSTEM INFORMATION:" >> "$LOG_FILE"
echo "OS: $(uname -s) $(uname -r)" >> "$LOG_FILE"
echo "CPU Info: $(lscpu | grep 'Model name' | awk -F': ' '{print $2}')" >> "$LOG_FILE"
echo "CPU Cores: $(nproc) logical, $(lscpu | grep 'Core(s) per socket' | awk '{print $4}') physical per socket" >> "$LOG_FILE"
echo "Memory Info:" >> "$LOG_FILE"
free -h | grep -E "Mem|Swap" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# GPU information (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU INFORMATION:" >> "$LOG_FILE"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >> "$LOG_FILE"
else
    echo "GPU INFORMATION:" >> "$LOG_FILE"
    echo "nvidia-smi not available" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"

# Disk information
echo "DISK INFORMATION:" >> "$LOG_FILE"
df -h . | tail -n 1 >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 颜色输出（仅在终端显示，不写入日志）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数：同时输出到终端和日志文件（去除颜色代码）
log() {
    local message="$1"
    # 去除ANSI颜色代码后写入日志文件
    echo -e "$message" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
    # 在终端显示带颜色的消息
    echo -e "$message"
}

# 打印配置信息
log "${BLUE}========================================${NC}"
log "${BLUE}TRANSFORMER WITH DSA TRAINING CONFIG${NC}"
log "${BLUE}========================================${NC}"
log "${GREEN}Training Configuration:${NC}"
log "  Epochs: ${YELLOW}$NUM_EPOCHS${NC}"
log "  Batch Size: ${YELLOW}$BATCH_SIZE${NC}"
log "  Learning Rate: ${YELLOW}$LEARNING_RATE${NC}"
log "  Save Every: ${YELLOW}$SAVE_EVERY${NC} epochs"
log "  Resume Training: ${YELLOW}$RESUME_TRAINING${NC}"
log ""
log "${GREEN}Model Configuration:${NC}"
log "  Input Dim: ${YELLOW}$INPUT_DIM${NC}"
log "  Embed Dim: ${YELLOW}$EMBED_DIM${NC}"
log "  Num Heads: ${YELLOW}$NUM_HEADS${NC}"
log "  Num Layers: ${YELLOW}$NUM_LAYERS${NC}"
log "  Hidden Dim: ${YELLOW}$HIDDEN_DIM${NC}"
log "  Output Dim: ${YELLOW}$OUTPUT_DIM${NC}"
log "  Dropout: ${YELLOW}$DROPOUT${NC}"
log ""
log "${GREEN}DSA Configuration:${NC}"
log "  DSA Enabled: ${YELLOW}$DSA_ENABLED${NC}"
log "  Initial Sparsity: ${YELLOW}$SPARSITY_RATIO${NC}"
log "  Target Sparsity: ${YELLOW}$TARGET_SPARSITY${NC}"
log "  Sparsity Weight: ${YELLOW}$SPARSITY_WEIGHT${NC}"
log ""
log "${GREEN}GPU Configuration:${NC}"
log "  Num GPUs: ${YELLOW}$NUM_GPUS${NC}"
log "  GPU IDs: ${YELLOW}${GPU_IDS:-"Auto"}${NC}"
log ""
log "${GREEN}Data Configuration:${NC}"
log "  Dataset Path: ${YELLOW}$DATASET_PATH${NC}"
log "  Max Files: ${YELLOW}${MAX_FILES:-"All"}${NC}"
log "  Train Ratio: ${YELLOW}$TRAIN_RATIO${NC}"
log ""
log "${GREEN}System Configuration:${NC}"
log "  Log Dir: ${YELLOW}$LOG_DIR${NC}"
log "  Model Dir: ${YELLOW}$MODEL_DIR${NC}"
log "  Log File: ${YELLOW}$LOG_FILE${NC}"
log "${BLUE}========================================${NC}"

# 检查Python环境和PyTorch
log "${BLUE}Checking Python environment...${NC}"
python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')
" 2>&1 | tee -a "$LOG_FILE"

log "${BLUE}Checking PyTorch...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA is not available! Training will use CPU.')
" 2>&1 | tee -a "$LOG_FILE"

# 根据NUM_GPUS设置CUDA_VISIBLE_DEVICES
if [ "$NUM_GPUS" -gt 0 ]; then
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        log "${GREEN}Using specified GPUs: $GPU_IDS${NC}"
    else
        # 自动选择前NUM_GPUS个GPU
        GPU_IDS=$(python -c "
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    num_gpus = min($NUM_GPUS, gpu_count)
    print(','.join(str(i) for i in range(num_gpus)))
else:
    print('')
" 2>&1)
        if [ -n "$GPU_IDS" ]; then
            export CUDA_VISIBLE_DEVICES="$GPU_IDS"
            log "${GREEN}Using auto-selected GPUs: $GPU_IDS${NC}"
        else
            log "${RED}WARNING: No GPUs available, falling back to CPU${NC}"
            NUM_GPUS=0
        fi
    fi
else
    log "${YELLOW}GPU training disabled, using CPU${NC}"
fi

# 检查数据路径
log "${BLUE}Checking dataset path...${NC}"
if [ ! -d "$DATASET_PATH" ]; then
    log "${YELLOW}Warning: Dataset path does not exist: $DATASET_PATH${NC}"
    log "${YELLOW}Please check the DATASET_PATH variable.${NC}"
fi

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")/../python"

# 计算test_size（避免浮点数运算）
TEST_SIZE=$(awk "BEGIN {printf \"%.2f\", 1-$TRAIN_RATIO}")

# 构建训练命令
TRAIN_CMD="python ../python/RunModule.py"
TRAIN_CMD="$TRAIN_CMD --TrainModel"
TRAIN_CMD="$TRAIN_CMD --mission_name dsa_transformer"
TRAIN_CMD="$TRAIN_CMD --pre_method TensorPre"
TRAIN_CMD="$TRAIN_CMD --train_model_name Transformer"
TRAIN_CMD="$TRAIN_CMD --pklfile_train_path \"$DATASET_PATH\""
# 添加MAX_FILES参数（如果设置了的话）
if [ -n "$MAX_FILES" ] && [ "$MAX_FILES" != "" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_files $MAX_FILES"
fi
TRAIN_CMD="$TRAIN_CMD --num_epochs $NUM_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --save_every $SAVE_EVERY"
TRAIN_CMD="$TRAIN_CMD --embed_dim $EMBED_DIM"
TRAIN_CMD="$TRAIN_CMD --num_heads $NUM_HEADS"
TRAIN_CMD="$TRAIN_CMD --num_layers $NUM_LAYERS"
TRAIN_CMD="$TRAIN_CMD --hidden_dim $HIDDEN_DIM"
TRAIN_CMD="$TRAIN_CMD --input_dim $INPUT_DIM"
TRAIN_CMD="$TRAIN_CMD --test_size $TEST_SIZE"
TRAIN_CMD="$TRAIN_CMD --GPUid \"$GPU_IDS\""
TRAIN_CMD="$TRAIN_CMD --run_timestamp $TIMESTAMP"

# 添加DSA参数
if [ "$DSA_ENABLED" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --dsa_enabled"
    TRAIN_CMD="$TRAIN_CMD --sparsity_ratio $SPARSITY_RATIO"
    TRAIN_CMD="$TRAIN_CMD --target_sparsity $TARGET_SPARSITY"
    TRAIN_CMD="$TRAIN_CMD --sparsity_weight $SPARSITY_WEIGHT"
    TRAIN_CMD="$TRAIN_CMD --entropy_weight $ENTROPY_WEIGHT"
fi

# 启动训练
log "${BLUE}Starting training...${NC}"
log "${BLUE}========================================${NC}"

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 记录训练开始时间
START_TIME=$(date)
log "Training started at: $START_TIME"

# Debug: Print the command that will be executed
log "${BLUE}Executing command:${NC}"
log "$TRAIN_CMD"
log "${BLUE}========================================${NC}"

# 运行训练
if [ "$NUM_GPUS" -gt 1 ]; then
    # 多GPU训练使用accelerate
    log "${GREEN}Using multi-GPU training with $NUM_GPUS GPUs${NC}"
    # For accelerate, we need to pass the command as arguments, not as a string
    bash -c "accelerate launch --num_processes $NUM_GPUS --main_process_port 29999 --multi_gpu $TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
else
    # 单GPU或CPU训练
    if [ "$NUM_GPUS" -eq 1 ]; then
        log "${GREEN}Using single GPU training${NC}"
    else
        log "${YELLOW}Using CPU training${NC}"
    fi
    # Use bash -c instead of eval to properly handle command arguments
    bash -c "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
fi

# 记录训练结束时间
END_TIME=$(date)
log "Training ended at: $END_TIME"

# 检查训练结果
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    log "${GREEN}========================================${NC}"
    log "${GREEN}TRAINING COMPLETED SUCCESSFULLY!${NC}"
    log "${GREEN}========================================${NC}"
    log "${GREEN}Models saved in: ${YELLOW}$MODEL_DIR/[timestamp]/${NC}"
    log "${GREEN}Logs saved in: ${YELLOW}$LOG_FILE${NC}"
    log ""
    log "Training Summary:"
    log "  Start time: $START_TIME"
    log "  End time: $END_TIME"
    log "  GPU configuration: $NUM_GPUS GPUs"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        log "  Used GPUs: $CUDA_VISIBLE_DEVICES"
    fi
else
    log "${RED}========================================${NC}"
    log "${RED}TRAINING FAILED!${NC}"
    log "${RED}========================================${NC}"
    log "${RED}Please check the error messages in the log file: $LOG_FILE${NC}"
    exit 1
fi

log "${BLUE}Done.${NC}"
log "========================================"
log "All outputs have been logged to: $LOG_FILE"
log "========================================"