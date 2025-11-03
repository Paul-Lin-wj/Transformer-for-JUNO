#!/bin/bash

# ==========================================
# TRANSFORMER WITH DSA TRAINING LAUNCH SCRIPT
# DSA优化Transformer训练启动脚本
# ==========================================

# 只需修改下面的变量来配置训练参数！
# ==========================================

# 训练基本配置
NUM_EPOCHS=200                 # 训练轮数
BATCH_SIZE=32                  # 批大小
LEARNING_RATE=0.001            # 学习率
SAVE_EVERY=50                  # 每隔多少epoch保存模型
RESUME_TRAINING=false          # 是否从上次训练继续 (true/false)

# 模型配置
INPUT_DIM=5                    # 输入特征维度
EMBED_DIM=128                  # 嵌入维度
NUM_HEADS=8                    # 注意力头数
NUM_LAYERS=4                   # Transformer层数
FF_DIM=512                     # 前馈网络隐藏层维度
OUTPUT_DIM=6                   # 输出维度（入射点3个+出射点3个坐标）
DROPOUT=0.1                    # Dropout率

# DSA配置
DSA_ENABLED=true                # 是否启用DSA优化 (true/false)
SPARSITY_RATIO=0.1             # 初始稀疏度比例
TARGET_SPARSITY=0.05           # 目标稀疏度
SPARSITY_WEIGHT=0.001          # 稀疏度正则化权重
ENTROPY_WEIGHT=0.0001          # 熵正则化权重

# 数据配置
DATASET_PATH="/data/juno/lin/JUNO/transformer/muon_track_reco_transformer/sample/data_test"
# DATASET_PATH="/scratchfs/juno/fanliangqianjin/muonRec/TRANSFORMER_FOR_TTinput/muon_track_reco_transformer/sample/TTdataset_small/"
MAX_FILES=100                  # 最大加载文件数（可选，留空表示加载所有文件）
SEQ_LEN=50                     # 序列长度（可选，留空自动确定）
TRAIN_RATIO=0.8                # 训练集比例（在训练+验证集中的比例）
NORMALIZE=true                 # 是否归一化数据 (true/false)
AUGMENT_TRAIN=false            # 是否对训练集进行数据增强 (true/false)

# 训练策略配置
SCHEDULER="plateau"             # 学习率调度器: "plateau", "cosine", "none"
WEIGHT_DECAY=1e-4              # 权重衰减
GRAD_CLIP=1.0                  # 梯度裁剪阈值（留空禁用）
EARLY_STOPPING_PATIENCE=50     # 早停耐心值
MIN_DELTA=1e-6                 # 早停最小改善阈值

# 优化器配置
OPTIMIZER="adamw"               # 优化器类型: "adam", "adamw", "sgd"
SCHEDULER_FACTOR=0.5           # 学习率衰减因子
SCHEDULER_PATIENCE=10          # 学习率调度耐心值
MIN_LR=1e-6                    # 最小学习率

# 任务配置
TASK_TYPE="regression"          # 任务类型: "regression", "classification", "binary_classification"

# GPU配置
NUM_GPUS=1                      # GPU数量，0表示CPU，>0表示使用GPU
GPU_IDS="0"                      # 指定使用的GPU ID，例如"0,1,2,3"，留空则自动选择

# 系统配置
NUM_WORKERS=4                  # 数据加载进程数
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

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

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
log "  FF Dim: ${YELLOW}$FF_DIM${NC}"
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
log "  Data Split: ${YELLOW}4:1 (Train+Val:Test)${NC}"
log "  Train/Val Split: ${YELLOW}$TRAIN_RATIO${NC} (in Train+Val set)"
log "  Sequence Length: ${YELLOW}${SEQ_LEN:-"Auto"}${NC}"
log "  Normalize: ${YELLOW}$NORMALIZE${NC}"
log "  Augment Train: ${YELLOW}$AUGMENT_TRAIN${NC}"
log ""
log "${GREEN}System Configuration:${NC}"
log "  Log Dir: ${YELLOW}$LOG_DIR${NC}"
log "  Model Dir: ${YELLOW}$MODEL_DIR${NC}"
log "  Num Workers: ${YELLOW}$NUM_WORKERS${NC}"
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

# 创建必要的目录
log "${BLUE}Creating directories...${NC}"
mkdir -p "$LOG_DIR"
mkdir -p "$MODEL_DIR"

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(dirname "$0")/../python"

# 构建DSA配置
DSA_CONFIG=""
if [ "$DSA_ENABLED" = true ]; then
    DSA_CONFIG="{
        \"sparsity_ratio\": $SPARSITY_RATIO,
        \"target_sparsity\": $TARGET_SPARSITY,
        \"adaptive_threshold\": True,
        \"min_connections\": 5,
        \"warmup_epochs\": 10,
        \"schedule_type\": \"adaptive\"
    }"
fi

# 构建Python训练脚本
cat > /tmp/train_dsa_config.py << EOF
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../python')

from trainer_dsa import TrainerWithDSA

# 训练配置
config = {
    # 基本配置
    'num_epochs': $NUM_EPOCHS,
    'batch_size': $BATCH_SIZE,
    'learning_rate': $LEARNING_RATE,
    'save_every': $SAVE_EVERY,
    'resume_training': $([ "$RESUME_TRAINING" = true ] && echo "True" || echo "False"),

    # 模型配置
    'input_dim': $INPUT_DIM,
    'embed_dim': $EMBED_DIM,
    'num_heads': $NUM_HEADS,
    'num_layers': $NUM_LAYERS,
    'ff_dim': $FF_DIM,
    'output_dim': $OUTPUT_DIM,
    'dropout': $DROPOUT,

    # DSA配置
    'dsa_enabled': $([ "$DSA_ENABLED" = true ] && echo "True" || echo "False"),
    'dsa_config': $([ "$DSA_ENABLED" = true ] && echo "$DSA_CONFIG" || echo "None"),
    'sparsity_weight': $SPARSITY_WEIGHT,
    'entropy_weight': $ENTROPY_WEIGHT,

    # 数据配置
    'dataset_path': '$DATASET_PATH',
    $( [ -n "$MAX_FILES" ] && echo "'max_files': $MAX_FILES," || echo "" )
    'seq_len': $([ -n "$SEQ_LEN" ] && echo $SEQ_LEN || echo "None"),
    'train_ratio': $TRAIN_RATIO,
    'normalize': $([ "$NORMALIZE" = true ] && echo "True" || echo "False"),
    'augment_train': $([ "$AUGMENT_TRAIN" = true ] && echo "True" || echo "False"),

    # 训练策略配置
    'scheduler': '$SCHEDULER',
    'weight_decay': $WEIGHT_DECAY,
    $( [ -n "$GRAD_CLIP" ] && echo "'grad_clip': $GRAD_CLIP," || echo "" )
    'early_stopping_patience': $EARLY_STOPPING_PATIENCE,
    'min_delta': $MIN_DELTA,

    # 优化器配置
    'scheduler_factor': $SCHEDULER_FACTOR,
    'scheduler_patience': $SCHEDULER_PATIENCE,
    'min_lr': $MIN_LR,

    # 任务配置
    'task_type': '$TASK_TYPE',

    # GPU配置
    'num_gpus': $NUM_GPUS,
    'gpu_ids': '$GPU_IDS',

    # 系统配置
    'num_workers': $NUM_WORKERS,
    'log_dir': '$LOG_DIR',
    'model_dir': '$MODEL_DIR',
}

if __name__ == "__main__":
    print("Starting Transformer with DSA training...")
    print(f"Configuration: {config}")

    try:
        # 创建训练器
        trainer = TrainerWithDSA(config)

        # 开始训练
        trainer.train()

        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

# 启动训练
log "${BLUE}Starting training...${NC}"
log "${BLUE}========================================${NC}"

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 记录训练开始时间
START_TIME=$(date)
log "Training started at: $START_TIME"

# 运行训练（根据GPU数量选择不同的启动方式）
if [ "$NUM_GPUS" -gt 1 ]; then
    # 多GPU训练使用accelerate
    log "${GREEN}Using multi-GPU training with $NUM_GPUS GPUs${NC}"
    accelerate launch --num_processes $NUM_GPUS --main_process_port 29999 /tmp/train_dsa_config.py 2>&1 | tee -a "$LOG_FILE"
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
elif [ "$NUM_GPUS" -eq 1 ]; then
    # 单GPU训练
    log "${GREEN}Using single GPU training${NC}"
    python /tmp/train_dsa_config.py 2>&1 | tee -a "$LOG_FILE"
    TRAINING_EXIT_CODE=${PIPESTATUS[0]}
else
    # CPU训练
    log "${YELLOW}Using CPU training${NC}"
    python /tmp/train_dsa_config.py 2>&1 | tee -a "$LOG_FILE"
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

# 清理临时文件
rm -f /tmp/train_dsa_config.py

log "${BLUE}Done.${NC}"
log "========================================"
log "All outputs have been logged to: $LOG_FILE"
log "========================================"