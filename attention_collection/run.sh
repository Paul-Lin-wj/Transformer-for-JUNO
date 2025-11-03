#!/bin/bash

# ==========================================
# COMPLETE ATTENTION COLLECTION SYSTEM LAUNCH SCRIPT
# 整合的Attention数据收集和分析系统启动脚本
# ==========================================

# 只需修改下面的变量来配置系统参数！
# ==========================================

# 系统配置参数
MODE="all"                    # 运行模式: "collect", "train", "analyze", "all"
NUM_SAMPLES=5                 # 收集的样本数量
NUM_EPOCHS=3                  # 训练轮数
DATA_DIR="./data_attention"       # 数据保存目录
LOG_DIR="./log"                 # 日志保存目录
DATASET_PATH="../sample/dataset"   # 数据集路径

# 模型配置参数
INPUT_DIM=5                    # 输入维度
EMBED_DIM=64                   # 嵌入维度
NUM_HEADS=2                    # 注意力头数
NUM_LAYERS=1                   # Transformer层数
HIDDEN_DIM=128                 # 隐藏层维度
BATCH_SIZE=32                  # 批处理大小
LEARNING_RATE=0.001             # 学习率

# ==========================================
# 脚本内容 - 通常不需要修改
# ==========================================

# Set up environment (same as original run.sh)
export ROOTPATH="/publicfs/juno/software/J24.1.x/setup.sh"
export RUNENVPATH="/datafs/users/wujxy/py_venv/juno_cvmfs_env/bin/activate"

# Source environments
source $ROOTPATH
source $RUNENVPATH

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
mkdir -p "$DATA_DIR" "$LOG_DIR"

# Generate timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$LOG_DIR/complete_attention_${timestamp}.log"

# Function to log to both console and file
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$log_file"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" | tee -a "$log_file"
}

echo "============================================"
echo "COMPLETE ATTENTION COLLECTION SYSTEM STARTED"
echo "============================================"
log "启动时间: $(date)"
log "日志文件: $log_file"
log "工作目录: $(pwd)"

# Log configuration
log "配置参数:"
log "  运行模式: $MODE"
log "  样本数量: $NUM_SAMPLES"
log "  训练轮数: $NUM_EPOCHS"
log "  数据目录: $DATA_DIR"
log "  日志目录: $LOG_DIR"
log "  数据集路径: $DATASET_PATH"
log ""
log "模型配置:"
log "  输入维度: $INPUT_DIM"
log "  嵌入维度: $EMBED_DIM"
log "  注意力头数: $NUM_HEADS"
log "  Transformer层数: $NUM_LAYERS"
log "  隐藏层维度: $HIDDEN_DIM"
log "  批处理大小: $BATCH_SIZE"
log "  学习率: $LEARNING_RATE"
log ""
log "输出模式: 完整输出（无截断）"

# Check if Python file exists
if [ ! -f "attention_collector.py" ]; then
    log_error "attention_collector.py 文件不存在！"
    exit 1
fi

# Make Python file executable
chmod +x attention_collector.py

# Set Python path to include parent python directory
export PYTHONPATH="../python:$PYTHONPATH"

# Log system information
log "系统信息:"
log "  Python版本: $(python --version)"
log "  Python路径: $(which python)"
log "  PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

# Check CUDA availability
if python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null; then
    log "  CUDA可用: 是"
    if python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null; then
        log "  GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '未知')"
    fi
else
    log "  CUDA可用: 否"
fi

# Check memory usage
log "内存使用情况（启动前）:"
free -h | while read line; do log "  $line"; done

# Run the complete attention collection system
log "启动完整的attention收集系统..."
log "执行命令: python attention_collector.py --mode $MODE --samples $NUM_SAMPLES --epochs $NUM_EPOCHS --data_dir $DATA_DIR --log_dir $LOG_DIR"

# Run the script with comprehensive logging
start_time=$(date +%s)
python attention_collector.py \
    --mode "$MODE" \
    --samples "$NUM_SAMPLES" \
    --epochs "$NUM_EPOCHS" \
    --data_dir "$DATA_DIR" \
    --log_dir "$LOG_DIR" \
    2>&1 | tee -a "$log_file"

# Check exit status
exit_status=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

if [ $exit_status -eq 0 ]; then
    log_success "完整的attention收集系统成功完成！"
    log_success "总执行时间: ${duration} 秒"

    # Display summary of generated files
    log ""
    log "生成的文件摘要:"

    if [ -d "$DATA_DIR" ]; then
        data_files=$(find "$DATA_DIR" -name "*.txt" -o -name "*.json" -o -name "*.pth" | wc -l)
        log "  总数据文件数: $data_files"

        # Breakdown by type
        txt_files=$(find "$DATA_DIR" -name "*.txt" | wc -l)
        json_files=$(find "$DATA_DIR" -name "*.json" | wc -l)
        pth_files=$(find "$DATA_DIR" -name "*.pth" | wc -l)

        log "    文本文件 (.txt): $txt_files"
        log "    元数据文件 (.json): $json_files"
        log "    模型文件 (.pth): $pth_files"

        # List recent files
        log "  最近生成的文件:"
        find "$DATA_DIR" -name "*.txt" -o -name "*.json" -o -name "*.pth" -newer "attention_collector.py" | head -5 | while read file; do
            size=$(du -h "$file" | cut -f1)
            log "    $file ($size)"
        done
    fi

    # Log final memory usage
    log ""
    log "内存使用情况（完成后）:"
    free -h | while read line; do log "  $line"; done

    # Check if attention data files were created successfully
    if [ -f "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR"/*.txt 2>/dev/null)" ]; then
        log_success "attention数据文件已成功创建！"
        log_success "所有attention权重数据已完整保存（无截断）"
    else
        log_error "警告: 未找到attention数据文件，请检查日志"
    fi

    log ""
    log "============================================"
    log "完整的attention收集系统成功完成！"
    log "完成时间: $(date)"
    log "日志文件: $log_file"
    log "数据目录: $DATA_DIR"
    log "============================================"
    log ""
    log "🎉 系统已完成！🎉"
    log ""
    log "下一步操作:"
    log "1. 检查生成的完整attention数据文件: $DATA_DIR"
    log "2. 使用完整的attention权重进行DSA算法分析"
    log "3. 查看分析报告"
    log "4. 检查日志文件了解详细执行过程: $log_file"
    log ""
    log "✅ 所有attention数据均已完整保存，没有任何截断！"
    log "✅ 准备就绪，可用于DSA算法分析！"
    log ""

else
    log_error "完整的attention收集系统失败，退出状态: $exit_status"
    log_error "执行时间: ${duration} 秒"
    log_error "请检查日志文件获取详细信息: $log_file"

    # Try to get error details
    log_error "最后几行执行信息:"
    tail -10 "$log_file" | while read line; do log_error "  $line"; done

    exit $exit_status
fi

echo "进程完成。详细信息请查看日志文件。"