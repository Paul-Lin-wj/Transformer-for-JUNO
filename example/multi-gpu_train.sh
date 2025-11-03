#!/bin/bash

######## Part 1 #########
#SBATCH --partition=neuph
#SBATCH --qos=junotmp
#SBATCH --account=junogpu
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=8192
#SBATCH --gpus=a100:4          # 建议改成 --gres 写法，更通用
#SBATCH --job-name=muon_reco
#SBATCH --time=12:00:00

######## Part 2 #########
# 作业开始信息
echo "作业开始时间: $(date)"
echo "运行节点: $(hostname)"

# 使用用户的工作目录而不是脚本临时目录
WORK_DIR="/junofs/users/dingxf/LiDian_data/muon_track_reco/multiGPU_train/muon_rec/example"

# 假设运行你的程序
train_script="${WORK_DIR}/run.sh"
bash "${train_script}" > "${WORK_DIR}/train_output_${SLURM_JOBID}.log" 2>&1

echo "作业结束时间: $(date)"
