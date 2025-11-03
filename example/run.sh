#!/bin/bash

# Independent environment and root environment

# Ubuntu for 192.168.237.165
# export RUNENVPATH="/datafs/users/wujxy/py_venv/juno_cvmfs_env/bin/activate"

export ROOTPATH="/publicfs/juno/software/J24.1.x/setup.sh"
export RUNENVPATH="/datafs/users/wujxy/py_venv/juno_cvmfs_env/bin/activate"

# export ROOTPATH="/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/Jlatest/setup.sh"
# AlmaLinux for lxlogin.ihep.ac.cn
# export RUNENVPATH="/junofs/users/dingxf/LiDian_data/muon_track_reco/py_env/juno_cvmfs_env/bin/activate"
# export ROOTPATH="/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.3.1/setup.sh"

export RUNPATH="../python"

source $ROOTPATH
source $RUNENVPATH

mission="test" #mission name

# Create log directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/run_${mission}_${timestamp}.log"

echo "============================================"
echo "Run started at: $(date)"
echo "Log file: $log_file"
echo "============================================"

# Function to log to both console and file
log() {
    echo "$1" | tee -a "$log_file"
}

# =========== First step : Dataset Creation
# input :
# track_file : the root file containing the muon track information (y_data)
# label_type : the label type for the track information, now only support "TT" and "Wp"
# hits_file  : the root (esd) file containing the pmt hits information (x_data)
# max_hits   : the maximum number of hits for each event, follows the time sort selection
# only_xdata : if create dataset only containing x_data (for prediction only), default is False
# output_file: the output file path for the created dataset (.pt file)

# mkdir -p ../sample/dataset_9926

# log "Start Dataset Creation"
# python -u $RUNPATH/RunModule.py \
#     --Createdataset \
#     --label_type "TT" \
#     --track_file "/datafs/users/fanlqj/JUNO_Data_Analysis_Training/muonRec/Bundle_TT/TT_data_check/9926_bugEvt_track_params.root" \
#     --hits_file_list "/junofs/users/fanliangqianjin/muonRec/ReadTimeStamp/prepare-input/prepare-input/eos_list/run_9926.txt" \
#     --max_hits 17000 \
#     --output_file "../sample/dataset_9926/9926_muon_TT" 2>&1 | tee -a "$log_file"

# =========== Second step : Model Training and Tuning
# input :
# mission_name       : the name for this training mission, will be used to create output folder
# pre_method         : the data pre-processing method, now only support "TensorPre"
# train_model_name   : the model name for training, now only support "Transformer"
# pklfile_train_path : the folder path for the training dataset (.pt file)
# model parameters   : embed_dim, num_heads, num_layers, hidden_dim, input_dim
# training parameters: num_epochs, test_size, learning_rate, scheduler_step, batch_size
# output :
# the output folder will be created under ../output/mission_name/, containing the model.pth file and training logs


echo "Start Model Training"
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29999 $RUNPATH/RunModule.py \
accelerate launch $RUNPATH/RunModule.py \
    --mission_name "${mission}" \
    --TrainModel \
    --pre_method "TensorPre" \
    --train_model_name "Transformer" \
    --pklfile_train_path "../sample/dataset" \
    --embed_dim 64 \
    --num_heads 2 \
    --num_layers 1 \
    --hidden_dim 128 \
    --input_dim 5 \
    --num_epochs 50 \
    --test_size 0.1 \
    --learning_rate 1e-3 \
    --batch_size 128 2>&1 | tee -a "$log_file"

# input :
# the parameters for model tuning, same as training
# ModelTuning : flag to indicate model tuning
# pretrained_model_path : the pre-trained model.pth file path for model tuning

# echo "Start Model tuning"
# accelerate launch $RUNPATH/RunModule.py \
#     --mission_name "${mission}" \
#     --TrainModel \
#     --ModelTuning \
#     --pre_method "TensorPre" \
#     --train_model_name "Transformer" \
#     --pretrained_model_path "../output/test/result/Transformer_final.pth" \
#     --pklfile_train_path "../sample/dataset" \
#     --embed_dim 64 \
#     --num_heads 2 \
#     --num_layers 1 \
#     --hidden_dim 128 \
#     --input_dim 5 \
#     --num_epochs 1 \
#     --test_size 0.1 \
#     --learning_rate 1e-3 \
#     --batch_size 128 \


# =========== Third step : Model Prediction
# input :
# predict_model_name : the model name for prediction, now only support "Transformer"
# predict_model_path : the model.pth file path for prediction
# pklfile_predict_path: the folder path for the prediction dataset (.pt file), only containing x_data (only_xdata=True)
# model parameters   : embed_dim, num_heads, num_layers, hidden_dim (same as training)

# log "Start Model Prediction"
# python -u $RUNPATH/RunModule.py \
#     --Predict \
#     --predict_model_name "Transformer" \
#     --predict_model_path "../output/test/result/Transformer_final.pth" \
#     --pklfile_predict_path "../sample/dataset" \
#     --embed_dim 64 \
#     --num_heads 2 \
#     --num_layers 1 \
#     --hidden_dim 128 \
#     --input_dim 5 2>&1 | tee -a "$log_file"

# Log completion
log "============================================"
log "Run completed at: $(date)"
log "============================================"
