#!/bin/bash

runid=9859
input="root://junoeos01.ihep.ac.cn//eos/juno/esd/J25.5.0.b/2025/0903/RUN.9857.JUNODAQ.Physics.ds-2.global_trigger.20250903055150.001_J25.5.0_J25.5.0.b.esd"
input_correlation="root://junoeos01.ihep.ac.cn//eos/juno/rtraw/2025/0903/RUN.9857.JUNODAQ.Physics.ds-2.global_trigger.20250903055150.001_J25.5.0.rtraw"
output_path="/eos/juno/groups/Reconstruction/dingxf/LiDian_data/muon_reco/RUN_${runid}"
dataset_output_path="/junofs/users/dingxf/LiDian_data/muon_track_reco/muon_reco/sample/RUN_${runid}_1000_time_sort"

eos root://junoeos01.ihep.ac.cn mkdir -p ${output_path}/user
mkdir -p ${dataset_output_path}

# hep_sub muon_rec.sh -g junogns -argu ${input} ${input_correlation} ${output_path} -o logfiles/log_${runid}.log -e logfiles/err_${runid}.err
nohup ./muon_rec.sh ${input} ${input_correlation} ${output_path} ${dataset_output_path} ${runid} >&logfiles/log_${runid}.log&
