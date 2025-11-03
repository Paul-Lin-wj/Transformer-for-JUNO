#!/bin/bash
export JUNOTOP=/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.1/
export EOS_TOP="root://junoeos01.ihep.ac.cn"

#specify the version of JUNOSW
JUNOSW_VERSION='J25.3.0.b'
source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.1.4/setup.sh
junosw_path="/junofs/users/guangbaosun/junotop/junosw/"
source ${junosw_path}/setup.sh

input_list=$1
input_correlation_list=$2
file_no=$3
output_path=$4
dataset_output_path=$5
runid=$6

# esd_path: the file_no line of input_list file
# rtraw_path: the file_no line of input_correlation_list file
# notice that file_no start from 0
esd_path=$(sed -n "$((file_no + 1))p" ${input_list})
rtraw_path=$(sed -n "$((file_no + 1))p" ${input_correlation_list})
output_user_file="${EOS_TOP}/${output_path}/user/${runid}_$((file_no+1))_user.root"

echo "============================="
echo "input_list: ${input_list}"
echo "input_correlation_list: ${input_correlation_list}"
echo "file_no: ${file_no}"
echo "output_path: ${output_path}"
echo "dataset_output_path: ${dataset_output_path}"
echo "runid: ${runid}"
echo "============================="
echo "esd_path: ${esd_path}"
echo "rtraw_path: ${rtraw_path}"
echo "output_user_file: ${output_user_file}"
echo "============================="
# date
# echo "==========================START OF PROCESSING============================="
# python /junofs/users/guangbaosun/junotop/junosw/Examples/Tutorial/share/tut_calib2rec.py \
#     --input ${esd_path} \
#     --input-correlation ${rtraw_path} \
#     --loglevel Warn \
#     --pmtcalibsvc-ChargeAlgType 0 \
#     --pmtcalibsvc-DBcur 20250620 \
#     --global-tag WaterPhase_J25.2 \
#     --method wp-classifytrack \
#     --EnableUserOutput \
#     --user-output ${output_user_file}
# echo "==========================END   OF PROCESSING============================="
# date

source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.3.1/setup.sh
source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.3.1/junosw/setup.sh
source /junofs/users/dingxf/LiDian_data/muon_track_reco/py_env/juno_cvmfs_env/bin/activate

export RUNPATH="../python"

track_file=$output_user_file
hits_file=$esd_path
output_file="${dataset_output_path}/${runid}_$((file_no+1))_dataset.pt"

echo "output_file: ${output_file}"

#dataset creation
echo "Start Dataset Creation"
python -u $RUNPATH/RunModule.py \
    --Createdataset \
    --track_file "${track_file}" \
    --hits_file "${hits_file}" \
    --max_hits 1000 \
    --output_file "${output_file}" \