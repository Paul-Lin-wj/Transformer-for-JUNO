#!/bin/bash
export WORKDIR=/junofs/users/dingxf/LiDian_data/muon_track_reco/process_run_bundle

# run_list=(9737 9739 9740 9754 9755 9756 9757 9766 9767 9768 9769 9770 9771 9773 9774 9775 9776 9777 9778 9779 9780 9781 9782 9783 9784 9785 9786)
# run_list=(9787 9788 9789 9790 9794 9796 9797 9800 9801 9802 9803 9804 9805 9807 9808 9816 9817 9818 9819 9820)
run_list=(9808 9816 9817 9818 9819 9820)

source /junofs/users/dingxf/python_venv/juno_env/bin/activate

cd $WORKDIR
mkdir -p inputs
mkdir -p logfiles
# loop to generate file list with ihepdb-export-data
for runid in "${run_list[@]}"; do
    echo "Generating file list for RUN_${runid}"
    # if inputs/${runid}.list exists skip
    if [ -f inputs/${runid}.list ]; then
        echo "inputs/${runid}.list exists skip"
        continue
    fi
    ihepdb-export-data ${runid} --output inputs/${runid}
done

# loop to submit with hep_sub -n
# check       hep_sub muon_rec.sh -g junogns -argu "inputs/${runid}.list inputs/${runid}-correlation.list 0 
#/eos/juno/groups/Reconstruction/dingxf/LiDian_data/muon_reco 
#/junofs/users/dingxf/LiDian_data/muon_track_reco/muon_reco/sample/RUN_${runid}_1000_time_sort ${runid}"
output_path="/eos/juno/groups/Reconstruction/dingxf/Realdata_rec/muon_reco"
for runid in "${run_list[@]}"; do
    echo "Submitting RUN_${runid}"
    nfiles=$(cat inputs/${runid}.list | wc -l)
    dataset_output_path="/junofs/users/dingxf/LiDian_data/muon_track_reco/process_run_bundle/muon_track_reco_transformer_ver2/sample/RUN_${runid}_2048_time_sort"
    eos root://junoeos01.ihep.ac.cn mkdir -p ${output_path}/user
    mkdir -p ${dataset_output_path}    
    ARGU="inputs/${runid}.list inputs/${runid}-correlation.list %{ProcId} ${output_path} ${dataset_output_path} ${runid}"
    hep_sub muon_rec1.sh -g junogns -n ${nfiles} -argu "$ARGU" -o logfiles/log_${runid}_%{ProcId}.log -e logfiles/err_${runid}_%{ProcId}.err
done