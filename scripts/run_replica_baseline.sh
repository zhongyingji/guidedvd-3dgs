#!/bin/bash

exp_name=$1
export CUDA_VISIBLE_DEVICES=$2  # CUDA device to use

datasets=('office_2/Sequence_2' 'office_3/Sequence_1' 'office_4/Sequence_2' 'room_0/Sequence_2' 'room_1/Sequence_1' 'room_2/Sequence_1')
# datasets=('room_0/Sequence_2')

workspaces=()
for dataset in "${datasets[@]}"; do
    workspace="output/${exp_name}/${dataset}"
    workspaces+=("$workspace")
done

for i in ${!datasets[@]}; do
    dataset="dataset/Replica/${datasets[$i]}"
    workspace=${workspaces[$i]}
    
    echo "Processing dataset: $dataset with workspace: $workspace"
    
    python train_baseline.py --dataset Replica --image rgb --source_path $dataset --model_path $workspace --eval  --n_views 6 \
            --dust3r_min_conf_thr 1 --densify_grad_threshold 1e10 --position_lr_init 0. --position_lr_final 0. \

    python render.py --source_path $dataset --model_path $workspace --iteration 10000

    python metrics.py --source_path $dataset --model_path $workspace --iteration 10000

done

python get_avg_results_replica.py -m ${exp_name}