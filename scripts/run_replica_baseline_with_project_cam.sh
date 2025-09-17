#!/bin/bash

exp_name=$1
export CUDA_VISIBLE_DEVICES=$2  # CUDA device to use

datasets=('office_2/Sequence_2' 'office_3/Sequence_1' 'office_4/Sequence_2' 'room_0/Sequence_2' 'room_1/Sequence_1' 'room_2/Sequence_1')

workspaces=()
for dataset in "${datasets[@]}"; do
    workspace="output/${exp_name}/${dataset}"
    workspaces+=("$workspace")
done

for i in ${!datasets[@]}; do
    dataset="dataset/Replica/${datasets[$i]}"
    workspace=${workspaces[$i]}
    
    echo "Processing dataset: $dataset with workspace: $workspace"
    
    python train_replica_baseline_with_project_cam.py --dataset Replica --image rgb --source_path $dataset --model_path $workspace \
            --eval --n_views 6 \
            --sample_pseudo_interval 1 --dust3r_min_conf_thr 1 --densify_grad_threshold 1e10 \
            --project_cam_prob 0.8 --project_cam_weight 0.05 \
            --replica_use_project_cam

    python render.py --source_path $dataset --model_path $workspace --iteration 10000

    python metrics.py --source_path $dataset --model_path $workspace --iteration 10000

done

python get_avg_results_replica.py -m ${exp_name}