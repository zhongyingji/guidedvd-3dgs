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
    
    echo "Processing dataset: $dataset with workspace: $workspace. "
    
    python train_replica_guidedvd_tworenderer.py --dataset Replica --image rgb \
            --source_path $dataset --model_path $workspace --eval  --n_views 6 \
            --dust3r_min_conf_thr 1 \
            --start_sample_pseudo 0 --sample_pseudo_interval 1 \
            --iterations 10000 \
            --pseudo_cam_weight 0.05 \
            --guidance_gpu_id 1 --guidance_ddim_steps 50 --guidance_vd_iter 260 \
            --use_trajectory_pool \
            --pseudo_cam_lpips --pseudo_cam_lpips_weight 0.1 \

    python render.py --source_path $dataset --model_path $workspace --iteration 10000
    # python render.py --source_path $dataset --model_path $workspace --iteration 10000 --render_depth --skip_train

    python metrics.py --source_path $dataset --model_path $workspace

done

python get_avg_results_replica.py -m ${exp_name}