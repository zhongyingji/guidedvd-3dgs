#!/bin/bash

exp_name=$1
export CUDA_VISIBLE_DEVICES=$2  # CUDA device to use

datasets=('8a20d62ac0' '94ee15e8ba' '7831862f02' 'a29cccc784')

workspaces=()
for dataset in "${datasets[@]}"; do
    workspace="output/${exp_name}/${dataset}"
    workspaces+=("$workspace")
done

for i in ${!datasets[@]}; do
    dataset="dataset/scannetpp_sparse/${datasets[$i]}"
    workspace=${workspaces[$i]}
    
    echo "Processing dataset: $dataset with workspace: $workspace"
    
    python train_scannetpp_guidedvd_hybrid_traj.py --dataset Scannetpp --image dslr/undistorted_images \
            --source_path $dataset --model_path $workspace --eval  --n_views 6 \
            --dust3r_min_conf_thr 1 \
            --start_sample_pseudo 0 --sample_pseudo_interval 1 \
            --iterations 10000 \
            --pseudo_cam_weight 0.05 \
            --guidance_gpu_id 1 --guidance_ddim_steps 50 --guidance_vd_iter 260 \
            --use_trajectory_pool \
            --pseudo_cam_lpips --pseudo_cam_lpips_weight 0.1 \
            --guidance_with_lpips --scannetpp_newres

    python render.py --source_path $dataset --model_path $workspace --iteration 10000

    python metrics.py --source_path $dataset --model_path $workspace

done

python get_avg_results_scannetpp.py -m ${exp_name}