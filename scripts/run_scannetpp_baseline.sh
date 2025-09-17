#!/bin/bash

exp_name=$1
export CUDA_VISIBLE_DEVICES=$2  # CUDA device to use

datasets=('8a20d62ac0' '94ee15e8ba' '7831862f02' 'a29cccc784')
# datasets=('8a20d62ac0')

workspaces=()
for dataset in "${datasets[@]}"; do
    workspace="output/${exp_name}/${dataset}"
    workspaces+=("$workspace")
done

for i in ${!datasets[@]}; do
    dataset="dataset/scannetpp/${datasets[$i]}"
    workspace=${workspaces[$i]}
    
    echo "Processing dataset: $dataset with workspace: $workspace"
    
    python train_baseline.py --dataset Scannetpp --image dslr/undistorted_images --source_path $dataset --model_path $workspace --eval  --n_views 6 \
            --dust3r_min_conf_thr 1 --densify_grad_threshold 1e10 \

    python render.py --source_path $dataset --model_path $workspace --iteration 10000

    python metrics.py --source_path $dataset --model_path $workspace --iteration 10000

done

python get_avg_results_scannetpp.py -m ${exp_name}