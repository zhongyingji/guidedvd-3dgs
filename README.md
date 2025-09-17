# Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs

<a href='https://openaccess.thecvf.com/content/CVPR2025/papers/Zhong_Taming_Video_Diffusion_Prior_with_Scene-Grounding_Guidance_for_3D_Gaussian_CVPR_2025_paper.pdf'><img src='https://img.shields.io/badge/CVPR-Highlight-red'></a> &nbsp; <a href='https://zhongyingji.github.io/guidevd-3dgs'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; <a href='https://arxiv.org/abs/2503.05082'><img src='https://img.shields.io/badge/arXiv-2503.05082-b31b1b.svg'></a> 

This repository contains the code release for the CVPR 2025 (Highlight) project 
> [**Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs**](hhttps://arxiv.org/abs/2503.05082),  
> [Yingji Zhong](https://github.com/zhongyingji), [Zhihao Li](https://scholar.google.com/citations?user=4cuefJ0AAAAJ&hl=en), [Dave Zhenyu Chen](https://daveredrum.github.io/), [Lanqing Hong](https://racheltechie.github.io/), [Dan Xu](https://www.danxurgb.net/)  
> Computer Vision and Pattern Recognition (CVPR), 2025

<br/>

![Teaser Image](assets/teaser.png)
📖 We tackle the critical issues of (a) extrapolation and (b) occlusion in sparse-input 3DGS by leveraging a video diffusion model. Vanilla generation often suffers from inconsistencies within the generated sequences (as highlighted by the yellow arrows), leading to black shadows in the rendered images. In contrast, our scene-grounding generation produces consistent sequences, effectively addressing these issues and enhancing overall quality (c), as indicated by the blue boxes. The numbers refer to PSNR values.

<br/>

## 🔧 Environmental Setup
### Build Environment
```bash
conda create -n guidedvd python=3.9
conda activate guidedvd

# We recommend using pytorch 1.13.1 and CUDA 11.7. 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
# pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py39_cu117_pyt1131.tar.bz2

pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn
```

### Download Checkpoints
In this project, we use [DUSt3R](https://github.com/naver/dust3r) for point cloud initialization and [ViewCrafter](https://github.com/Drexubery/ViewCrafter) for video sequence generation. Download the `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` from [here](https://github.com/naver/dust3r) and `model.ckpt` from [here](https://huggingface.co/Drexubery/ViewCrafter_25/blob/main/model.ckpt). Put them under `third_party/ViewCrafter/checkpoints/`. 


## 📊 Data Preparation
We use Replica and ScanNet++ datasets in this project. We download the Replica dataset from the [link](https://www.dropbox.com/scl/fo/puh6djua6ewgs0afsswmz/AHiqYQQv7ydbWMcAULTZk1w/Replica_Dataset?dl=0&rlkey=ep5495umv628y2sk8hvnh8msc&subfolder_nav_tracking=1) provided by [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf). We select 6 scenes: `office_2`, `office_3`, `office_4`, `room_0`, `room_1`, `room_2`. For the ScanNet++ dataset, we download it from the [official link](https://kaldir.vc.in.tum.de/scannetpp/) and select 4 scenes: `8a20d62ac0`, `94ee15e8ba`, `7831862f02`, `a29cccc784`. The downloaded datasets are placed in `dataset/`, which is organized as:  
```
├── dataset                                                              
│   ├── Replica                                                                                                  
│   │   └── office_2                                                            
|   |   └── ...
|   ├── scannetpp
|   |   └── 8a20d62ac0
|   |   └── ...
```
### Convert to Colmap Format
We convert the camara data into colmap format. 
```bash
# Replace the path in the following files. 

python tools/replica_to_colmap.py
python tools/scannetpp_to_colmap.py
```
After the conversion, there should be a `sparse/0/` directory under each scene directory, e.g., `dataset/Replica/office_3/Sequence_1/sparse/0/`. 

### Get DUSt3R Point Cloud
We use [DUSt3R](https://github.com/naver/dust3r) to get the point cloud from sparse inputs for each scene. This might take a while. The setting can be referred to [here](https://github.com/zhongyingji/guidedvd-3dgs/blob/master/scene/dataset_readers.py#L339-L470). 
```bash
python tools/get_replica_dust3r_pcd.py
python tools/get_scannetpp_dust3r_pcd.py
```
Then you should see a directory containing a point cloud for each scene, e.g., `dust3r_results/Replica_6v_thr1_trimeshsave_minconf1/office2_seq2/sparse/0/points3D.ply`. 

## 🚀 Quick Start

### Train the Baseline 3DGS
We train a baseline 3DGS for each scene to provide scene-grounding guidance later. 
```bash
bash scripts/run_replica_baseline.sh replica_baseline 0
bash scripts/run_scannetpp_baseline.sh scannetpp_baseline 0

# 0 refers to GPU id, feel free to replace it with available gpus. 
# Both `replica_baseline` and `scannetpp_baseline` are directories for saving the outputs of the baseline model. You can change them to other names. 
```
After training, all quantitative results and qualitative results can be found at `output/replica_baseline/` and `output/scannetpp_baseline`. 

### Train the Guidedvd 3DGS

```bash
# Remember to replace the baseline 3DGS model path in `train_guidedvd.py` (L60-70) with the directory name that you used to save the baseline model!
bash scripts/run_replica_guidedvd.sh replica_guidedvd 0,1
bash scripts/run_scannetpp_guidedvd.sh scannetpp_guidedvd 0,1

# 0,1 refers to GPU ids, feel free to replace them with available gpus. 
```
We use two gpus to train a guidedvd 3DGS model, one for running the video diffusion model, while the other for optimizing the 3DGS. **The peak memory of the gpu running the video diffusion model will reach 32G**. On V100 GPUS, training one scene will typically take 3-4 hours. If your GPU memory is less than 32G, make the resolution smaller in `train_guidedvd.py` (L97-98, L110-111). 

After the training has completed, there are many visualization results provided. For example, under the directory of `output/replica_guidedvd/office_3/`: 
```
define_traj*/ : the renderings of sampled poses around each input view
vd/train_iter*/: 
  gs_render.mp4: the rendered sequence by the baseline 3DGS
  gs_render_alpha.mp4: the mask sequence decided by the baseline 3DGS
  diffusion0.mp4: the generated sequence from the video diffusion model
vd/pred_x0/train_iter*/: generated sequence from each ddim step
```

We provide the example results [here](./assets/results/). Note that in the paper, we report the lpips with alexnet, and normalize the input rgb to [-1, 1]. 

NOTE: we provide another way of training a slighly better model on the Replica dataset. Please refer to [here](./assets/replica.md) if you are interested in it. 


## 📚 Citation
Please kindly cite the followings if you find our work helpful. 
```
@inproceedings{zhong2025taming,
    title={Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs},
    author={Zhong, Yingji and Li, Zhihao and Chen, Dave Zhenyu and Hong, Lanqing and Xu, Dan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
@article{zhong2025empowering,
  title={Empowering sparse-input neural radiance fields with dual-level semantic guidance from dense novel views},
  author={Zhong, Yingji and Zhou, Kaichen and Li, Zhihao and Hong, Lanqing and Li, Zhenguo and Xu, Dan},
  journal={arXiv preprint arXiv:2503.02230},
  year={2025}
}
```

## 🙏 Acknowledgements
Thanks to these great repositories: [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [FSGS](https://github.com/VITA-Group/FSGS), [DUSt3R](https://github.com/naver/dust3r), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [FreeDoM](https://github.com/yujiwen/FreeDoM), [GANeRF](https://github.com/barbararoessle/ganerf) and many other inspiring works in the community. 

## 📧 Contact
If you have any question or collaboration needs, please email `zzhongyj@gmail.com`.

<br/>
<div align="center">
⭐ If you find this project helpful, please consider giving us a star! ⭐
</div> 