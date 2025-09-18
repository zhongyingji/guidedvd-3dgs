For the Replica dataset, the 3DGS model trained from the below scripts will perform slightly better.  

### Train the baseline 3DGS model
Beside initialization, we observe that the DUSt3R point cloud can also provide the multi-view supervision by the simple projection, e.g., project the point cloud on multiple pre-defined cameras. 

After obtaining the DUSt3R point cloud, run the following script: 
```bash
python scripts/get_replica_dust3r_project_2d.py
```
Then there should be a directory `projected_dir/` containing projected images and masks of all scenes. 

With the DUSt3R initialization, we can train the baseline 3DGS model not only from sparse multi-view supervision, but also from dense multi-view supervision. The multi-view supervision is projected from DUSt3R point cloud. 
```bash
bash scripts/run_replica_baseline_with_project_cam.sh replica_baseline_with_project_cam 0
```

### Train the Guidedvd 3DGS
We now have two baseline 3DGS models, i.e., `replica_baseline` and `replica_baseline_with_project_cam`. The performance of `replica_baseline` is slightly lower than `replica_baseline_with_project_cam`, but `replica_baseline` can provide a more accurate mask of deciding the invisibile regions during rendering. 
We thus use these two baselines together for training the ultimate model. Specifically, we use `replica_baseline` to render an alpha map to decide the mask, while we use `replica_baseline_with_project_cam` to render rgb map to serve as the guidance during ddim sampling. We can train the final model with the following script: 
```bash
bash scripts/run_replica_guidedvd_tworenderer.sh replica_guidedvd_tworenderer 0,1
```
NOTE: for the guidedvd 3DGS training, we **do not** use the projected images from DUSt3R for supervision. 

### Results
As shown below, training the guidedvd 3DGS with two baseline models brings slight performance improvement. The performance reported in the paper on the Replica dataset was that of this model (with a PSNR of 26.35 in the paper).
| Replica | PSNR&uparrow; | SSIM&uparrow; | LPIPS&downarrow;|
| :--- | :---: | :---: | :---: |
| guidedvd_3dgs | 26.05 | 0.871 | 0.127 |
| +tworenderer  | 26.29 | 0.872 | 0.123 |
