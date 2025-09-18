For the ScanNet++ dataset, the 3DGS model trained from the below scripts will perform slightly better.  


### Train the Guidedvd 3DGS
We use a hybrid trajectory initialization for better performance on the ScanNet++ dataset. 
```bash
bash scripts/run_scannetpp_guidedvd_hybrid_traj.sh scannetpp_guidedvd_hybrid_traj 0,1
```


### Results
As shown below, using hybrid trajectory initialization brings slight performance improvement. The performance reported in the paper on the ScanNet++ dataset was that of this model (with a PSNR of 23.89 in the paper).
| Replica | PSNR&uparrow; | SSIM&uparrow; | LPIPS&downarrow;|
| :--- | :---: | :---: | :---: |
| guidedvd_3dgs | 23.77 | 0.848 | 0.187 |
| +hybrid trajectory  | 23.88 | 0.850 | 0.183 |
