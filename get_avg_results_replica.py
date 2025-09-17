import os
import json
import numpy as np
from argparse import ArgumentParser

def evaluate(model_path): 
        scenes = ['office_2/Sequence_2', 'office_3/Sequence_1', 'office_4/Sequence_2', 'room_0/Sequence_2', 'room_1/Sequence_1', 'room_2/Sequence_1']


        print(len(scenes))
        results = {"psnr": [], "ssim": [], "lpips": [], "lpips_alex": []}
        root_dir = os.path.join("./output/", model_path[0])
        for scene in scenes:
                file = os.path.join(root_dir, scene, "results.json")
                with open(file) as f:
                        tmp_result = json.load(f)
                tmp_result = tmp_result["ours_10000"]
                results["psnr"].append(tmp_result["PSNR"])
                results["ssim"].append(tmp_result["SSIM"])
                results["lpips"].append(tmp_result["LPIPS"])
                results["lpips_alex"].append(tmp_result["LPIPS_alex"])

        mean_results = {}
        for k, v in results.items():
                mean_results[k+"_all"] = np.mean(v)
        results.update(mean_results)
        print(results)

        with open(os.path.join(root_dir, "results_allscenes.json"), 'w') as fp:
                json.dump(results, fp, indent=True)


if __name__ == "__main__":
        # Set up command line argument parser
        parser = ArgumentParser(description="Avg")
        parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
        args = parser.parse_args()
        evaluate(args.model_paths)