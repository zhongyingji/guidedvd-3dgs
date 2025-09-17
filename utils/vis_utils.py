import torch
import numpy as np
import cv2
import torchvision
from torchvision.transforms.functional import to_pil_image


def plot_images(images, weight_map, save_image_name):
    # images: [n, 3, h, w]
    # weight_map: [1, n-2, h, w]
    n, _, h, w = images.shape
    weight_map = weight_map[0] # [n-2, h, w]
    weight_map_norm = (weight_map-weight_map.min())/(weight_map.max()-weight_map.min())
    weight_map_norm = weight_map_norm.detach().cpu().numpy()
    
    weight_map_colored = []
    for wmn in weight_map_norm: 
        weight_map_colored.append(cv2.applyColorMap((wmn*255).astype(np.uint8), cv2.COLORMAP_JET)) # [h, w, 3]
    weight_map_colored = np.stack(weight_map_colored, 0)/255. # [n-2, h, w, 3]
    weight_map_colored = torch.tensor(weight_map_colored).permute(0, 3, 1, 2) # n-2, 3, h, w
    blank_images = torch.ones(2, 3, h, w)


    weight_map_full = torch.cat((blank_images, weight_map_colored), dim=0) # n, 3, h, w
    grid_images = torchvision.utils.make_grid(images.detach().cpu(), nrow=n, padding=2) # 3, h, n*w
    grid_weight_map = torchvision.utils.make_grid(weight_map_full, nrow=n, padding=2)
    combined_grid = torch.cat((grid_images, grid_weight_map), dim=1)
    combined_image_pil = to_pil_image(combined_grid)
    combined_image_pil.save(save_image_name)