import torch
import numpy as np
import cv2


def featuremap_to_greymap(feature_map):
    """
    feature_map: (C, sizey, sizex)
    grey_map: (sizey, sizex)
    """
    if len(feature_map.shape) == 3:
        feature_map = feature_map.unsqueeze(dim=0) # (b, c, sizey, sizex)
    elif len(feature_map.shape) == 4:
        pass
    else:
        raise NotImplementedError 

    # 1. GPA, (B, C, sizey, sizex) -> (B, C, 1, 1)
    channel_weights = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1,1))

    # 2. reweighting sum cross channels, (B, C, sizey, sizex) -> (B, sizey, sizex) -> (sizey, sizex)
    reduced_map = (channel_weights * feature_map).sum(dim=1).squeeze(dim=0)

    # 3. clamp
    reduced_map = torch.relu(reduced_map)

    # 4. normalize
    a_min = torch.min(reduced_map)
    a_max = torch.max(reduced_map)
    normed_map = (reduced_map - a_min) / (a_max - a_min)

    # 5. output
    grey_map = normed_map

    return grey_map


def greymap_to_rgbimg(map_grey, background=None, background_ratio=0.2, CHW_format=False):
    """
    map_grey: np, (sizey, sizex), values in 0-1
    background: np, (sizey, sizex, 3), values in 0-255.
    """
    if background is None:
        background = np.zeros((map_grey.shape[0], map_grey.shape[1], 3))

    map_uint8 = (255 * map_grey).astype(np.uint8) # 0-255
    map_bgr = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET) # 0-255
    map_rbg = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
    map_img = map_rbg + background_ratio * background
    map_img = np.clip(map_img, a_min=0, a_max=255).astype(np.uint8)

    if CHW_format:
        # (sizey, sizex, 3) -> (3, sizey, sizex)
        map_img = np.transpose(map_img, (2,0,1))

    return map_img