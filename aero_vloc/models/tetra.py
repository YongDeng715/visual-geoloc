import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as tvf

from typing import Iterable
from PIL import Image

# from aero_vloc.utils import transform_image_for_vpr

class TETRA(nn.Module):
    """
    Wrapper for [TETRA-VPR](https://github.com/OliverGrainge/TeTRA-VPR)
    """
    
    def __init__(self,
                 path_to_weights: str = "./weights/pkgs/tetra_boq.pth",
                 resize: int = 322,
                 aggregation_arch: str = "BoQ",
                 device: torch.device = "cpu"):
        super().__init__()
        assert aggregation_arch in ["BoQ", "SALAD"], \
            "aggregation_arch must be one of ['BoQ', 'SALAD']"
        self.resize = resize
        self.device = device
        self.model = torch.hub.load(
            repo_or_dir="OliverGrainge/TeTRA-VPR",
            # repo_or_dir=r"D:\torch_cache\hub\OliverGrainge_TeTRA-VPR_main",
            # repo_or_dir="/mnt/d/torch_cache/hub/OliverGrainge_TeTRA-VPR_main",
            model="TeTRA",
            aggregation_arch=aggregation_arch,
            pretrained=True,
            # force_reload=True
        )
        
        checkpoint = torch.load(path_to_weights, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)
        
    def forward(self, x, binary_desc: bool=True):
        return self.model(x, binary_desc=binary_desc)
    
    def get_image_descriptor(self, image: np.ndarray):
        image = self.image_loader(image, self.resize).to(self.device)
        _, h, w = image.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_cropped = tvf.CenterCrop((h_new, w_new))(image)[None, ...]
        
        with torch.inference_mode():
            descriptor = self.model(img_cropped, binary_desc=True)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor
    
    @staticmethod
    def image_loader(image: np.ndarray, resize: int):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = tvf.Compose([
            tvf.Resize((resize, resize), interpolation=tvf.InterpolationMode.BILINEAR),
            tvf.ToTensor(),
            tvf.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform(image)
    
# %% test

def test(img_folder_path):
    import os
    import cv2
    from glob import glob
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = TETRA(
    #     path_to_weights="./weights/pkgs/tetra_boq.pth",
    #     resize=322,
    #     aggregation_arch="BoQ",
    #     device=device
    # )

    model = TETRA(
        path_to_weights="./weights/pkgs/tetra_salad.pth",
        resize=322,
        aggregation_arch="SALAD",
        device=device
    )
    img_paths = glob(os.path.join(img_folder_path, "*.jpg")) + \
        glob(os.path.join(img_folder_path, "*.png"))
        
    features, filenames = [], []
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        img = cv2.imread(img_path)
        with torch.no_grad():
            # output as `np.ndarray`
            feature = model.get_image_descriptor(img)
        
        features.append(feature)
        filenames.append(os.path.basename(img_path))
        
    features = np.vstack(features)
    print(f"shape: {features.shape}, dtype: {features.dtype}")
        
        
    if isinstance(features, list):
        features = np.vstack(features)

    print(f"shape: {features.shape}")

if __name__ == "__main__":
    img_folder_path = "./data/nardo-air/test_40_midref_rot0/query_images"
    test(img_folder_path)