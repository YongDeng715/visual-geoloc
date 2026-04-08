#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np
import torch
import torchvision
import torch.nn as nn
import einops as ein

from pathlib import Path
from torchvision import transforms as tvf

from aero_vloc.models.backbones.dinov2_vanilla import DinoV2ExtractFeatures
from aero_vloc.models.aggregators.vlad import VLAD
from aero_vloc.utils import transform_image_for_vpr



# %%
class AnyLocVladDinov2(nn.Module):
    """
        Wrapper around the AnyLoc-VLAD-DINOv2 model in the paper for
        the domain vocabularies (default).
        It basically has the DINOv2 ViT feature extraction and the
        VLAD descriptor construction in a single module.
    """
    def __init__(self, c_centers_path: Path = None,
                dino_model: str = "dinov2_vitg14",
                resize: int = 800,
                layer: int = 31, 
                facet: str = "value", 
                num_c: int = 32, 
                device: torch.device = "cpu"):
        super().__init__()
        # DINOv2 feature extractor
        self.dino_model = dino_model
        self.resize = resize
        self.layer = layer
        self.facet = facet
        self.device = torch.device(device)
        self.dino_extractor = self._get_dino_extractor()
        # VLAD clustering
        
        c_centers_path = r"E:\quantize-vpr\weights\pkgs\anyloc_cluster_centers.pt" if c_centers_path is None else c_centers_path
        c_centers = torch.load(c_centers_path).to(self.device)
        
        self.vlad = VLAD(num_c,
                         c_centers=c_centers,
                         desc_dim=None)
        self.vlad.fit(None) # Load the database (vocabulary/c_centers)
    
    # Extractor
    def _get_dino_extractor(self):
        return DinoV2ExtractFeatures(
            dino_model=self.dino_model, layer=self.layer, 
            facet=self.facet, device=self.device
        )
    
    # Move DINO model to device
    def to(self, device: torch.device):
        self.device = torch.device(device)
        self.dino_extractor = self._get_dino_extractor()
    
    # Wrapper around CUDA
    def cuda(self):
        self.to("cuda")
    
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_pt = x
        shapes = ein.parse_shape(img_pt, "b c h w")
        assert shapes["c"] == 3, "Image(s) must be RGB!"
        assert shapes["h"] % 14 == shapes["w"] % 14 == 0, \
                "Height and width should be multiple of 14 (for patching)"
        img_pt = img_pt.to(self.device)
        # Extract features
        ret = self.dino_extractor(img_pt)   # [b, (nH*nW), dino_dim]
        gds = self.vlad.generate_multi(ret)
        return gds.to(self.device)
    
    def get_image_descriptor(self, image: np.ndarray) -> np.ndarray:
        """
        Get the global descriptor for a given image.

        Parameters:
        - image (np.ndarray): The input image in HWC format.

        Returns:
        - descriptor (np.ndarray): The global descriptor as a 1D numpy array.
        """
        image = transform_image_for_vpr(
            image, self.resize, tvf.InterpolationMode.BICUBIC
        ).to(self.device)
        
        _, h, w = image.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_cropped = tvf.CenterCrop((h_new, w_new))(image)[None, ...]

        with torch.no_grad():
            ret = self.dino_extractor(img_cropped)
        desc = self.vlad.generate_multi(ret)
        return desc.cpu().numpy()[0]    # shape: (49152, )
    

# %%
def test_main(img_folder_path, center_path):
    import os
    import cv2
    from glob import glob
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_file = "output/anyloc_features.pth"
    c_centers = torch.load(center_path)
    
    model = AnyLocVladDinov2(
        c_centers=c_centers,
        dino_model="dinov2_vitg14",
        resize=322,
        device=device
    )
    
    
    img_paths = glob(os.path.join(img_folder_path, "*.jpg")) + \
        glob(os.path.join(img_folder_path, "*.png"))
    
    features, filenames = [], []
    for i, img_path in tqdm(enumerate(img_paths), total=len(img_paths)):
        # img = Image.open(img_path).convert('RGB')
        # img_tensor = transform(img).unsqueeze(0).to(device)
        # with torch.no_grad():
        #     feature = model(img_tensor)
        
        img = cv2.imread(img_path)
        with torch.no_grad():
            # output as `np.ndarray`
            feature = model.get_image_descriptor(img)
        
        features.append(feature)
        filenames.append(os.path.basename(img_path))
        
    if isinstance(features, list):
        features = np.vstack(features)
    torch.save({
        'features': features,
        'filenames': filenames
    }, output_file)
    
    print(f"Features saved to {output_file}, shape: {features.shape}")
    
    
if __name__ == "__main__":
    test_main(
        img_folder_path="./data/nardo-air/test_40_midref_rot0/query_images",
        center_path="./weights/pkgs/anyloc_cluster_centers.pt",
    )
    