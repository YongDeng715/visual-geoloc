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
import torch.nn as nn
import torchvision.transforms as tvf

from aero_vloc.utils import transform_image_for_vpr


class SALAD(nn.Module):
    """
    Wrapper for [SALAD](https://github.com/serizba/salad) VPR method
    """

    def __init__(
        self,
        path_to_weights: str = "./weights/pkgs/salad_cliquemining.ckpt",
        resize: int = 800,
        device: torch.device = "cpu",
    ):
        """
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param device: The device to be used for inference
        """
        super().__init__()
        self.resize = resize
        self.device = device
        ### 绕过验证步骤
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        # self.model = torch.hub.load("serizba/salad", "dinov2_salad")
        # self.model = torch.hub.load("/mnt/d/torch_cache/hub/serizba_salad_main", "dinov2_salad", source="local")
        
        ### load local model and weights
        self.model = torch.hub.load(r"D:\torch_cache\hub\serizba_salad_main", "dinov2_salad", source="local", pretrained=False)
        
        checkpoint = torch.load(path_to_weights, map_location=self.device)  ## Using cliqemining model
        
        ### 如果 checkpoint 是 dict 并且有 'model_state_dict' 这个 key，就拿这个；否则直接当作 state_dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)
    
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize).to(self.device)
        _, h, w = image.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_cropped = tvf.CenterCrop((h_new, w_new))(image)[None, ...]
        with torch.no_grad():
            descriptor = self.model(img_cropped)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor
    
# %%

import torchinfo
import cv2
import time
def test():
    model = SALAD(
        path_to_weights="./weights/pkgs/salad_cliquemining.ckpt",
        resize=448,
        device="cuda"
    )

    torchinfo.summary(model, input_size=(16, 3, 448, 448))
    
    img = cv2.imread("./data/visloc/04/drone/04_0001.JPG")
    with torch.no_grad():
        desc = model.get_image_descriptor(img)
    print("Descriptor shape: ", desc.shape)

if __name__ == "__main__":
    test()