#  Copyright (c) 2023, Nikhil Keetha, Avneesh Mishra, Jay Karhade,
#  Krishna Murthy Jatavallabhula, Sebastian Scherer, Madhava Krishna, Sourav Garg,
#  Ivan Moskalenko, Anastasiia Kornilova
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
import torch

from torch import nn
from torch.nn import functional as F
from typing import Literal

_DINO_V2_MODELS = Literal[
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
]
_DINO_FACETS = Literal["query", "key", "value", "token"]

_DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DinoV2ExtractFeatures:
    """
    Extract features from an intermediate layer in Dino-v2
    """

    def __init__(
        self,
        dino_model: _DINO_V2_MODELS,
        layer: int,
        facet: _DINO_FACETS = "token",
        use_cls=False,
        norm_descs=True,
        device: str = "cpu",
    ) -> None:
        """
        Parameters:
        - dino_model:   The DINO-v2 model to use
        - layer:        The layer to extract features from
        - facet:    "query", "key", or "value" for the attention
                    facets. "token" for the output of the layer.
        - use_cls:  If True, the CLS token (first item) is also
                    included in the returned list of descriptors.
                    Otherwise, only patch descriptors are used.
        - norm_descs:   If True, the descriptors are normalized
        - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].register_forward_hook(
                self._generate_forward_hook()
            )
        else:
            self.fh_handle = self.dino_model.blocks[
                self.layer
            ].attn.qkv.register_forward_hook(self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len : 2 * d_len]
                else:
                    res = res[:, :, 2 * d_len :]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        # print("AnyLoc extractor shape: ", {res.shape})
        return res

    def __del__(self):
        try:
            if hasattr(self, 'fh_handle') and self.fh_handle is not None:
                self.fh_handle.remove()
        except Exception as e:
            # 忽略析构时的任何错误，避免二次报错
            print("Error in __del__ of DinoV2ExtractFeatures:", e)
            pass


class DINOv2(nn.Module):
    """
    Docstring for DINOv2
    
    """
    def __init__(self, 
                 model_name="dinov2_vitb14",
                 num_trainable_blocks=2,
                 norm_layer=False,
                 return_token=False):
        super().__init__()
        
        assert model_name in _DINOV2_ARCHS.keys(), f"Model {model_name} not supported."
        # self.model = torch.hub.load(
        #     repo_or_dir="D:/torch_cache/hub/facebookresearch_dinov2_main",
        #     model=model_name,
        # )
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        
        self.num_channels = _DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f