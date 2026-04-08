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
import einops as ein
import fast_pytorch_kmeans as fpk
import numpy as np
import torch

from pathlib import Path
from torch.nn import functional as F
from typing import Union


class VLAD:
    """
    An implementation of VLAD algorithm given database and query
    descriptors.

    Constructor arguments:
    - num_clusters:     Number of cluster centers for VLAD
    - desc_dim:         Descriptor dimension. If None, then it is
                        inferred when running `fit` method.
    - intra_norm:       If True, intra normalization is applied
                        when constructing VLAD
    - norm_descs:       If True, the given descriptors are
                        normalized before training and predicting
                        VLAD descriptors. Different from the
                        `intra_norm` argument.
    - dist_mode:        Distance mode for KMeans clustering for
                        vocabulary (not residuals). Must be in
                        {'euclidean', 'cosine'}.
    - vlad_mode:        Mode for descriptor assignment (to cluster
                        centers) in VLAD generation. Must be in
                        {'soft', 'hard'}
    - soft_temp:        Temperature for softmax (if 'vald_mode' is
                        'soft') for assignment
    - cache_dir:        Directory to cache the VLAD vectors. If
                        None, then no caching is done. If a str,
                        then it is assumed as the folder path. Use
                        absolute paths.

    Notes:
    - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
        Proceedings of the IEEE conference on Computer Vision and
        Pattern Recognition. 2013.
    """

    def __init__(
        self,
        num_clusters: int,
        c_centers: torch.Tensor,
        desc_dim: Union[int, None] = None,
        **kwargs,
        # intra_norm: bool = True,
        # norm_descs: bool = True,
        # dist_mode: str = "cosine",
        # vlad_mode: str = "hard",
        # soft_temp: float = 1.0,
    ) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.mode = kwargs["dist_mode"] if "dist_mode" in kwargs else "cosine"
        
        # Set in the training phase, set the caching
        self.c_centers = c_centers
        self.kmeans = None

    # Generate cluster centers
    def fit(self, train_descs: Union[np.ndarray, torch.Tensor, None]):
        """
            Using the training descriptors, generate the cluster 
            centers (vocabulary). Function expects all descriptors in
            a single list (see `fit_and_generate` for a batch of 
            images).
            If the cache directory is valid, then retrieves cluster
            centers from there (the `train_descs` are ignored). 
            Otherwise, stores the cluster centers in the cache 
            directory (if using caching).
            
            Parameters:
            - train_descs:  Training descriptors of shape 
                            [num_train_desc, desc_dim]. If None, then
                            caching should be valid (else ValueError).
        """
        # Clustering to create vocabulary
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        # Check if cache exists
        if train_descs is None:
            if self.c_centers is None:
                raise ValueError("No training descriptors given for cluster centers")
            self.kmeans.centroids = self.c_centers
            if self.desc_dim is None:
                self.desc_dim = self.c_centers.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
        else:
            if type(train_descs) == np.ndarray:
                train_descs = torch.from_numpy(train_descs).to(torch.float32)
            if self.desc_dim is None:
                self.desc_dim = train_descs.shape[1]
                print(f"Desc dim set to {self.desc_dim}")
            train_descs = F.normalize(train_descs)
            self.kmeans.fit(train_descs)
            self.c_centers = self.kmeans.centroids
    
    def fit_and_generate(self, train_descs: Union[np.ndarray, torch.Tensor]
                         ) -> torch.Tensor:
        """
            Given a batch of descriptors over images, `fit` the VLAD
            and generate the global descriptors for the training
            images. Use only when there are a fixed number of 
            descriptors in each image.
            
            Parameters:
            - train_descs:  Training image descriptors of shape
                            [num_imgs, num_descs, desc_dim]. There are
                            'num_imgs' images, each image has 
                            'num_descs' descriptors and each 
                            descriptor is 'desc_dim' dimensional.
            
            Returns:
            - train_vlads:  The VLAD vectors of all training images.
                            Shape: [num_imgs, num_clusters*desc_dim]
        """
        # Generate vocabulary
        all_descs = ein.rearrange(train_descs, "n k d -> (n k) d")
        self.fit(all_descs)
        # For each image, stack VLAD
        return torch.stack([self.generate(tr) for tr in train_descs])
    
    def generate(self, query_descs: torch.Tensor) -> torch.Tensor:
        """
            Given the query descriptors, generate a VLAD vector. Call
            `fit` before using this method. Use this for only single
            images and with descriptors stacked. Use function
            `generate_multi` for multiple images.
            
            Parameters:
            - query_descs:  Query descriptors of shape [b, n_q, desc_dim] 
                            where 'b' is the batch size, 
                            'n_q' is number of 'desc_dim' descriptors.
            Returns:
            - n_vlas:   Normalized VLAD: [b, num_clusters*desc_dim]
        """
        if len(query_descs.shape) == 2:
            query_descs = ein.rearrange(query_descs, "q d -> 1 q d")
        assert query_descs.device == self.c_centers.device
        img_descs = query_descs         # [b, q, d]
        c_centers = self.c_centers      # [c, d]
        # Cluster labels (dot product ~ cosine distance; need max)
        _i1 = F.normalize(img_descs, dim=2)
        _c1 = F.normalize(c_centers, dim=1)
        labels = ein.rearrange(_i1, "b n d -> (b n) d") \
            @ ein.rearrange(_c1, "c d -> d c")
        labels = ein.rearrange(labels, "(b n) c -> b n c", 
                b=img_descs.shape[0], n=img_descs.shape[1])
        labels = labels.argmax(dim=2)   # [b, q, c]
        # Residuals
        residuals = ein.rearrange(_i1, "b n d -> b n 1 d") \
            - ein.repeat(c_centers, "c d -> b 1 c d", b=img_descs.shape[0])
        b, q, c, d = residuals.shape
        b_, q_ = labels.shape
        un_vlad = torch.zeros(b, c, d, device=c_centers.device)
        assert b == b_ and q == q_
        # TODO: Probably make this batching more efficient
        """
            Can we make this even more efficient (instead of using two for loops)?
            See: 
            - https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
        """
        for bi in range(b):
            for ci in range(c):
                # Find all indices where label matches
                un_vlad[bi, ci] += residuals[bi, labels[bi] == ci][:, ci, :].sum(dim=0)
        n_vlad = F.normalize(un_vlad, dim=2)
        raw_vlad = ein.rearrange(n_vlad, "b c d -> b (c d)")
        vlad = F.normalize(raw_vlad, dim=1) # [b, c*d]
        return vlad
    
    def generate_multi(self, 
            multi_query: Union[np.ndarray, torch.Tensor, list],
            **kwargs) -> Union[torch.Tensor, list]:
        """
            Given query descriptors from multiple images, generate
            the VLAD for them.
            
            Parameters:
            - multi_query:  Descriptors of shape [n_imgs, n_kpts, d]
                            There are 'n_imgs' and each image has
                            'n_kpts' keypoints, with 'd' dimensional
                            descriptor each. If a List (can then have
                            different number of keypoints in each 
                            image), then the result is also a list.
            
            Returns:
            - multi_res:    VLAD descriptors for the queries
        """
        return self.generate(multi_query)
