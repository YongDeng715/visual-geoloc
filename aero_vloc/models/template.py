import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod


class VPRModel(nn.Module):
    def __init__(self, device: str | None = None):
        """
        :param gpu_index: The index of the GPU to be used
        """
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Running inference on device "{}"'.format(self.device))

    @abstractmethod
    def get_image_descriptor(self, image: np.ndarray):
        """
        Gets the descriptor of the image given
        :param image: Image in the OpenCV format
        :return: Descriptor of the image
        """
        pass