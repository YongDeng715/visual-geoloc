import math

from abc import ABC, abstractmethod
from typing import Tuple

from .map_tile import MapTile

class GeoReferencer(ABC):
    @abstractmethod
    def get_lat_lon(
        self,
        map_tile: MapTile,
        pixel: Tuple[int, int],
        resize: int | Tuple[int, int] = None,
    ) -> Tuple[float, float]:
        """
        Finds geographic coordinates of a given pixel on a satellite image

        :param map_tile: Satellite map tile
        :param pixel: Pixel coordinates
        :param resize: The image resize parameter that was used in keypoint matching
        :return: Latitude and longitude of the pixel
        """
        pass
    
    

class GoogleMapsReferencer(GeoReferencer):
    def __init__(self, zoom):
        self.zoom = zoom

        # Magic constants
        self.map_size = 256
        self.img_size = 640
        self.scale = math.pow(2, self.zoom) / (self.img_size / self.map_size)

    def __lat_lon_to_world(self, lat, lon):
        x = (lon + 180) * (self.map_size / 360)
        y = (
            (
                1
                - math.log(
                    math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
                )
                / math.pi
            )
            / 2
        ) * self.map_size

        return x, y

    def __world_to_lat_lon(self, x, y):
        lon = x / self.map_size * 360 - 180

        n = math.pi - 2 * math.pi * y / self.map_size
        lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

        return lat, lon

    def get_lat_lon(
        self,
        map_tile: MapTile,
        pixel: Tuple[int, int],
        resize: int | Tuple[int, int] = None,
    ) -> Tuple[float, float]:
        top_left_x, top_left_y = self.__lat_lon_to_world(
            map_tile.top_left_lat, map_tile.top_left_lon
        )
        resize_x, resize_y = self.img_size * 2, self.img_size * 2
        if resize is not None:
            height, width = map_tile.shape
            if type(resize) is tuple:
                new_height, new_width = resize
                if width > new_width:
                    resize_x = resize_x * (new_width / width)
                if height > new_height:
                    resize_y = resize_y * (new_height / height)
            elif type(resize) is int:
                tile_size = max(height, width)
                if tile_size > resize:
                    resize_x = resize_x * (resize / tile_size)
                    resize_y = resize_x
            else:
                raise ValueError("Resize param should be int or Tuple[int, int]")

        desired_x = top_left_x + (self.map_size * abs(pixel[0]) / resize_x) / self.scale
        desired_y = top_left_y + (self.map_size * abs(pixel[1]) / resize_y) / self.scale

        lat, lon = self.__world_to_lat_lon(desired_x, desired_y)
        return lat, lon
    

from aero_vloc.utils.aero_utils import get_new_size
    
class LinearReferencer(GeoReferencer):
    def get_lat_lon(
        self,
        map_tile: MapTile,
        pixel: Tuple[int, int],
        resize: int | Tuple[int, int] = None,
    ) -> Tuple[float, float]:
        height, width = map_tile.shape
        if resize is not None:
            if type(resize) is tuple:
                height, width = resize
            elif type(resize) is int:
                height, width = get_new_size(height, width, resize)
            else:
                raise ValueError("Resize param should be int or Tuple[int, int]")

        lat = map_tile.top_left_lat + (abs(pixel[1]) / height) * (
            map_tile.bottom_right_lat - map_tile.top_left_lat
        )
        lon = map_tile.top_left_lon + (abs(pixel[0]) / width) * (
            map_tile.bottom_right_lon - map_tile.top_left_lon
        )
        return lat, lon