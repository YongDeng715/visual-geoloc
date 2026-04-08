#  Copyright (c) 2023, Laura Hulley, Ivan Moskalenko, Anastasiia Kornilova
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
import cv2
import math
import requests
import os
import io
import time
import numpy as np
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from urllib.request import Request, urlopen
# import rasterio
# from rasterio.transform import from_bounds
# from osgeo import gdal, osr

class MapDownloader:
    """
    A class that allows to download a map from GoogleMaps in tile format
    """

    def __init__(
        self,
        north_west_lat: float,
        north_west_lon: float,
        south_east_lat: float,
        south_east_lon: float,
        zoom: int,
        api_key: str,
        folder_to_save: Path,
        max_retries: int = 5,
        retry_delay: int = 1,
    ):
        """
        :param north_west_lat: Latitude of the northwest point of the map
        :param north_west_lon: Longitude of the northwest point of the map
        :param south_east_lat: Latitude of the southeast point of the map
        :param south_east_lon: Longitude of the southeast point of the map
        :param zoom: Zoom level of the map
        :param api_key: API key for Google Maps API
        :param folder_to_save: Path to save map (including tiles and map_metadata.txt)
        :param max_retries: Maximum number of retries for failed downloads
        :param retry_delay: Delay between retries in seconds
        """
        self.north_west_lat = north_west_lat
        self.north_west_lon = north_west_lon
        self.south_east_lat = south_east_lat
        self.south_east_lon = south_east_lon
        self.zoom = zoom
        self.api_key = api_key
        self.folder_to_save = folder_to_save
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Maximum allowed shape and scale
        self.img_size = 640
        self.img_scale = 2
        self.map_type = "satellite"

        # The number of pixels from the bottom to be cropped to remove the watermark
        self.bottom_crop = 50
        self.crop_scale = self.bottom_crop / (self.img_size * self.img_scale)

        # Magic constants
        self.map_size = 256
        self.map_scale = math.pow(2, zoom) / (self.img_size / self.map_size)

        # Create save directory if it doesn't exist
        os.makedirs(self.folder_to_save, exist_ok=True)
        ### denote map tiles have downloaded
        self.download_tiles = False

    def __lat_lon_to_point(self, lat, lon):
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

    def __point_to_lat_lon(self, x, y):
        lon = x / self.map_size * 360 - 180

        n = math.pi - 2 * math.pi * y / self.map_size
        lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

        return lat, lon

    def __get_image_bounds(self, lat, lon):
        centre_x, centre_y = self.__lat_lon_to_point(lat, lon)

        south_east_x = centre_x + (self.map_size / 2) / self.map_scale
        south_east_y = (
            centre_y
            + (self.map_size / 2 - self.map_size * self.crop_scale) / self.map_scale
        )
        bottom_right_lat, bottom_right_lon = self.__point_to_lat_lon(
            south_east_x, south_east_y
        )

        north_west_x = centre_x - (self.map_size / 2) / self.map_scale
        north_east_y = centre_y - (self.map_size / 2) / self.map_scale
        top_left_lat, top_left_lon = self.__point_to_lat_lon(north_west_x, north_east_y)

        return top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon

    def __get_lat_step(self, lat, lon):
        point_x, point_y = self.__lat_lon_to_point(lat, lon)

        stepped_point_y = point_y - (
            (self.map_size - self.map_size * self.crop_scale) / self.map_scale
        )
        new_lat, _ = self.__point_to_lat_lon(point_x, stepped_point_y)

        lat_step = lat - new_lat

        return lat_step

    def __request_image(self, lat, lon):
        center = str(lat) + "," + str(lon)
        url = (
            "https://maps.googleapis.com/maps/api/staticmap?center="
            + center
            + "&zoom="
            + str(self.zoom)
            + "&size="
            + str(self.img_size)
            + "x"
            + str(self.img_size)
            + "&key="
            + self.api_key
            + "&maptype="
            + self.map_type
            + "&scale="
            + str(self.img_scale)
        )
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                print(f"Download failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)
        
        return None

    def download_map(self):
        start_corners = self.__get_image_bounds(
            self.north_west_lat, self.north_west_lon
        )
        lon_step = start_corners[3] - start_corners[1]

        lat = self.north_west_lat
        index = 0
        
        # Calculate total number of tiles
        total_lat_steps = 0
        temp_lat = self.north_west_lat
        while temp_lat >= self.south_east_lat:
            lat_step = self.__get_lat_step(temp_lat, self.north_west_lon)
            temp_lat += lat_step
            total_lat_steps += 1

        total_lon_steps = math.ceil((self.south_east_lon - self.north_west_lon) / lon_step)
        total_tiles = total_lat_steps * total_lon_steps

        print(f"Total tiles to download: {total_tiles} ({total_lat_steps} x {total_lon_steps})")

        metadata_file = open(self.folder_to_save / "map_metadata.txt", "w")
        metadata_file.write(
            "filename top_left_lat top_left_lon bottom_right_lat bottom_right_lon\n"
        )

        with tqdm(total=total_tiles, desc="Downloading map tiles") as pbar:
            while lat >= self.south_east_lat:
                lon = self.north_west_lon

                while lon <= self.south_east_lon:
                    filename = f"{str(index).zfill(4)}.png"
                    path_to_image = str(self.folder_to_save / filename)

                    # Skip if file already exists
                    if not os.path.exists(path_to_image):
                        try:
                            image = self.__request_image(lat, lon)
                            if image:
                                with open(path_to_image, "wb") as image_file:
                                    image_file.write(image)
                                    image = cv2.imread(path_to_image)
                                    image = image[: -self.bottom_crop]
                                    cv2.imwrite(path_to_image, image)
                        except Exception as e:
                            print(f"\nFailed to download tile at lat={lat}, lon={lon}: {str(e)}")
                            pbar.update(1)
                            lon = lon + lon_step
                            index += 1
                            continue

                    # Write metadata regardless of whether we downloaded or skipped
                    (
                        top_left_lat,
                        top_left_lon,
                        bottom_right_lat,
                        bottom_right_lon,
                    ) = self.__get_image_bounds(lat, lon)
                    metadata_file.write(
                        f"{filename} {top_left_lat} {top_left_lon} {bottom_right_lat} {bottom_right_lon}\n"
                    )

                    lon = lon + lon_step
                    index += 1
                    pbar.update(1)

                lat_step = self.__get_lat_step(lat, lon)
                lat = lat + lat_step

        metadata_file.close()
        self.download_tiles = True
        print(f"\nDownload completed. Total tiles: {total_tiles}")
        print(f"Tiles saved in: {self.folder_to_save}")

    def create_tiff_map(self):
        """
        Download tiles and create a single GeoTIFF map with geographic information
        1. 读取 metadata.txt，得到每个 tile 的经纬度边界，以及推导出 tile 在网格中的 row/col
        2. 计算整体拼图的地理边界（west, south, east, north）和像素尺寸（full_width, full_height）
        3. 新建一个空的 GeoTIFF，按窗口 (window) 写入每张 tile，做到"逐块"写出，避免内存 OOM
        """
        # First download all tiles
        pass

from .config import SATE_LATLON, SATE_SIZE

if __name__ == "__main__":
    
    #### Google Maps API
    # API_KEY = "YOUR API KEY HERE"
    EXAMPLE_API_KEY = "AIzaSyCQplXJKkB8wS2jNTzgUFMIRalUR25dR2U"
    
    
    region = '04'
    
    tf_lat, tf_lon, br_lat, br_lon = SATE_LATLON[region]
    
    # tf_lat = 32.254036
    # tf_lon = 119.90598
    # br_lat = 32.151018
    # br_lon = 119.954509


    map_downloader = MapDownloader(
        north_west_lat=tf_lat,
        north_west_lon=tf_lon,
        south_east_lat=br_lat,
        south_east_lon=br_lon,
        zoom=17,
        api_key=EXAMPLE_API_KEY,
        folder_to_save=Path(f"data/visloc/{region}/map"),
    )
    map_downloader.download_map()
    