import cv2
import math
import requests
import os
import io
import time
import numpy as np
import pandas as pd  

from typing import Optional
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

class MapDownloaderAug:
    """
    An enhanced map downloader that supports multiple map sources including Google Maps, Esri, and Tianditu
    """

    def __init__(
        self,
        north_west_lat: float,
        north_west_lon: float,
        south_east_lat: float,
        south_east_lon: float,
        zoom: int,
        folder_to_save: Path,
        source: str = "esri",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        retry_delay: int = 3,
    ):
        """
        :param north_west_lat: Latitude of the northwest point of the map
        :param north_west_lon: Longitude of the northwest point of the map
        :param south_east_lat: Latitude of the southeast point of the map
        :param south_east_lon: Longitude of the southeast point of the map
        :param zoom: Zoom level of the map
        :param folder_to_save: Path to save map
        :param source: Map source ('google', 'esri', 'tianditu')
        :param api_key: API key (required for Google Maps and Tianditu)
        :param max_retries: Maximum number of retries for failed downloads
        :param retry_delay: Delay between retries in seconds
        """
        self.north_west_lat = north_west_lat
        self.north_west_lon = north_west_lon
        self.south_east_lat = south_east_lat
        self.south_east_lon = south_east_lon
        self.zoom = zoom
        self.folder_to_save = folder_to_save    ## folder to save original map tiles
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key
        
        self.source = source.lower()
        if self.source not in ["google", "esri", "tianditu", "tiandi", "gaode"]:
            raise ValueError(f"Unsupported map source: {source}")
        
        ### Magic constants
        self.zoom = zoom
        self.img_size = 256     ### img_size(for original map tile)
        self.map_size = 256     ### tile_size
        self.map_scale = math.pow(2, zoom) / (self.img_size / self.map_size)

        # Create save directory
        os.makedirs(self.folder_to_save, exist_ok=True)
        
        if self.api_key is None:
            if self.source == 'tianditu' or self.source == 'tiandi':
                self.api_key = "2f4533e8b46559bee9697bace3432f6f"
            
        ### denote map tiles have downloaded
        self.download_tiles = False
        
        x_min, y_min = self.__lat_lon_to_tile(self.north_west_lat, self.north_west_lon)
        x_max, y_max = self.__lat_lon_to_tile(self.south_east_lat, self.south_east_lon)
        self.total_tiles_ori = (x_max - x_min + 1) * (y_max - y_min + 1)
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.map_info = defaultdict()
        """map_info format:{
            'sate_name':{
                'x': x_tile,
                'y': y_tile,
                'north_west_lat': lat_top,
                'south_east_lat': lat_bottom,
                'north_west_lon': lon_left,
                'south_east_lon': lon_right
            }
        }
        
        """
        
    def __lat_lon_to_tile(self, lat, lon):
        """Convert lat/lon to tile coordinates"""
        n = self.map_scale ## n = 2.0 ** self.zoom
        x_tile = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y_tile = int(
            (
                1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi
            ) / 2.0 * n
        )
        return x_tile, y_tile
    
    def __get_tile_bounds(self, x_tile, y_tile):
        n = self.map_scale
        tile_deg_size = 360.0 / n

        ### compute longtitude of left-top, bottom-right
        lon_left = x_tile * tile_deg_size - 180.0
        lon_right = (x_tile + 1) * tile_deg_size - 180.0
        
        def _tile_y_to_lat(y_tile):
            return math.degrees(
                math.atan(
                    math.sinh(
                        math.pi * (1 - 2 * y_tile / n)
                    )
                )
            )

        lat_top = _tile_y_to_lat(y_tile)
        lat_bottom = _tile_y_to_lat(y_tile + 1)
        
        ### return: lat_tl, lon_tl, lat_br, lon_br
        return lat_top, lon_left, lat_bottom, lon_right
        

    def __get_tile_url(self, x_tile, y_tile):
        """Get the URL for a specific tile based on the source"""
        if self.source == 'google':
            return f"https://mt0.google.com/vt/lyrs=s&hl=en&x={x_tile}&y={y_tile}&z={self.zoom}"
        elif self.source == 'esri':
            return f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{y_tile}/{x_tile}'
        elif self.source == 'tiandi' or self.source == 'tianditu':
            return f'http://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TILEMATRIX={self.zoom}&TILEROW={y_tile}&TILECOL={x_tile}&tk={self.api_key}',
        elif self.source == 'gaode':
            return f"https://webst01.is.autonavi.com/appmaptile?style=6&x={x_tile}&y={y_tile}&z={self.zoom}"
        
       
    def __download_tile(self, x_tile, y_tile):
        """Download a single tile with retry mechanism"""
        url = self.__get_tile_url(x_tile, y_tile)
        
        for attempt in range(self.max_retries):
            try:
                resp = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urlopen(resp, timeout=5)
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                print(f"Download failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay)

    def download_map(self):
        """Download the map tiles for the specified region"""
        ### Calculate tile coordinates for the corners
        # x_min, y_min = self.__lat_lon_to_tile(self.north_west_lat, self.north_west_lon)
        # x_max, y_max = self.__lat_lon_to_tile(self.south_east_lat, self.south_east_lon)
        x_min, y_min = self.x_min, self.y_min
        x_max, y_max = self.x_max, self.y_max

        print(f"Downloading tiles range from {self.source}\nlat:{self.south_east_lat} ->{self.north_west_lat}\nlon:{self.north_west_lon} ->{self.south_east_lon}")
        print(f"Downloading tiles range z={self.zoom}: \nx= {x_min} -> {x_max}, y= {y_min} -> {y_max}")     

        ### Calculate total number of tiles
        total_tiles = self.total_tiles_ori
        print(f"Total tiles to download: {total_tiles} ({x_max-x_min+1} x {y_max-y_min+1})")
        self.total_tiles_ori = total_tiles

        ### Create metadata file of csv file
        metadata_file = open(Path(self.folder_to_save).parent / "map_metadata.txt", "w")
        metadata_file.write("filename top_left_lat top_left_lon bottom_right_lat bottom_right_lon x_tile y_tile\n")
        records = []
        
        index = 0
        with tqdm(total=total_tiles, desc=f"Downloading {self.source} map tiles") as pbar:
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    filename = f"{self.source}_{str(index).zfill(4)}_{x}_{y}_.png"
                    path_to_image = os.path.join(self.folder_to_save, filename)

                    # Skip if file already exists
                    if not os.path.exists(path_to_image):
                        try:
                            image_data = self.__download_tile(x, y)
                            if image_data:
                                image_data.save(path_to_image)
                        except Exception as e:
                            print(f"\nFailed to download tile at x={x}, y={y}: {str(e)}")
                            pbar.update(1)
                            index += 1
                            continue

                    ### Calculate tile bounds
                    top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon = self.__get_tile_bounds(x, y)
                    
                    records.append({
                        'filename': filename,
                        'top_left_lat': top_left_lat,
                        'top_left_lon': top_left_lon,
                        'bottom_right_lat': bottom_right_lat,
                        'bottom_right_lon': bottom_right_lon,
                        'x_tile': x,
                        'y_tile': y
                    })
                    ### Write metadata
                    metadata_file.write(
                        f"{filename} {top_left_lat} {top_left_lon} {bottom_right_lat} {bottom_right_lon}\n"
                    )

                    index += 1
                    pbar.update(1)

        metadata_file.close()
        self.download_tiles = True
        print(f"\nDownload completed. Total tiles: {total_tiles}")
        print(f"Tiles saved in: {self.folder_to_save}")
        csv_path = Path(self.folder_to_save).parent / "map_metadata.csv"
        df = pd.DataFrame.from_records(
            records, 
            columns=['filename', 'top_left_lat', 'top_left_lon', 'bottom_right_lat', 'bottom_right_lon', 'x_tile', 'y_tile']
        )
        df.to_csv(csv_path, index=False)
        print(f"Map metadata saved in csv path: {csv_path}")
        
    def stitch_multi_scale_map(
        self,
        folder_to_save: Path,
        img_size: int = 640,
        stitch_values: list[int]=[2, 3, 4],
        multi_thread: bool=True,
        stitch_workers: int = 8
    ):
        """
        : param folder_to_save: folder to save the stitched map
        : param stitch_values: list of zoom levels to stitch
        : param multi_thread: whether to use multi-threading for stitching
        : param stitch_workers: number of workers for multi-threading
        """
        tiles = {}
        total_num = self.total_tiles_ori
        os.makedirs(folder_to_save, exist_ok=True)
        
        with tqdm(total=total_num, ncols=100, desc='Loading original tiles') as pbar:
            for img_name in os.listdir(self.folder_to_save): ## This folder contains all original map tiles
                tile_path = os.path.join(self.folder_to_save, img_name)
                tile = Image.open(tile_path).convert('RGB')
                
                _name_split = img_name.split('_')   ## img_name: 0000_0_0_.png (index_x_y_.png) or esri_index_x_y_.png
                x, y = int(_name_split[-3]), int(_name_split[-2])
                tiles[(x, y)] = tile    
                pbar.update(1)
        
        ### paste single tiles of zoom_18 into 2*2, 3*3, 4*4 images and store in output_root
        def _process_tiles_section(x, y, grid_size):
            if grid_size % 2 == 0: ## for s=4: x-2 -> x+1, s=2: x-1 -> x, 
                pad1, pad2 = grid_size // 2, grid_size // 2 - 1
            else:   ## for s=3, x-1 -> x+1, y-1 -> y+1
                pad1, pad2 = grid_size // 2, grid_size // 2
            mosaic_name = f'{self.source}_{self.zoom}_{grid_size}_{x-pad1}_{y-pad1}_{x+pad2}_{y+pad2}_.jpg'
            tile_mosaic = Image.new('RGB', (self.map_size * grid_size, self.map_size * grid_size))
            
            if not os.path.isfile(os.path.join(folder_to_save, mosaic_name)):
                for r in range(grid_size):
                    for c in range(grid_size):
                        tile_key = (x + c - pad1, y + r - pad1)
                        if tile_key in tiles:
                            tile_new = tiles[tile_key]
                        else:
                            tile_new = self.__download_tile(x + c - pad1, y + r - pad1)
                            if tile_new is None:
                                print(f"Tile {tile_key} not found. Skipping...")
                                continue
                        left = c * self.map_size
                        upper = r * self.map_size
                        tile_mosaic.paste(tile_new, (left, upper))
                
                tile_mosaic = tile_mosaic.resize((img_size, img_size)) 
                tile_mosaic.save(os.path.join(folder_to_save, mosaic_name))
            else:
                return  
        
        x_min, y_min = self.x_min, self.y_min
        x_max, y_max = self.x_max, self.y_max
        
        if multi_thread:
            with ThreadPoolExecutor(max_workers=stitch_workers) as executor:
                for s in stitch_values: 
                    assert s in [2, 3, 4, 5], 'Now value stitch can only support 2, 3, 4, 5.'
                    if s % 2 == 0:
                        params = [(x, y) \
                            for x in range(x_min + s//2, x_max - s//2 + 1, s) \
                                for y in range(y_min + s//2, y_max - s//2 + 1, s)]
                        total_num = ((x_max - x_min - s + 1) // s + 1) \
                            * ((y_max - y_min - s + 1) // s + 1)
                    else: 
                        params = [(x, y) \
                            for x in range(x_min + s//2, x_max - s//2 + 1, s) \
                                for y in range(y_min + s//2, y_max - s//2 + 1, s)]
                        total_num = ((x_max - x_min - s + 1) // s + 1) \
                            * ((y_max - y_min - s + 1) // s + 1)
                        
                    func = lambda p: _process_tiles_section(*p, grid_size=s)
                    for _ in tqdm(executor.map(func, params), \
                        total=total_num, ncols=120, desc=f'Stitching tiles into {s}*{s} images'):
                        pass
        else:
            for s in stitch_values:
                pass
            
    def create_tiff_map(self, convert_to_tiff=False):
        """
        default generating a png image to show the whole region
        :param convert_to_tiff: whether to convert the stitched png map to tiff format
        """
        # First download all tiles
        if not self.download_tiles:
            self.download_map()
            
        tile_names_list = os.listdir(self.folder_to_save)
        total_tiles = len(tile_names_list)
   
        ### stictching map tiles into single '.png' image
        width = (self.x_max - self.x_min) * self.map_size
        height = (self.y_max - self.y_min) * self.map_size
        
        stitched_map_image = Image.new('RGB', (width, height))
        with tqdm(total=total_tiles, desc='Stitching tiles') as pbar:
            for map_name in tile_names_list:
                idx, x, y, _ = map_name.split('_')
                x, y = int(x), int(y)
                
                x_offset = (x - self.x_min) * self.map_size
                y_offset = (y - self.y_min) * self.map_size
                
                map_path = os.path.join(self.folder_to_save, map_name)

                stitched_map_image.paste(Image.open(map_path), (x_offset, y_offset))
                pbar.update(1)
                
        output_filename = f"{self.source}_map@{self.zoom}@{self.south_east_lat}@{self.north_west_lat}@{self.north_west_lon}@{self.south_east_lon}@.tiff"
        output_png_filename = output_filename.replace('.tiff', '.png')

        stitched_map_image.save(Path(self.folder_to_save).parent / output_png_filename)
        print(f'Successfully save satellite image in png format.')
        
        if not convert_to_tiff:
            return
        
        # ### convert '.png' image to '.tiff' image
        # pixel_width = (self.south_east_lon - self.north_west_lon) / float(width)
        # pixel_height = (self.north_west_lat - self.south_east_lat) / float(height)
        
        # driver = gdal.GetDriverByName('GTiff')
        # output_path = Path(self.folder_to_save).parent / output_filename
        # dataset = driver.Create(output_path, width, height, 3, gdal.GDT_Byte)
        
        # ### 设置仿射变换参数：(左上角经度, 像素宽度, 0, 左上角纬度, 0, -像素高度)
        # geotransform = (self.north_west_lon, pixel_width, 0, self.north_west_lat, 0, -pixel_height)
        # dataset.SetGeoTransform(geotransform)

        # ### 设置坐标系统，这里使用 WGS84 (EPSG:4326)
        # srs = osr.SpatialReference()
        # srs.ImportFromEPSG(4326)
        # dataset.SetProjection(srs.ExportToWkt())

        # ### 分离RGB通道，并写入各个波段
        # r, g, b = stitched_map_image.split()
        # dataset.GetRasterBand(1).WriteArray(np.array(r))
        # dataset.GetRasterBand(2).WriteArray(np.array(g))
        # dataset.GetRasterBand(3).WriteArray(np.array(b))
        # dataset.FlushCache()
        # dataset = None
        # print(f'Successfully save satellite image in tiff format.')
        


from .config import SATE_LATLON, SATE_SIZE

def download_script(region='01'):
    data_root = "data/visloc"
    map_downloader = MapDownloaderAug(
        north_west_lat=SATE_LATLON[region][0],
        north_west_lon=SATE_LATLON[region][1],
        south_east_lat=SATE_LATLON[region][2],
        south_east_lon=SATE_LATLON[region][3],
        zoom = 18,
        folder_to_save = f'{data_root}/{region}/tiles',
        source='esri',
    )
    
    map_downloader.download_map()
    map_downloader.stitch_multi_scale_map(
        folder_to_save=f'{data_root}/{region}/map',
        stitch_values=[2, 3, 4],
        multi_thread=True,
        stitch_workers=8
    )
    map_downloader.create_tiff_map()
    
if __name__ == "__main__":
    download_script(region='04')