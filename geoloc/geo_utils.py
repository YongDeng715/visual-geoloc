from __future__ import annotations

import math


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """WGS84 经纬度 -> WebMercator XYZ 瓦片坐标。"""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_xy_to_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """瓦片左上/右下边界: north_lat, west_lon, south_lat, east_lon。"""
    n = 2**zoom

    west_lon = x / n * 360.0 - 180.0
    east_lon = (x + 1) / n * 360.0 - 180.0

    def _tile_y_to_lat(tile_y: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))

    north_lat = _tile_y_to_lat(y)
    south_lat = _tile_y_to_lat(y + 1)
    return north_lat, west_lon, south_lat, east_lon


def bounds_intersects(
    a_north: float,
    a_west: float,
    a_south: float,
    a_east: float,
    b_north: float,
    b_west: float,
    b_south: float,
    b_east: float,
) -> bool:
    """判断两个经纬度 bbox 是否相交。"""
    lat_overlap = not (a_south > b_north or b_south > a_north)
    lon_overlap = not (a_east < b_west or b_east < a_west)
    return lat_overlap and lon_overlap


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
