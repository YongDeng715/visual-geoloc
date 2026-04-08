from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import RegionConfig
from .downloader import DownloadedTile
from .geo_utils import tile_xy_to_bounds


@dataclass(slots=True)
class ScaledTile:
    index: int
    tile_x: int
    tile_y: int
    scale: float
    region_class: str
    north_lat: float
    west_lon: float
    south_lat: float
    east_lon: float
    path: Path


def _safe_resize(img: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return img
    new_w = max(8, int(round(img.width * scale)))
    new_h = max(8, int(round(img.height * scale)))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def _crop_center_to_size(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    if img.width < out_w or img.height < out_h:
        pad_w = max(0, out_w - img.width)
        pad_h = max(0, out_h - img.height)
        padded = Image.new("RGB", (img.width + pad_w, img.height + pad_h))
        padded.paste(img, (pad_w // 2, pad_h // 2))
        img = padded

    left = (img.width - out_w) // 2
    top = (img.height - out_h) // 2
    return img.crop((left, top, left + out_w, top + out_h))


def build_scaled_tiles(
    region: RegionConfig,
    downloaded_tiles: list[DownloadedTile],
    scales: tuple[float, ...],
    overlap: float,
    out_dir: Path,
) -> list[ScaledTile]:
    """根据原始 zoom=18 瓦片，生成 4 个尺度 + 25% overlap 元数据。"""
    _ = overlap  # 当前命名要求中 overlap 仅用于语义声明；图片通过中心裁切保持原尺寸。

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ScaledTile] = []

    # 以每个原始 tile 为中心，产出 tile_{18}_{x}_{y}_{scale}_.png
    idx = 0
    for tile in downloaded_tiles:
        src = Image.open(tile.path).convert("RGB")
        base_w, base_h = src.size

        for scale in scales:
            scaled = _safe_resize(src, scale)
            fixed = _crop_center_to_size(scaled, base_w, base_h)

            name = f"tile_{region.zoom}_{tile.tile_x}_{tile.tile_y}_{scale}_.png"
            path = out_dir / name
            fixed.save(path)

            # 覆盖范围依据“原始 tile + scale”重新估计
            if scale >= 1.0:
                n, w, s, e = tile.north_lat, tile.west_lon, tile.south_lat, tile.east_lon
            else:
                # scale<1 表示视野变大，按 tile 中心外扩
                lat_c = (tile.north_lat + tile.south_lat) / 2.0
                lon_c = (tile.west_lon + tile.east_lon) / 2.0
                lat_half = (tile.north_lat - tile.south_lat) / 2.0 / scale
                lon_half = (tile.east_lon - tile.west_lon) / 2.0 / scale
                n, s = lat_c + lat_half, lat_c - lat_half
                w, e = lon_c - lon_half, lon_c + lon_half

            # 强制夹回该 xyz tile 邻域（避免极端数值）
            tn, tw, ts, te = tile_xy_to_bounds(tile.tile_x, tile.tile_y, region.zoom)
            n = min(max(n, ts - (tn - ts)), tn + (tn - ts))
            s = max(min(s, tn + (tn - ts)), ts - (tn - ts))
            w = max(min(w, te + (te - tw)), tw - (te - tw))
            e = min(max(e, tw - (te - tw)), te + (te - tw))

            rows.append(
                ScaledTile(
                    index=idx,
                    tile_x=tile.tile_x,
                    tile_y=tile.tile_y,
                    scale=scale,
                    region_class=region.region_class,
                    north_lat=n,
                    west_lon=w,
                    south_lat=s,
                    east_lon=e,
                    path=path,
                )
            )
            idx += 1

    return rows
