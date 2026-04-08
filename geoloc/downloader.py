from __future__ import annotations

import io
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import GeolocConfig, RegionConfig
from .geo_utils import bounds_intersects, latlon_to_tile_xy, tile_xy_to_bounds


@dataclass(slots=True)
class DownloadedTile:
    index: int
    tile_x: int
    tile_y: int
    north_lat: float
    west_lon: float
    south_lat: float
    east_lon: float
    path: Path


class TileDownloader:
    """下载给定区域在指定 zoom 的原始卫星瓦片。"""

    def __init__(self, region: RegionConfig, cfg: GeolocConfig, tile_dir: Path):
        self.region = region
        self.cfg = cfg
        self.tile_dir = tile_dir
        self.tile_dir.mkdir(parents=True, exist_ok=True)

    def _url(self, x: int, y: int) -> str:
        z = self.region.zoom
        source = self.region.map_source
        if source == "esri":
            return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        if source == "google":
            return f"https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}"
        if source == "tianditu":
            # 可由调用侧自行在 URL 中注入 tk，这里使用公开模板。
            return (
                "http://t0.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile"
                f"&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles"
                f"&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
            )
        if source == "gaode":
            return f"https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
        raise ValueError(f"Unsupported map source: {source}")

    def _fetch_bytes(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self.cfg.user_agent})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
            return resp.read()

    def _download_one(self, x: int, y: int, path: Path) -> None:
        if path.exists():
            return

        last_err: Exception | None = None
        for _ in range(self.cfg.max_retries):
            try:
                content = self._fetch_bytes(self._url(x, y))
                image = Image.open(io.BytesIO(content)).convert("RGB")
                image.save(path)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.3)
        if last_err is not None:
            raise last_err

    def iter_tile_xy(self) -> Iterable[tuple[int, int]]:
        x_min, y_min = latlon_to_tile_xy(self.region.north_lat, self.region.west_lon, self.region.zoom)
        x_max, y_max = latlon_to_tile_xy(self.region.south_lat, self.region.east_lon, self.region.zoom)
        for x in range(min(x_min, x_max), max(x_min, x_max) + 1):
            for y in range(min(y_min, y_max), max(y_min, y_max) + 1):
                yield x, y

    def download(self) -> list[DownloadedTile]:
        rows: list[DownloadedTile] = []
        index = 0
        for x, y in self.iter_tile_xy():
            n, w, s, e = tile_xy_to_bounds(x, y, self.region.zoom)
            if not bounds_intersects(
                n,
                w,
                s,
                e,
                self.region.north_lat,
                self.region.west_lon,
                self.region.south_lat,
                self.region.east_lon,
            ):
                continue
            tile_name = f"tile_{self.region.zoom}_{x}_{y}_.png"
            tile_path = self.tile_dir / tile_name
            self._download_one(x, y, tile_path)
            rows.append(
                DownloadedTile(
                    index=index,
                    tile_x=x,
                    tile_y=y,
                    north_lat=n,
                    west_lon=w,
                    south_lat=s,
                    east_lon=e,
                    path=tile_path,
                )
            )
            index += 1

        if not rows:
            raise RuntimeError("No map tiles downloaded. Please check bbox / map source / network.")
        return rows
