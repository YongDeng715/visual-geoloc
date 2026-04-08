from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from .config import BuildContext
from .downloader import TileDownloader
from .tiling import ScaledTile, build_scaled_tiles


MAP_CSV_HEADERS = [
    "index",
    "tile_x",
    "tile_y",
    "scale",
    "class",
    "north_lat",
    "west_lon",
    "south_lat",
    "east_lon",
]


class RegionDatasetBuilder:
    """构建 data/{region_name} 结构与 map.csv。"""

    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.paths = ctx.paths()

    def ensure_layout(self) -> None:
        self.paths.region_root.mkdir(parents=True, exist_ok=True)
        self.paths.drone_dir.mkdir(parents=True, exist_ok=True)
        self.paths.tile_dir.mkdir(parents=True, exist_ok=True)
        self.paths.map_root.mkdir(parents=True, exist_ok=True)
        for scale in self.ctx.geoloc.scales:
            (self.paths.map_root / f"scale_{scale}").mkdir(parents=True, exist_ok=True)
        self.paths.descriptor_dir.mkdir(parents=True, exist_ok=True)
        self.paths.result_dir.mkdir(parents=True, exist_ok=True)

    def _save_map_csv(self, rows: list[ScaledTile]) -> None:
        with self.paths.map_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(MAP_CSV_HEADERS)
            for r in rows:
                writer.writerow(
                    [
                        r.index,
                        r.tile_x,
                        r.tile_y,
                        r.scale,
                        r.region_class,
                        r.north_lat,
                        r.west_lon,
                        r.south_lat,
                        r.east_lon,
                    ]
                )

    def _copy_scaled_to_subdirs(self, rows: list[ScaledTile]) -> None:
        for row in rows:
            target = self.paths.map_root / f"scale_{row.scale}" / row.path.name
            if not target.exists():
                Image.open(row.path).save(target)

    def _stitch_big_map(self, rows: list[ScaledTile]) -> None:
        # 取 scale=1.0 且 x/y 完整网格拼接。
        rows_1 = [r for r in rows if float(r.scale) == 1.0]
        if not rows_1:
            return
        xs = sorted({r.tile_x for r in rows_1})
        ys = sorted({r.tile_y for r in rows_1})
        row_map = {(r.tile_x, r.tile_y): r for r in rows_1}

        sample = Image.open(rows_1[0].path).convert("RGB")
        w, h = sample.size
        canvas = Image.new("RGB", (len(xs) * w, len(ys) * h))
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                r = row_map.get((x, y))
                if r is None:
                    continue
                tile = Image.open(r.path).convert("RGB")
                canvas.paste(tile, (xi * w, yi * h))

        canvas.save(self.paths.map_big_image)

    def build(self) -> Path:
        self.ensure_layout()

        downloader = TileDownloader(self.ctx.region, self.ctx.geoloc, self.paths.tile_dir)
        downloaded = downloader.download()

        scaled = build_scaled_tiles(
            region=self.ctx.region,
            downloaded_tiles=downloaded,
            scales=self.ctx.geoloc.scales,
            overlap=self.ctx.geoloc.overlap,
            out_dir=self.paths.map_root,
        )
        self._save_map_csv(scaled)
        self._copy_scaled_to_subdirs(scaled)
        self._stitch_big_map(scaled)
        return self.paths.map_csv
