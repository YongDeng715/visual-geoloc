from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


VPRModelName = Literal["anyloc", "salad"]
MapSourceName = Literal["esri", "google", "tianditu", "gaode"]


@dataclass(slots=True)
class RegionConfig:
    """区域与地图下载配置。"""

    region_name: str
    north_lat: float
    west_lon: float
    south_lat: float
    east_lon: float
    zoom: int = 18
    region_class: str = "default"
    map_source: MapSourceName = "esri"


@dataclass(slots=True)
class GeolocConfig:
    """数据集构建配置。"""

    data_root: Path = Path("data")
    scales: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    overlap: float = 0.25
    tile_size: int = 256
    timeout_sec: float = 10.0
    max_retries: int = 3
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass(slots=True)
class ModelConfig:
    """模型配置（仅 AnyLoc / SALAD 与 SuperPoint+LightGlue）。"""

    vpr_model: VPRModelName = "anyloc"
    device: str | None = None
    resize: int = 800

    # AnyLoc
    anyloc_cluster_centers: Path | None = Path("weights/pkgs/anyloc_cluster_centers.pt")

    # SALAD
    salad_ckpt: Path | None = Path("weights/pkgs/salad_cliquemining.ckpt")

    # Local matcher
    local_min_matches: int = 20
    ransac_reproj_threshold: float = 5.0


@dataclass(slots=True)
class PipelineConfig:
    """定位流程配置。"""

    vpr_topk: int = 20
    rerank_topk: int = 5
    save_match_vis: bool = True
    save_descriptor_cache: bool = True
    descriptor_dirname: str = "descriptors"
    result_dirname: str = "results"
    result_filename: str = "retrieval_results.csv"


@dataclass(slots=True)
class RuntimePaths:
    """由 region_name 与 data_root 推导出的路径。"""

    region_root: Path
    drone_dir: Path
    drone_csv: Path
    tile_dir: Path
    map_root: Path
    map_csv: Path
    map_big_image: Path
    descriptor_dir: Path
    result_dir: Path


@dataclass(slots=True)
class BuildContext:
    """整体验证/运行上下文。"""

    region: RegionConfig
    geoloc: GeolocConfig = field(default_factory=GeolocConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def paths(self) -> RuntimePaths:
        region_root = self.geoloc.data_root / self.region.region_name
        drone_dir = region_root / "drone"
        drone_csv = region_root / "drone.csv"
        tile_dir = region_root / "tile"
        map_root = region_root / "map"
        map_csv = map_root / "map.csv"
        map_big_image = region_root / (
            f"map@{self.region.region_class}@"
            f"{self.region.north_lat}@{self.region.west_lon}@"
            f"{self.region.south_lat}@{self.region.east_lon}@.jpg"
        )
        descriptor_dir = region_root / self.pipeline.descriptor_dirname
        result_dir = region_root / self.pipeline.result_dirname
        return RuntimePaths(
            region_root=region_root,
            drone_dir=drone_dir,
            drone_csv=drone_csv,
            tile_dir=tile_dir,
            map_root=map_root,
            map_csv=map_csv,
            map_big_image=map_big_image,
            descriptor_dir=descriptor_dir,
            result_dir=result_dir,
        )
