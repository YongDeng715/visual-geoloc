"""Geoloc backend package.

此包提供：
1) 卫星瓦片下载与多尺度切片构建
2) VPR（AnyLoc / SALAD）检索
3) SuperPoint + LightGlue 局部匹配与重排序
4) 同构估计与经纬度解算
"""

from .config import GeolocConfig, ModelConfig, PipelineConfig, RegionConfig
from .dataset_builder import RegionDatasetBuilder
from .pipeline import GeoLocalizationPipeline, LocalizeResult
from .retrieval import build_vpr_model, run_retrieval_topk

__all__ = [
    "GeolocConfig",
    "ModelConfig",
    "PipelineConfig",
    "RegionConfig",
    "RegionDatasetBuilder",
    "GeoLocalizationPipeline",
    "LocalizeResult",
    "build_vpr_model",
    "run_retrieval_topk",
]
