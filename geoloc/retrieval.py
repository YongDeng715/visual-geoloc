from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from aero_vloc.feat_detector.superpoint import SuperPoint
from aero_vloc.feat_matcher.lightglue.lightglue_matcher import LightGlueMatcher
from aero_vloc.models.anyloc import AnyLocVladDinov2
from aero_vloc.models.salad import SALAD
from aero_vloc.utils.aero_utils import transform_image_for_sp

from .config import ModelConfig

VPRModelName = Literal["anyloc", "salad"]


class _VPRModelAdapter:
    def __init__(self, model):
        self.model = model

    def get_image_descriptor(self, image_bgr: np.ndarray) -> np.ndarray:
        desc = self.model.get_image_descriptor(image_bgr)
        if isinstance(desc, np.ndarray):
            return desc.astype(np.float32)
        return np.asarray(desc, dtype=np.float32)


def _safe_device(cfg: ModelConfig) -> str:
    if cfg.device is not None:
        return cfg.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def build_vpr_model(cfg: ModelConfig) -> _VPRModelAdapter:
    device = _safe_device(cfg)

    if cfg.vpr_model == "anyloc":
        centers = cfg.anyloc_cluster_centers
        if centers is None:
            raise ValueError("ModelConfig.anyloc_cluster_centers is required for AnyLoc")
        model = AnyLocVladDinov2(c_centers_path=Path(centers), resize=cfg.resize, device=device)
        return _VPRModelAdapter(model)

    if cfg.vpr_model == "salad":
        ckpt = cfg.salad_ckpt
        if ckpt is None:
            raise ValueError("ModelConfig.salad_ckpt is required for SALAD")
        model = SALAD(path_to_weights=str(ckpt), resize=cfg.resize, device=device)
        return _VPRModelAdapter(model)

    raise ValueError(f"Unsupported VPR model: {cfg.vpr_model}")


def l2_search(
    query_desc: np.ndarray,
    db_descs: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = query_desc[None, :]
    diff = db_descs - q
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    idx = np.argsort(dist)[:topk]
    return idx, dist[idx]


class LocalMatcher:
    """SuperPoint + LightGlue 局部匹配，用于 Top-K 重排序。"""

    def __init__(self, resize: int = 800, device: str = "cpu"):
        import torch

        self.resize = resize
        self.device = device
        self._torch = torch
        self.sp = SuperPoint().eval().to(device)
        # 使用本地权重模式，避免在线下载失败导致初始化中断。
        self.lg = LightGlueMatcher(features=None, weights="superpoint_lightglue").eval().to(device)

    def _extract(self, image_bgr: np.ndarray) -> dict:
        img = transform_image_for_sp(image_bgr, self.resize).to(self.device)
        shape = img.shape[-2:][::-1]
        with self._torch.no_grad():
            feats = self.sp({"image": img})
        feats["descriptors"] = feats["descriptors"].transpose(-1, -2).contiguous()
        feats = {k: v for k, v in feats.items()}
        feats["image_size"] = self._torch.tensor(shape, device=self.device).float()[None]
        return feats

    def match_count(self, query_bgr: np.ndarray, map_bgr: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        qf = self._extract(query_bgr)
        mf = self._extract(map_bgr)
        with self._torch.no_grad():
            out = self.lg({"image0": qf, "image1": mf})
        matches = out["matches"][0]

        if matches.numel() == 0:
            return 0, np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        qk = qf["keypoints"][0][matches[:, 0]].detach().cpu().numpy().astype(np.float32)
        mk = mf["keypoints"][0][matches[:, 1]].detach().cpu().numpy().astype(np.float32)
        return int(len(qk)), qk, mk


def run_retrieval_topk(
    query_image: np.ndarray,
    map_images: list[np.ndarray],
    map_descs: np.ndarray,
    vpr_model: _VPRModelAdapter,
    topk: int,
    matcher: LocalMatcher | None = None,
    rerank_topk: int = 0,
) -> list[dict]:
    qdesc = vpr_model.get_image_descriptor(query_image)
    idx, dist = l2_search(qdesc, map_descs, topk=topk)

    results: list[dict] = []
    for rank, (i, d) in enumerate(zip(idx.tolist(), dist.tolist()), start=1):
        results.append(
            {
                "rank_vpr": rank,
                "map_index": i,
                "vpr_distance": float(d),
                "match_count": -1,
                "rank_final": rank,
                "matched_q": np.empty((0, 2), dtype=np.float32),
                "matched_m": np.empty((0, 2), dtype=np.float32),
            }
        )

    if matcher is not None and rerank_topk > 0:
        n = min(rerank_topk, len(results))
        for j in range(n):
            mi = results[j]["map_index"]
            count, qk, mk = matcher.match_count(query_image, map_images[mi])
            results[j]["match_count"] = count
            results[j]["matched_q"] = qk
            results[j]["matched_m"] = mk

        head = sorted(results[:n], key=lambda x: (-x["match_count"], x["vpr_distance"]))
        results = head + results[n:]
        for k, r in enumerate(results, start=1):
            r["rank_final"] = k

    return results


def estimate_query_latlon_from_homography(
    query_image: np.ndarray,
    matched_q: np.ndarray,
    matched_m: np.ndarray,
    bbox_nwse: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """基于匹配点做单应，估计 query 中心对应的 map 经纬度。"""
    if len(matched_q) < 4 or len(matched_m) < 4:
        return None

    H, _ = cv2.findHomography(matched_q, matched_m, cv2.RANSAC, 5.0)
    if H is None:
        return None

    qh, qw = query_image.shape[:2]
    center = np.array([[[qw / 2.0, qh / 2.0]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(center, H)[0, 0]
    except cv2.error:
        return None

    x, y = float(mapped[0]), float(mapped[1])

    north, west, south, east = bbox_nwse
    # 单应后坐标属于 map 图像平面，使用该图平面宽高归一化。
    min_x = float(np.min(matched_m[:, 0]))
    max_x = float(np.max(matched_m[:, 0]))
    min_y = float(np.min(matched_m[:, 1]))
    max_y = float(np.max(matched_m[:, 1]))

    mx = float(np.clip(x, min_x, max_x))
    my = float(np.clip(y, min_y, max_y))
    rx = (mx - min_x) / max(1.0, (max_x - min_x))
    ry = (my - min_y) / max(1.0, (max_y - min_y))

    # 为减小极端外推，使用匹配点覆盖范围做局部比例近似。
    lon = west + rx * (east - west)
    lat = north + ry * (south - north)
    return lat, lon
