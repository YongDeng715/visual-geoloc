from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import BuildContext
from .io_utils import DroneRecord, MapRecord, read_drone_csv, read_map_csv, save_retrieval_csv
from .retrieval import (
    LocalMatcher,
    build_vpr_model,
    estimate_query_latlon_from_homography,
    run_retrieval_topk,
)


@dataclass(slots=True)
class LocalizeResult:
    filename: str
    gt_latitude: float
    gt_longitude: float
    pred_latitude: float | None
    pred_longitude: float | None
    top1_map_index: int
    top1_tile_x: int
    top1_tile_y: int
    top1_scale: float
    top1_north_lat: float
    top1_west_lon: float
    top1_south_lat: float
    top1_east_lon: float
    vpr_distance: float
    match_count: int
    match_vis_path: str | None


class GeoLocalizationPipeline:
    """端到端：读取 drone/map CSV，执行 VPR + 局部匹配 + 结果落盘。"""

    def __init__(self, ctx: BuildContext):
        self.ctx = ctx
        self.paths = ctx.paths()

    def _compute_map_descriptors(
        self,
        map_rows: list[MapRecord],
        map_images: list[np.ndarray],
        vpr_model,
    ) -> np.ndarray:
        desc_cache = self.paths.descriptor_dir / f"map_desc_{self.ctx.model.vpr_model}.npy"
        if self.ctx.pipeline.save_descriptor_cache and desc_cache.exists():
            return np.load(desc_cache)

        descs = []
        for img in map_images:
            descs.append(vpr_model.get_image_descriptor(img))
        mat = np.asarray(descs, dtype=np.float32)

        if self.ctx.pipeline.save_descriptor_cache:
            self.paths.descriptor_dir.mkdir(parents=True, exist_ok=True)
            np.save(desc_cache, mat)
        return mat

    def _draw_match_vis(
        self,
        qimg: np.ndarray,
        mimg: np.ndarray,
        qpts: np.ndarray,
        mpts: np.ndarray,
        out_path: Path,
    ) -> None:
        if len(qpts) == 0:
            return
        matches = [cv2.DMatch(i, i, 1.0) for i in range(len(qpts))]
        qk = [cv2.KeyPoint(float(x), float(y), 1) for x, y in qpts]
        mk = [cv2.KeyPoint(float(x), float(y), 1) for x, y in mpts]
        vis = cv2.drawMatches(qimg, qk, mimg, mk, matches, None)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)

    def run(self) -> list[LocalizeResult]:
        map_rows = read_map_csv(self.paths.map_csv, self.paths.map_root)
        drone_rows = read_drone_csv(self.paths.drone_csv, self.paths.drone_dir)

        map_images: list[np.ndarray] = []
        for r in map_rows:
            img = cv2.imread(str(r.image_path))
            if img is None:
                raise FileNotFoundError(f"Map image not found: {r.image_path}")
            map_images.append(img)

        vpr_model = build_vpr_model(self.ctx.model)
        map_descs = self._compute_map_descriptors(map_rows, map_images, vpr_model)

        matcher = LocalMatcher(resize=self.ctx.model.resize, device=self.ctx.model.device or "cpu")

        all_results: list[LocalizeResult] = []
        csv_rows: list[dict] = []

        for d in drone_rows:
            qimg = cv2.imread(str(d.image_path))
            if qimg is None:
                raise FileNotFoundError(f"Drone image not found: {d.image_path}")

            retrievals = run_retrieval_topk(
                query_image=qimg,
                map_images=map_images,
                map_descs=map_descs,
                vpr_model=vpr_model,
                topk=self.ctx.pipeline.vpr_topk,
                matcher=matcher,
                rerank_topk=self.ctx.pipeline.rerank_topk,
            )
            top1 = retrievals[0]
            mrec = map_rows[top1["map_index"]]

            pred_latlon = None
            if top1["match_count"] >= self.ctx.model.local_min_matches:
                pred_latlon = estimate_query_latlon_from_homography(
                    query_image=qimg,
                    matched_q=top1["matched_q"],
                    matched_m=top1["matched_m"],
                    bbox_nwse=(mrec.north_lat, mrec.west_lon, mrec.south_lat, mrec.east_lon),
                )

            vis_path: str | None = None
            if self.ctx.pipeline.save_match_vis and top1["match_count"] > 0:
                p = self.paths.result_dir / "matches" / f"{Path(d.filename).stem}_top1.jpg"
                self._draw_match_vis(
                    qimg,
                    map_images[top1["map_index"]],
                    top1["matched_q"],
                    top1["matched_m"],
                    p,
                )
                vis_path = str(p)

            pred_lat = pred_latlon[0] if pred_latlon is not None else None
            pred_lon = pred_latlon[1] if pred_latlon is not None else None

            result = LocalizeResult(
                filename=d.filename,
                gt_latitude=d.latitude,
                gt_longitude=d.longitude,
                pred_latitude=pred_lat,
                pred_longitude=pred_lon,
                top1_map_index=mrec.index,
                top1_tile_x=mrec.tile_x,
                top1_tile_y=mrec.tile_y,
                top1_scale=mrec.scale,
                top1_north_lat=mrec.north_lat,
                top1_west_lon=mrec.west_lon,
                top1_south_lat=mrec.south_lat,
                top1_east_lon=mrec.east_lon,
                vpr_distance=float(top1["vpr_distance"]),
                match_count=int(top1["match_count"]),
                match_vis_path=vis_path,
            )
            all_results.append(result)

            csv_rows.append(
                {
                    "filename": result.filename,
                    "gt_latitude": result.gt_latitude,
                    "gt_longitude": result.gt_longitude,
                    "pred_latitude": result.pred_latitude,
                    "pred_longitude": result.pred_longitude,
                    "top1_map_index": result.top1_map_index,
                    "top1_tile_x": result.top1_tile_x,
                    "top1_tile_y": result.top1_tile_y,
                    "top1_scale": result.top1_scale,
                    "top1_north_lat": result.top1_north_lat,
                    "top1_west_lon": result.top1_west_lon,
                    "top1_south_lat": result.top1_south_lat,
                    "top1_east_lon": result.top1_east_lon,
                    "vpr_distance": result.vpr_distance,
                    "match_count": result.match_count,
                    "match_vis_path": result.match_vis_path,
                }
            )

        out_csv = self.paths.result_dir / self.ctx.pipeline.result_filename
        save_retrieval_csv(out_csv, csv_rows)
        return all_results
