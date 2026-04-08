from __future__ import annotations

import argparse
from pathlib import Path

from .config import BuildContext, GeolocConfig, ModelConfig, PipelineConfig, RegionConfig
from .dataset_builder import RegionDatasetBuilder
from .pipeline import GeoLocalizationPipeline


def _build_context_from_args(args: argparse.Namespace) -> BuildContext:
    region = RegionConfig(
        region_name=args.region_name,
        north_lat=args.north_lat,
        west_lon=args.west_lon,
        south_lat=args.south_lat,
        east_lon=args.east_lon,
        zoom=args.zoom,
        region_class=args.region_class,
        map_source=args.map_source,
    )
    geoloc = GeolocConfig(
        data_root=Path(args.data_root),
        scales=(0.5, 1.0, 1.5, 2.0),
        overlap=0.25,
    )
    model = ModelConfig(
        vpr_model=args.vpr_model,
        device=args.device,
        resize=args.resize,
        anyloc_cluster_centers=Path(args.anyloc_centers) if args.anyloc_centers else None,
        salad_ckpt=Path(args.salad_ckpt) if args.salad_ckpt else None,
        local_min_matches=args.local_min_matches,
    )
    pipeline = PipelineConfig(
        vpr_topk=args.vpr_topk,
        rerank_topk=args.rerank_topk,
        save_match_vis=not args.no_match_vis,
        save_descriptor_cache=not args.no_desc_cache,
    )
    return BuildContext(region=region, geoloc=geoloc, model=model, pipeline=pipeline)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Geoloc backend CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--region-name", required=True)
    common.add_argument("--north-lat", type=float, required=True)
    common.add_argument("--west-lon", type=float, required=True)
    common.add_argument("--south-lat", type=float, required=True)
    common.add_argument("--east-lon", type=float, required=True)
    common.add_argument("--zoom", type=int, default=18)
    common.add_argument("--region-class", default="default")
    common.add_argument("--map-source", choices=["esri", "google", "tianditu", "gaode"], default="esri")
    common.add_argument("--data-root", default="data")

    pb = sub.add_parser("build-dataset", parents=[common], help="下载瓦片并构建 map.csv")
    pb.set_defaults(func=cmd_build_dataset)

    pl = sub.add_parser("localize", parents=[common], help="运行检索与定位")
    pl.add_argument("--vpr-model", choices=["anyloc", "salad"], default="anyloc")
    pl.add_argument("--device", default=None)
    pl.add_argument("--resize", type=int, default=800)
    pl.add_argument("--anyloc-centers", default="weights/pkgs/anyloc_cluster_centers.pt")
    pl.add_argument("--salad-ckpt", default="weights/pkgs/salad_cliquemining.ckpt")
    pl.add_argument("--vpr-topk", type=int, default=20)
    pl.add_argument("--rerank-topk", type=int, default=5)
    pl.add_argument("--local-min-matches", type=int, default=20)
    pl.add_argument("--no-match-vis", action="store_true")
    pl.add_argument("--no-desc-cache", action="store_true")
    pl.set_defaults(func=cmd_localize)

    pall = sub.add_parser("run-all", parents=[common], help="build-dataset + localize")
    pall.add_argument("--vpr-model", choices=["anyloc", "salad"], default="anyloc")
    pall.add_argument("--device", default=None)
    pall.add_argument("--resize", type=int, default=800)
    pall.add_argument("--anyloc-centers", default="weights/pkgs/anyloc_cluster_centers.pt")
    pall.add_argument("--salad-ckpt", default="weights/pkgs/salad_cliquemining.ckpt")
    pall.add_argument("--vpr-topk", type=int, default=20)
    pall.add_argument("--rerank-topk", type=int, default=5)
    pall.add_argument("--local-min-matches", type=int, default=20)
    pall.add_argument("--no-match-vis", action="store_true")
    pall.add_argument("--no-desc-cache", action="store_true")
    pall.set_defaults(func=cmd_run_all)

    return p


def cmd_build_dataset(args: argparse.Namespace) -> int:
    ctx = _build_context_from_args(args)
    builder = RegionDatasetBuilder(ctx)
    map_csv = builder.build()
    print(f"[build-dataset] done: {map_csv}")
    return 0


def cmd_localize(args: argparse.Namespace) -> int:
    ctx = _build_context_from_args(args)
    pipeline = GeoLocalizationPipeline(ctx)
    results = pipeline.run()
    print(f"[localize] done: {len(results)} frames")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    ctx = _build_context_from_args(args)
    builder = RegionDatasetBuilder(ctx)
    map_csv = builder.build()
    print(f"[run-all] dataset built: {map_csv}")

    pipeline = GeoLocalizationPipeline(ctx)
    results = pipeline.run()
    print(f"[run-all] localized: {len(results)} frames")
    return 0


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
