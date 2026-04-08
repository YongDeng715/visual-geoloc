from __future__ import annotations

import csv
import os
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw


def _parse_map_bbox_from_filename(path: Path) -> tuple[float, float, float, float]:
    # map@{class}@{north_lat}@{west_lon}@{south_lat}@{east_lon}@.jpg
    stem = path.stem
    parts = stem.split("@")
    if len(parts) < 6:
        raise ValueError(f"Invalid map filename format: {path.name}")
    north = float(parts[2])
    west = float(parts[3])
    south = float(parts[4])
    east = float(parts[5])
    return north, west, south, east


def _read_drone_csv(csv_path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename") or row.get("image") or row.get("name")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")
            if not filename or lat is None or lon is None:
                continue
            result[filename] = (float(lat), float(lon))
    return result


def _read_results_csv(csv_path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            result[filename] = row
    return result


def _latlon_to_xy(
    lat: float,
    lon: float,
    north: float,
    west: float,
    south: float,
    east: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    x = 0 if east == west else int((lon - west) / (east - west) * width)
    y = 0 if north == south else int((lat - north) / (south - north) * height)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return x, y


def _draw_plane(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 12) -> None:
    # 简单飞机符号（蓝色）
    draw.polygon(
        [
            (x, y - size),
            (x - size // 3, y + size // 2),
            (x, y + size // 4),
            (x + size // 3, y + size // 2),
        ],
        fill=(30, 100, 255),
    )
    draw.line((x - size, y, x + size, y), fill=(30, 100, 255), width=2)


def _draw_top1_box(
    draw: ImageDraw.ImageDraw,
    rec: dict,
    north: float,
    west: float,
    south: float,
    east: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    tn = float(rec["top1_north_lat"])
    tw = float(rec["top1_west_lon"])
    ts = float(rec["top1_south_lat"])
    te = float(rec["top1_east_lon"])

    x1, y1 = _latlon_to_xy(tn, tw, north, west, south, east, width, height)
    x2, y2 = _latlon_to_xy(ts, te, north, west, south, east, width, height)
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=3)
    return left, top, right, bottom


def render(
    region_root: str,
    drone_filename: str,
    auto_crop: bool = True,
) -> tuple[Image.Image, str]:
    root = Path(region_root)
    map_candidates = sorted(root.glob("map@*@*.jpg"))
    if not map_candidates:
        raise gr.Error("未找到 map@...@.jpg，请先在 geoloc 阶段生成大图")

    map_path = map_candidates[0]
    drone_csv = root / "drone.csv"
    result_csv = root / "results" / "retrieval_results.csv"
    if not drone_csv.exists():
        raise gr.Error(f"未找到 drone.csv: {drone_csv}")
    if not result_csv.exists():
        raise gr.Error(f"未找到结果文件: {result_csv}")

    gt = _read_drone_csv(drone_csv)
    rs = _read_results_csv(result_csv)

    if drone_filename not in gt:
        raise gr.Error(f"drone.csv 中不存在该文件: {drone_filename}")
    if drone_filename not in rs:
        raise gr.Error(f"results 中不存在该文件: {drone_filename}")

    north, west, south, east = _parse_map_bbox_from_filename(map_path)

    base = Image.open(map_path).convert("RGB")
    draw = ImageDraw.Draw(base)
    w, h = base.size

    lat, lon = gt[drone_filename]
    px, py = _latlon_to_xy(lat, lon, north, west, south, east, w, h)
    _draw_plane(draw, px, py)

    top1_box = _draw_top1_box(draw, rs[drone_filename], north, west, south, east, w, h)

    if auto_crop:
        l, t, r, b = top1_box
        margin = 180
        l = max(0, min(l, px) - margin)
        t = max(0, min(t, py) - margin)
        r = min(w, max(r, px) + margin)
        b = min(h, max(b, py) + margin)
        base = base.crop((l, t, r, b))

    md = (
        f"### 定位结果\n"
        f"- drone: `{drone_filename}`\n"
        f"- GT(lat, lon): `{lat:.7f}, {lon:.7f}`\n"
        f"- Top1 tile: x={rs[drone_filename]['top1_tile_x']}, y={rs[drone_filename]['top1_tile_y']}, "
        f"scale={rs[drone_filename]['top1_scale']}\n"
        f"- Top1 bbox: N={rs[drone_filename]['top1_north_lat']}, W={rs[drone_filename]['top1_west_lon']}, "
        f"S={rs[drone_filename]['top1_south_lat']}, E={rs[drone_filename]['top1_east_lon']}\n"
        f"- Match count: {rs[drone_filename]['match_count']}"
    )
    return base, md


def launch(region_root: str = "data/demo_region") -> None:
    root = Path(region_root)
    default_choices = []
    drone_csv = root / "drone.csv"
    if drone_csv.exists():
        default_choices = sorted(_read_drone_csv(drone_csv).keys())

    with gr.Blocks(title="GeoLoc Visualization") as demo:
        gr.Markdown("## UAV-Satellite 检索定位可视化")
        with gr.Row():
            in_root = gr.Textbox(label="region 根目录", value=region_root)
            in_name = gr.Dropdown(
                label="无人机图片文件名",
                choices=default_choices,
                value=default_choices[0] if default_choices else None,
                allow_custom_value=True,
            )
            in_crop = gr.Checkbox(label="自适应显示小图区域", value=True)

        out_img = gr.Image(label="卫星大图可视化", type="pil")
        out_md = gr.Markdown()

        run_btn = gr.Button("渲染")
        run_btn.click(render, [in_root, in_name, in_crop], [out_img, out_md])

    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    launch()
