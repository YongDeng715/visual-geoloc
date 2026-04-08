from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MapRecord:
    index: int
    tile_x: int
    tile_y: int
    scale: float
    cls: str
    north_lat: float
    west_lon: float
    south_lat: float
    east_lon: float
    image_path: Path


@dataclass(slots=True)
class DroneRecord:
    filename: str
    latitude: float
    longitude: float
    image_path: Path


def read_map_csv(map_csv: Path, map_dir: Path) -> list[MapRecord]:
    rows: list[MapRecord] = []
    with map_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            scale = float(r["scale"])
            name = f"tile_18_{r['tile_x']}_{r['tile_y']}_{scale}_.png"
            p = map_dir / name
            rows.append(
                MapRecord(
                    index=int(r["index"]),
                    tile_x=int(r["tile_x"]),
                    tile_y=int(r["tile_y"]),
                    scale=scale,
                    cls=r["class"],
                    north_lat=float(r["north_lat"]),
                    west_lon=float(r["west_lon"]),
                    south_lat=float(r["south_lat"]),
                    east_lon=float(r["east_lon"]),
                    image_path=p,
                )
            )
    return rows


def read_drone_csv(drone_csv: Path, drone_dir: Path) -> list[DroneRecord]:
    rows: list[DroneRecord] = []
    with drone_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 兼容字段名: [filename, latitude, longitude] 或 [filename, lat, lon]
        for r in reader:
            filename = r.get("filename") or r.get("image") or r.get("name")
            if filename is None:
                raise ValueError("drone.csv missing filename/image/name column")

            lat_raw = r.get("latitude") or r.get("lat")
            lon_raw = r.get("longitude") or r.get("lon")
            if lat_raw is None or lon_raw is None:
                raise ValueError("drone.csv missing latitude/longitude (or lat/lon) columns")

            rows.append(
                DroneRecord(
                    filename=filename,
                    latitude=float(lat_raw),
                    longitude=float(lon_raw),
                    image_path=drone_dir / filename,
                )
            )
    return rows


def save_retrieval_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
