# Frontend 可视化说明

本前端基于 Gradio，满足以下需求：

1. 展示 `map@{class}@{north_lat}@{west_lon}@{south_lat}@{east_lon}@.jpg` 作为卫星背景，并从文件名解析经纬度范围
2. 从 `drone.csv` 中读取无人机真实经纬度
3. 在卫星图上用蓝色小飞机符号标注 UAV 真实位置
4. 从 `results/retrieval_results.csv` 读取每张 UAV 图的 top1 卫星图边界，并在大图上以红框高亮
5. 支持“自适应显示小图区域”（自动裁切到 UAV 与 top1 红框附近）

## 启动

```bash
pip install -r frontend/requirements.txt
python -m frontend.app
```

默认读取 `data/demo_region`，也可在 UI 中手动输入 region 根目录。
