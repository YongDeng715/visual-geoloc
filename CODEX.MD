# Codex/Claude 项目代码需求详情

该项目是一个自定义区域多尺度无人机视觉定位系统. 你需要根据 参考文件 [aero-vloc](@reference{aero-vloc}) 和之前已经编写的一部分代码 [aero_vloc](@{aero_vloc}) 来完成这个项目，其主要包括 后端 [geoloc](@{geoloc}) 和前端 [frontend](@{frontend}) 和 前后端测试 [tests](@{tests}) 三个部分.

其中 后端主要包括 以下 1， 2， 3 部分。

## 1.卫星图像下载端口和多尺度卫星瓦图处理逻辑

卫星瓦片按照 原始18级图片下载，原始图片的命名方法是 `tile_{18}_{x}_{y}_.png` , 通过 [aero-vloc](@reference{aero-vloc}) 中的图片处理流程， 将其获得 25% overlap 和 0.5, 1.0, 1.5, 2.0 四个尺度的卫星图片， 并保存为 `tile_{18}_{x}_{y}_{scale}_.png` , 其中 scale 表示图片的缩放比例，例如 0.5 表示图片缩小为原来的一半， 1.0 表示图片保持不变， 1.5 表示图片放大为原来的1.5倍， 2.0 表示图片放大为原来的2倍, 此外通过 `.csv` 文件存储 每个瓦片的经纬度信息，分为如下标题 `index tile_x tile_y scale class north_lat west_lon south_lat east_lon` , 其中 `index` 表示瓦片的编号， `tile_x` 和 `tile_y` 表示瓦片的坐标， `scale` 表示瓦片的缩放比例， `class` 表示瓦片的类别， `north_lat` 和 `west_lon` 表示瓦片的左上角坐标， `south_lat` 和 `east_lon` 表示瓦片的右下角坐标.

然后按照 所给定的图片检索模型和 特征点提取和匹配模型处理图片

```txt
data/{region_name}
│   ├── drone.csv
    ├── drone
    │   ├── .jpg
    │   ├── ...
    ├── tile
    ├── map
    │   ├── scale_0.5/
    │   ├── scale_1.0/
    │   ├── scale_1.5/
    │   ├── scale_2.0/
    │   ├── map.csv
    ├── map@{class}@{north_lat}@{west_lon}@{south_lat}@{east_lon}@.jpg
```

## 2.图像检索(VPR)模型

只使用 [AnyLoc](@reference/aero-vloc/aero_vloc/vpr_systems/{anyloc}) 和 [salad](@reference/aero-vloc/aero_vloc/vpr_systems/{salad})的模型，不需要训练

## 3. 图像特征点提取和匹配模型

只使用 [superpoint](@reference/aero-vloc/aero_vloc/{feature_detectors}) 和 [lightglue](@reference/aero-vloc/aero_vloc/feature_matchers/{lightglue}) 的 特征点训练方法，

部分特征提取方法可以参考 [vismatch](@reference/{vismatch}) 的实现.

特征点检索和匹配用于计算 无人机和卫星图片形变计算两者的无人机的精确距离（使用图像匹配方法）

## 4. 检索定位结果可视化前端

通过 gradio 或 flask 编写一个网页，实现以下功能：

1. 呈现一张巨幅 map 作为卫星图片别背景，通过文件名提取其经纬度信息
2. 通过无人机图片序列以及 `drone.csv` 文件提取无人机的经纬度位置
3. 通过一个小飞机标识展示 无人机的实际经纬度位置
4. 通过检索到的 map 的 边角的经纬度数值，对于每个无人机图片对应的 top1 卫星图片，在卫星大图上通过 红框或红色覆盖重点标识出来
5. 卫星大图自适应显示小图区域（不用显示全部）

## 5. 测试部分

### 5.1 后端测试

1. 下载卫星图片区域构建数据集
2. 通过模型的输出张量保存
3. 无人机序列的检索结果，可视化每个特征匹配结果

### 5.2 前端测试

1. 通过网页前端展示结果

## 6. 参考文档

完善 [README.md](README.md) 文档，所有相关技术细节 填充在 [docs](@{docs}) 文件夹中，包括但不限于：

## 许可证和其他信息