# YOLO-Fastest Darknet → ONNX 转换指南

## 方法1: 使用官方 NCNN 工具链 (推荐)

YOLO-Fastest 官方推荐使用 NCNN 部署,可以通过以下流程转换:

### 步骤

1. **安装 NCNN**
```bash
git clone https://github.com/Tencent/ncnn
cd ncnn
mkdir build && cd build
cmake ..
make -j4
```

2. **Darknet → NCNN**
```bash
# 使用 NCNN 的 darknet2ncnn 工具
./tools/darknet/darknet2ncnn \
    ../../models/yolo-fastest-1.1/yolo-fastest-1.1.cfg \
    ../../models/yolo-fastest-1.1/yolo-fastest-1.1.weights \
    yolo-fastest-1.1.param \
    yolo-fastest-1.1.bin
```

3. **NCNN → ONNX** (需要第三方工具)
```bash
# 使用 ncnn2onnx (非官方)
pip install ncnn2onnx
ncnn2onnx yolo-fastest-1.1.param yolo-fastest-1.1.bin -o yolo-fastest-1.1.onnx
```

## 方法2: 使用 PyTorch 重新实现 (复杂)

### 步骤

1. **解析 .cfg 文件**
   - 读取 Darknet 配置文件
   - 构建对应的 PyTorch 网络层

2. **加载 .weights 文件**
   - 读取 Darknet 权重
   - 映射到 PyTorch 模型参数

3. **导出 ONNX**
   ```python
   torch.onnx.export(model, dummy_input, "output.onnx")
   ```

### 参考实现
- https://github.com/Tianxiaomo/pytorch-YOLOv4 (类似架构)
- https://github.com/CaoWGG/TensorRT-YOLOv4 (Darknet → ONNX)

## 方法3: 直接使用预转换模型 (最简单) ⭐

**检查是否有官方 ONNX 模型**:
```bash
# 访问 YOLO-Fastest releases
https://github.com/dog-qiuqiu/Yolo-Fastest/releases

# 查找 ModelZoo
# 可能包含 .onnx 或 .ncnn.param/.bin 文件
```

**如果没有官方 ONNX**:
1. 在 Issues 中搜索 "onnx"
2. 查看是否有社区贡献的转换版本
3. 或者使用 NCNN 格式部署(需要修改 Rust 代码)

## 方法4: 使用 Darknet 官方工具

YOLO-Fastest 基于 AlexeyAB/darknet,可能支持直接导出:

```bash
# 克隆 darknet
git clone https://github.com/AlexeyAB/darknet
cd darknet

# 检查是否支持 ONNX 导出
./darknet detector test cfg/coco.data \\
    ../models/yolo-fastest-1.1/yolo-fastest-1.1.cfg \\
    ../models/yolo-fastest-1.1/yolo-fastest-1.1.weights \\
    data/dog.jpg -save_onnx
```

**注意**: 不是所有 darknet 版本都支持 `-save_onnx` 标志。

## 验证 ONNX 模型

转换完成后,验证模型:

```python
import onnx

# 加载模型
model = onnx.load("yolo-fastest-1.1.onnx")

# 检查模型
onnx.checker.check_model(model)

# 打印模型信息
print(f"Input: {model.graph.input[0].name}")
print(f"Output: {model.graph.output[0].name}")
print(f"Ops: {len(model.graph.node)}")
```

## 在 Rust 项目中使用

转换完成后,将 ONNX 文件放到 `models/` 目录:

```bash
cargo run --bin yolov8-rtsp --release -- --model fastest
```

项目会自动加载:
- `models/yolo-fastest-1.1.onnx`
- `models/yolo-fastest-xl.onnx`

## 常见问题

### Q1: 转换后模型精度下降?
**原因**: 转换过程中算子不兼容或精度损失

**解决**:
- 使用官方转换工具
- 检查 ONNX opset 版本(推荐 opset 11+)
- 验证输入输出尺寸是否正确

### Q2: 转换后模型无法运行?
**原因**: 算子不支持或动态shape问题

**解决**:
- 固定输入尺寸 (320x320)
- 检查 ONNXRuntime 支持的算子
- 简化模型结构(去除非必要层)

### Q3: 在哪里找到官方 ONNX?
**查找位置**:
1. GitHub Releases: https://github.com/dog-qiuqiu/Yolo-Fastest/releases
2. ModelZoo 目录
3. Issues/Discussions 中的社区分享

## 推荐方案

**最简单**: 
1. 从 GitHub 下载预转换的 ONNX (如果有)
2. 或使用 NCNN 格式部署(需要修改 Rust 代码使用 ncnn-rs)

**最可靠**:
1. 使用官方 Darknet → NCNN 工具链
2. 然后转换 NCNN → ONNX (或直接使用 NCNN)

**学习目的**:
1. 手动实现 PyTorch 版本
2. 逐层对比验证精度
3. 深入理解模型架构

---

**更新日期**: 2025-11-13
**相关文档**: docs/22.YOLO-Fastest模型.md
