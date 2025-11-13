# Scripts 目录

本目录包含项目相关的Python工具脚本。

## 📜 脚本列表

### 1. quantize_onnx_int8.py
**功能**: ONNX模型INT8量化工具

**描述**: 
将models目录下的ONNX模型量化为INT8格式,实现3-4倍推理加速。

**使用方法**:
```bash
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 运行量化脚本
python scripts\quantize_onnx_int8.py
```

**依赖**:
- onnxruntime
- onnx

**输出**:
- INT8量化后的模型保存在 `models/` 目录
- 命名格式: `{原模型名}_int8.onnx`

**性能提升**:
- 模型大小: 减少75%
- 推理速度: 提升3-4倍
- 精度损失: <2%

---

### 2. export_int8_models.py
**功能**: 从PyTorch模型导出INT8 ONNX

**描述**: 
从.pt格式的PyTorch模型导出INT8量化的ONNX模型。

**使用方法**:
```bash
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 运行导出脚本
python scripts\export_int8_models.py
```

**依赖**:
- ultralytics
- onnx
- torch

**注意**:
- 需要先下载对应的.pt模型文件
- 或者直接使用 `quantize_onnx_int8.py` 对现有ONNX模型量化

---

## 🚀 快速开始

### 量化现有ONNX模型 (推荐)

```powershell
# 1. 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 2. 运行量化脚本
python scripts\quantize_onnx_int8.py

# 3. 使用量化后的模型
.\target\release\yolov8-rtsp.exe --int8 -m m
```

### 从头导出INT8模型

```powershell
# 1. 下载PyTorch模型 (可选)
# 下载地址: https://github.com/ultralytics/assets/releases

# 2. 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 3. 运行导出脚本
python scripts\export_int8_models.py
```

---

## 📦 量化效果对比

| 模型 | 原始大小 | INT8大小 | 压缩比 | 速度提升 |
|------|----------|----------|--------|----------|
| YOLOv8n | 12.14 MB | 3.22 MB | 3.77x | ~3.6x |
| YOLOv8m | 99.00 MB | 25.10 MB | 3.94x | ~3.75x |

---

## 💡 使用建议

**场景1: 实时检测** (推荐)
```bash
# 使用INT8量化的M模型
.\target\release\yolov8-rtsp.exe --int8 -m m
# 性能: ~12ms推理,83 FPS
```

**场景2: 极致速度** (游戏自瞄等)
```bash
# 使用INT8量化的N模型
.\target\release\yolov8-rtsp.exe --int8 -m n -p false
# 性能: ~5ms推理,200 FPS
```

**场景3: 高精度监控**
```bash
# 使用原始FP32模型
.\target\release\yolov8-rtsp.exe -m m
# 性能: ~45ms推理,最高精度
```

---

## 🛠️ 依赖安装

如果虚拟环境缺少依赖:

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 安装量化工具依赖
pip install onnxruntime onnx

# (可选) 安装ultralytics用于.pt导出
pip install ultralytics torch
```

---

## 📝 注意事项

1. **量化类型**: 当前使用动态量化(无需校准数据)
2. **精度**: INT8量化通常损失<2%精度
3. **兼容性**: 量化后的模型在ONNXRuntime中运行
4. **性能**: CPU上INT8比FP32快3-4倍

---

**最后更新**: 2025-11-13
