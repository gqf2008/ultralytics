# 仿射变换模块 (Affine Transform)

实现了与 OpenCV `cv2::warpAffine` 相当的功能,用于图像的仿射变换。

## 功能特性

- ✅ 完整的仿射变换矩阵操作
- ✅ 支持灰度图和RGB图像
- ✅ 多种插值方法 (最近邻、双线性)
- ✅ 多种边界处理模式 (常数、复制、反射、环绕)
- ✅ 常用变换快捷方法 (平移、缩放、旋转)
- ✅ 矩阵组合和求逆
- ✅ 从对应点计算变换矩阵

## 使用示例

### 1. 基本仿射变换

```rust
use yolov8_rs::utils::affine_transform::*;
use ndarray::Array2;

// 创建源图像
let src = Array2::<u8>::zeros((100, 100));

// 创建平移矩阵 (向右10像素,向下20像素)
let matrix = AffineMatrix::translation(10.0, 20.0);

// 执行仿射变换
let dst = warp_affine_gray(
    &src,
    &matrix,
    (100, 100),                          // 输出尺寸
    InterpolationMethod::Bilinear,       // 双线性插值
    BorderMode::Constant(0),             // 边界填充0
);
```

### 2. 旋转变换

```rust
// 绕中心点旋转45度
let matrix = AffineMatrix::rotation_around_center(
    50.0, 50.0,  // 中心点坐标
    45.0         // 旋转角度(度)
);

let dst = warp_affine_gray(
    &src,
    &matrix,
    (100, 100),
    InterpolationMethod::Bilinear,
    BorderMode::Constant(0),
);
```

### 3. 缩放变换

```rust
// 放大2倍
let matrix = AffineMatrix::scale(2.0, 2.0);

let dst = warp_affine_gray(
    &src,
    &matrix,
    (200, 200),  // 输出尺寸放大
    InterpolationMethod::Bilinear,
    BorderMode::Constant(0),
);
```

### 4. 组合变换

```rust
// 先缩放,再旋转,最后平移
let scale = AffineMatrix::scale(0.5, 0.5);
let rotate = AffineMatrix::rotation(30.0);
let translate = AffineMatrix::translation(20.0, 10.0);

// 矩阵组合 (注意顺序: 从右到左应用)
let combined = translate.compose(&rotate.compose(&scale));

let dst = warp_affine_gray(
    &src,
    &combined,
    (100, 100),
    InterpolationMethod::Bilinear,
    BorderMode::Constant(0),
);
```

### 5. RGB图像变换

```rust
// RGB图像数据 (height * width * 3)
let src_rgb: Vec<u8> = vec![0u8; 100 * 100 * 3];
let width = 100;
let height = 100;

let matrix = AffineMatrix::rotation_around_center(50.0, 50.0, 45.0);

let dst_rgb = warp_affine_rgb(
    &src_rgb,
    width,
    height,
    &matrix,
    (width, height),
    InterpolationMethod::Bilinear,
    BorderMode::Constant(0),
);
```

### 6. 从对应点计算变换矩阵

```rust
// 定义源图像中的3个点
let src_pts = [
    (0.0, 0.0),      // 点1
    (100.0, 0.0),    // 点2
    (0.0, 100.0),    // 点3
];

// 定义目标图像中对应的3个点
let dst_pts = [
    (20.0, 10.0),
    (120.0, 5.0),
    (10.0, 110.0),
];

// 计算仿射变换矩阵
let matrix = get_affine_transform(src_pts, dst_pts).unwrap();

// 使用计算得到的矩阵进行变换
let dst = warp_affine_gray(&src, &matrix, (150, 150), 
    InterpolationMethod::Bilinear, BorderMode::Constant(0));
```

## API 文档

### 仿射矩阵 (AffineMatrix)

```rust
// 创建单位矩阵
let m = AffineMatrix::identity();

// 从数组创建
let m = AffineMatrix::from_array([[a11, a12, b1], [a21, a22, b2]]);

// 平移
let m = AffineMatrix::translation(dx, dy);

// 缩放
let m = AffineMatrix::scale(sx, sy);

// 旋转 (角度制)
let m = AffineMatrix::rotation(angle_degrees);

// 绕中心旋转
let m = AffineMatrix::rotation_around_center(cx, cy, angle_degrees);

// 矩阵组合
let m3 = m1.compose(&m2);  // m3 = m1 * m2

// 矩阵求逆
let inv = m.inverse().unwrap();

// 点变换
let (new_x, new_y) = m.transform_point(x, y);
```

### 插值方法

```rust
pub enum InterpolationMethod {
    Nearest,   // 最近邻插值 (快速,质量较低)
    Bilinear,  // 双线性插值 (较慢,质量较高)
}
```

### 边界处理模式

```rust
pub enum BorderMode {
    Constant(u8),  // 使用常数填充边界
    Replicate,     // 复制边缘像素
    Reflect,       // 反射模式
    Wrap,          // 环绕模式
}
```

## 运行示例程序

```bash
# 编译并运行示例
cargo run --example affine_transform_demo --release

# 运行测试
cargo test --lib affine_transform
```

## 性能说明

- **最近邻插值**: 速度快,适合实时应用
- **双线性插值**: 质量好,适合离线处理
- **边界处理**: Constant模式最快,Reflect/Wrap较慢

## 与OpenCV对比

| 功能 | OpenCV (Python) | 本实现 (Rust) |
|------|----------------|---------------|
| 仿射变换 | `cv2.warpAffine()` | `warp_affine_gray()` / `warp_affine_rgb()` |
| 获取矩阵 | `cv2.getAffineTransform()` | `get_affine_transform()` |
| 插值方法 | `cv2.INTER_NEAREST`, `cv2.INTER_LINEAR` | `InterpolationMethod::Nearest`, `Bilinear` |
| 边界模式 | `cv2.BORDER_CONSTANT`, `cv2.BORDER_REPLICATE` | `BorderMode::Constant`, `Replicate` |

## 注意事项

1. 矩阵组合顺序遵循数学惯例 (从右到左应用)
2. 坐标系统: 左上角为原点 (0,0)
3. 角度单位: 度 (degrees)
4. RGB图像格式: height × width × 3 (连续存储)
5. 灰度图像使用 `ndarray::Array2<u8>`

## 依赖项

```toml
ndarray = "0.16"
ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
```
