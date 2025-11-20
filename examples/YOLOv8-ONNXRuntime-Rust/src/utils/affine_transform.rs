/// 仿射变换工具模块
/// 实现类似于 cv2::warpAffine 的功能
use ndarray::Array2;

/// 仿射变换矩阵 (2x3)
/// | a11 a12 b1 |
/// | a21 a22 b2 |
#[derive(Debug, Clone, Copy)]
pub struct AffineMatrix {
    pub a11: f32,
    pub a12: f32,
    pub b1: f32,
    pub a21: f32,
    pub a22: f32,
    pub b2: f32,
}

impl AffineMatrix {
    /// 创建单位仿射矩阵
    pub fn identity() -> Self {
        Self {
            a11: 1.0,
            a12: 0.0,
            b1: 0.0,
            a21: 0.0,
            a22: 1.0,
            b2: 0.0,
        }
    }

    /// 从2x3数组创建
    pub fn from_array(matrix: [[f32; 3]; 2]) -> Self {
        Self {
            a11: matrix[0][0],
            a12: matrix[0][1],
            b1: matrix[0][2],
            a21: matrix[1][0],
            a22: matrix[1][1],
            b2: matrix[1][2],
        }
    }

    /// 转换为2x3数组
    pub fn to_array(&self) -> [[f32; 3]; 2] {
        [[self.a11, self.a12, self.b1], [self.a21, self.a22, self.b2]]
    }

    /// 应用仿射变换到点 (x, y)
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let new_x = self.a11 * x + self.a12 * y + self.b1;
        let new_y = self.a21 * x + self.a22 * y + self.b2;
        (new_x, new_y)
    }

    /// 计算逆矩阵 (用于反向映射)
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a11 * self.a22 - self.a12 * self.a21;
        if det.abs() < 1e-10 {
            return None; // 矩阵不可逆
        }

        let inv_det = 1.0 / det;
        Some(Self {
            a11: self.a22 * inv_det,
            a12: -self.a12 * inv_det,
            b1: (self.a12 * self.b2 - self.a22 * self.b1) * inv_det,
            a21: -self.a21 * inv_det,
            a22: self.a11 * inv_det,
            b2: (self.a21 * self.b1 - self.a11 * self.b2) * inv_det,
        })
    }

    /// 创建平移矩阵
    pub fn translation(dx: f32, dy: f32) -> Self {
        Self {
            a11: 1.0,
            a12: 0.0,
            b1: dx,
            a21: 0.0,
            a22: 1.0,
            b2: dy,
        }
    }

    /// 创建缩放矩阵
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            a11: sx,
            a12: 0.0,
            b1: 0.0,
            a21: 0.0,
            a22: sy,
            b2: 0.0,
        }
    }

    /// 创建旋转矩阵 (角度制)
    pub fn rotation(angle_degrees: f32) -> Self {
        let angle_rad = angle_degrees.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        Self {
            a11: cos_a,
            a12: -sin_a,
            b1: 0.0,
            a21: sin_a,
            a22: cos_a,
            b2: 0.0,
        }
    }

    /// 创建围绕中心点旋转的矩阵
    pub fn rotation_around_center(center_x: f32, center_y: f32, angle_degrees: f32) -> Self {
        let t1 = Self::translation(-center_x, -center_y);
        let r = Self::rotation(angle_degrees);
        let t2 = Self::translation(center_x, center_y);
        t2.compose(&r.compose(&t1))
    }

    /// 矩阵组合 (self * other)
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a11: self.a11 * other.a11 + self.a12 * other.a21,
            a12: self.a11 * other.a12 + self.a12 * other.a22,
            b1: self.a11 * other.b1 + self.a12 * other.b2 + self.b1,
            a21: self.a21 * other.a11 + self.a22 * other.a21,
            a22: self.a21 * other.a12 + self.a22 * other.a22,
            b2: self.a21 * other.b1 + self.a22 * other.b2 + self.b2,
        }
    }
}

/// 插值方法
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    Nearest,  // 最近邻插值
    Bilinear, // 双线性插值
}

/// 边界处理方法
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    Constant(u8), // 常数填充
    Replicate,    // 边缘复制
    Reflect,      // 反射
    Wrap,         // 环绕
}

/// 仿射变换函数 (单通道灰度图)
///
/// # 参数
/// - `src`: 源图像 (height x width)
/// - `matrix`: 仿射变换矩阵
/// - `dst_size`: 目标图像尺寸 (width, height)
/// - `interpolation`: 插值方法
/// - `border_mode`: 边界处理方法
pub fn warp_affine_gray(
    src: &Array2<u8>,
    matrix: &AffineMatrix,
    dst_size: (usize, usize),
    interpolation: InterpolationMethod,
    border_mode: BorderMode,
) -> Array2<u8> {
    let (dst_width, dst_height) = dst_size;
    let (src_height, src_width) = (src.nrows(), src.ncols());
    let mut dst = Array2::<u8>::zeros((dst_height, dst_width));

    // 使用逆变换进行反向映射
    let inv_matrix = matrix.inverse().expect("矩阵不可逆");

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // 反向映射到源图像坐标
            let (src_x, src_y) = inv_matrix.transform_point(dst_x as f32, dst_y as f32);

            // 根据插值方法获取像素值
            let pixel_value = match interpolation {
                InterpolationMethod::Nearest => {
                    get_pixel_nearest(src, src_x, src_y, src_width, src_height, border_mode)
                }
                InterpolationMethod::Bilinear => {
                    get_pixel_bilinear(src, src_x, src_y, src_width, src_height, border_mode)
                }
            };

            dst[[dst_y, dst_x]] = pixel_value;
        }
    }

    dst
}

/// 仿射变换函数 (RGB图像)
///
/// # 参数
/// - `src`: 源图像 (height x width x 3)
/// - `matrix`: 仿射变换矩阵
/// - `dst_size`: 目标图像尺寸 (width, height)
/// - `interpolation`: 插值方法
/// - `border_mode`: 边界处理方法
pub fn warp_affine_rgb(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    matrix: &AffineMatrix,
    dst_size: (usize, usize),
    interpolation: InterpolationMethod,
    border_mode: BorderMode,
) -> Vec<u8> {
    let (dst_width, dst_height) = dst_size;
    let mut dst = vec![0u8; dst_height * dst_width * 3];

    // 使用逆变换进行反向映射
    let inv_matrix = matrix.inverse().expect("矩阵不可逆");

    for dst_y in 0..dst_height {
        for dst_x in 0..dst_width {
            // 反向映射到源图像坐标
            let (src_x, src_y) = inv_matrix.transform_point(dst_x as f32, dst_y as f32);

            // 对每个通道进行插值
            for c in 0..3 {
                let pixel_value = match interpolation {
                    InterpolationMethod::Nearest => get_pixel_nearest_rgb(
                        src,
                        src_x,
                        src_y,
                        src_width,
                        src_height,
                        c,
                        border_mode,
                    ),
                    InterpolationMethod::Bilinear => get_pixel_bilinear_rgb(
                        src,
                        src_x,
                        src_y,
                        src_width,
                        src_height,
                        c,
                        border_mode,
                    ),
                };

                let dst_idx = (dst_y * dst_width + dst_x) * 3 + c;
                dst[dst_idx] = pixel_value;
            }
        }
    }

    dst
}

/// 最近邻插值 (灰度图)
fn get_pixel_nearest(
    src: &Array2<u8>,
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    border_mode: BorderMode,
) -> u8 {
    let ix = x.round() as i32;
    let iy = y.round() as i32;

    get_border_pixel(src, ix, iy, width, height, border_mode)
}

/// 双线性插值 (灰度图)
fn get_pixel_bilinear(
    src: &Array2<u8>,
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    border_mode: BorderMode,
) -> u8 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = get_border_pixel(src, x0, y0, width, height, border_mode) as f32;
    let p01 = get_border_pixel(src, x0, y1, width, height, border_mode) as f32;
    let p10 = get_border_pixel(src, x1, y0, width, height, border_mode) as f32;
    let p11 = get_border_pixel(src, x1, y1, width, height, border_mode) as f32;

    let v0 = p00 * (1.0 - fx) + p10 * fx;
    let v1 = p01 * (1.0 - fx) + p11 * fx;
    let result = v0 * (1.0 - fy) + v1 * fy;

    result.clamp(0.0, 255.0) as u8
}

/// 边界处理 (灰度图)
fn get_border_pixel(
    src: &Array2<u8>,
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    border_mode: BorderMode,
) -> u8 {
    let (bx, by) = handle_border(x, y, width, height, border_mode);

    if bx >= 0 && bx < width as i32 && by >= 0 && by < height as i32 {
        src[[by as usize, bx as usize]]
    } else {
        match border_mode {
            BorderMode::Constant(val) => val,
            _ => 0,
        }
    }
}

/// 最近邻插值 (RGB)
fn get_pixel_nearest_rgb(
    src: &[u8],
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    channel: usize,
    border_mode: BorderMode,
) -> u8 {
    let ix = x.round() as i32;
    let iy = y.round() as i32;

    get_border_pixel_rgb(src, ix, iy, width, height, channel, border_mode)
}

/// 双线性插值 (RGB)
fn get_pixel_bilinear_rgb(
    src: &[u8],
    x: f32,
    y: f32,
    width: usize,
    height: usize,
    channel: usize,
    border_mode: BorderMode,
) -> u8 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = get_border_pixel_rgb(src, x0, y0, width, height, channel, border_mode) as f32;
    let p01 = get_border_pixel_rgb(src, x0, y1, width, height, channel, border_mode) as f32;
    let p10 = get_border_pixel_rgb(src, x1, y0, width, height, channel, border_mode) as f32;
    let p11 = get_border_pixel_rgb(src, x1, y1, width, height, channel, border_mode) as f32;

    let v0 = p00 * (1.0 - fx) + p10 * fx;
    let v1 = p01 * (1.0 - fx) + p11 * fx;
    let result = v0 * (1.0 - fy) + v1 * fy;

    result.clamp(0.0, 255.0) as u8
}

/// 边界处理 (RGB)
fn get_border_pixel_rgb(
    src: &[u8],
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    channel: usize,
    border_mode: BorderMode,
) -> u8 {
    let (bx, by) = handle_border(x, y, width, height, border_mode);

    if bx >= 0 && bx < width as i32 && by >= 0 && by < height as i32 {
        let idx = (by as usize * width + bx as usize) * 3 + channel;
        src[idx]
    } else {
        match border_mode {
            BorderMode::Constant(val) => val,
            _ => 0,
        }
    }
}

/// 边界坐标处理
fn handle_border(
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    border_mode: BorderMode,
) -> (i32, i32) {
    match border_mode {
        BorderMode::Constant(_) => (x, y),
        BorderMode::Replicate => (x.clamp(0, width as i32 - 1), y.clamp(0, height as i32 - 1)),
        BorderMode::Reflect => {
            let mut bx = x;
            let mut by = y;

            if bx < 0 {
                bx = -bx - 1;
            } else if bx >= width as i32 {
                bx = 2 * width as i32 - bx - 1;
            }

            if by < 0 {
                by = -by - 1;
            } else if by >= height as i32 {
                by = 2 * height as i32 - by - 1;
            }

            (
                bx.clamp(0, width as i32 - 1),
                by.clamp(0, height as i32 - 1),
            )
        }
        BorderMode::Wrap => {
            let mut bx = x % width as i32;
            let mut by = y % height as i32;

            if bx < 0 {
                bx += width as i32;
            }
            if by < 0 {
                by += height as i32;
            }

            (bx, by)
        }
    }
}

/// 从3个对应点对获取仿射变换矩阵
///
/// # 参数
/// - `src_pts`: 源图像中的3个点 [(x1,y1), (x2,y2), (x3,y3)]
/// - `dst_pts`: 目标图像中的3个点 [(x1,y1), (x2,y2), (x3,y3)]
pub fn get_affine_transform(
    src_pts: [(f32, f32); 3],
    dst_pts: [(f32, f32); 3],
) -> Option<AffineMatrix> {
    // 构建线性方程组求解仿射矩阵
    // [x1 y1 1  0  0  0] [a11]   [x1']
    // [0  0  0  x1 y1 1] [a12]   [y1']
    // [x2 y2 1  0  0  0] [b1 ] = [x2']
    // [0  0  0  x2 y2 1] [a21]   [y2']
    // [x3 y3 1  0  0  0] [a22]   [x3']
    // [0  0  0  x3 y3 1] [b2 ]   [y3']

    // 使用高斯消元法求解 6x6 线性方程组
    let mut matrix = [
        [src_pts[0].0, src_pts[0].1, 1.0, 0.0, 0.0, 0.0, dst_pts[0].0],
        [0.0, 0.0, 0.0, src_pts[0].0, src_pts[0].1, 1.0, dst_pts[0].1],
        [src_pts[1].0, src_pts[1].1, 1.0, 0.0, 0.0, 0.0, dst_pts[1].0],
        [0.0, 0.0, 0.0, src_pts[1].0, src_pts[1].1, 1.0, dst_pts[1].1],
        [src_pts[2].0, src_pts[2].1, 1.0, 0.0, 0.0, 0.0, dst_pts[2].0],
        [0.0, 0.0, 0.0, src_pts[2].0, src_pts[2].1, 1.0, dst_pts[2].1],
    ];

    // 高斯消元
    for i in 0..6 {
        // 找到主元
        let mut max_row = i;
        for j in (i + 1)..6 {
            if matrix[j][i].abs() > matrix[max_row][i].abs() {
                max_row = j;
            }
        }

        // 交换行
        if max_row != i {
            matrix.swap(i, max_row);
        }

        // 检查奇异性
        if matrix[i][i].abs() < 1e-10 {
            return None;
        }

        // 消元
        for j in (i + 1)..6 {
            let factor = matrix[j][i] / matrix[i][i];
            for k in i..=6 {
                matrix[j][k] -= factor * matrix[i][k];
            }
        }
    }

    // 回代求解
    let mut x = [0.0f32; 6];
    for i in (0..6).rev() {
        x[i] = matrix[i][6];
        for j in (i + 1)..6 {
            x[i] -= matrix[i][j] * x[j];
        }
        x[i] /= matrix[i][i];
    }

    Some(AffineMatrix {
        a11: x[0],
        a12: x[1],
        b1: x[2],
        a21: x[3],
        a22: x[4],
        b2: x[5],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let matrix = AffineMatrix::identity();
        let (x, y) = matrix.transform_point(10.0, 20.0);
        assert_eq!(x, 10.0);
        assert_eq!(y, 20.0);
    }

    #[test]
    fn test_translation() {
        let matrix = AffineMatrix::translation(5.0, 10.0);
        let (x, y) = matrix.transform_point(10.0, 20.0);
        assert_eq!(x, 15.0);
        assert_eq!(y, 30.0);
    }

    #[test]
    fn test_scale() {
        let matrix = AffineMatrix::scale(2.0, 3.0);
        let (x, y) = matrix.transform_point(10.0, 20.0);
        assert_eq!(x, 20.0);
        assert_eq!(y, 60.0);
    }

    #[test]
    fn test_rotation_90() {
        let matrix = AffineMatrix::rotation(90.0);
        let (x, y) = matrix.transform_point(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-6);
        assert!((y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverse() {
        let matrix = AffineMatrix::translation(5.0, 10.0);
        let inv = matrix.inverse().unwrap();
        let composed = matrix.compose(&inv);

        assert!((composed.a11 - 1.0).abs() < 1e-6);
        assert!((composed.a22 - 1.0).abs() < 1e-6);
        assert!((composed.a12).abs() < 1e-6);
        assert!((composed.a21).abs() < 1e-6);
        assert!((composed.b1).abs() < 1e-6);
        assert!((composed.b2).abs() < 1e-6);
    }
}
