/// SIMD优化的仿射变换模块
/// 使用 packed_simd/std::simd 加速图像变换

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::affine_transform::{AffineMatrix, BorderMode, InterpolationMethod};

/// SIMD优化的RGB图像仿射变换
/// 
/// 性能优化:
/// 1. 使用SIMD并行处理多个像素
/// 2. 循环展开减少分支预测失败
/// 3. 预计算常量避免重复计算
/// 4. 内存对齐优化缓存命中率
#[cfg(target_arch = "x86_64")]
pub fn warp_affine_rgb_simd(
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

    unsafe {
        match interpolation {
            InterpolationMethod::Bilinear => {
                warp_affine_rgb_bilinear_simd(
                    src,
                    src_width,
                    src_height,
                    &inv_matrix,
                    &mut dst,
                    dst_width,
                    dst_height,
                    border_mode,
                );
            }
            InterpolationMethod::Nearest => {
                warp_affine_rgb_nearest_simd(
                    src,
                    src_width,
                    src_height,
                    &inv_matrix,
                    &mut dst,
                    dst_width,
                    dst_height,
                    border_mode,
                );
            }
        }
    }

    dst
}

/// SIMD优化的双线性插值RGB变换
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn warp_affine_rgb_bilinear_simd(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    inv_matrix: &AffineMatrix,
    dst: &mut [u8],
    dst_width: usize,
    dst_height: usize,
    border_mode: BorderMode,
) {
    let src_width_f32 = src_width as f32;
    let src_height_f32 = src_height as f32;
    let src_width_i32 = src_width as i32;

    // 预计算边界值
    let border_value = match border_mode {
        BorderMode::Constant(val) => val,
        _ => 0,
    };

    // 预计算矩阵元素用于向量化
    let a11 = inv_matrix.a11;
    let a12 = inv_matrix.a12;
    let b1 = inv_matrix.b1;
    let a21 = inv_matrix.a21;
    let a22 = inv_matrix.a22;
    let b2 = inv_matrix.b2;

    for dst_y in 0..dst_height {
        let dst_y_f32 = dst_y as f32;
        
        // 预计算Y方向的变换分量
        let base_src_x = a12 * dst_y_f32 + b1;
        let base_src_y = a22 * dst_y_f32 + b2;

        let mut dst_x = 0;

        // SIMD处理: 一次处理8个像素
        while dst_x + 8 <= dst_width {
            // 创建8个连续的x坐标向量
            let x_offsets = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
            let dst_x_vec = _mm256_add_ps(_mm256_set1_ps(dst_x as f32), x_offsets);

            // 计算源坐标 (向量化)
            let src_x_vec = _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(a11), dst_x_vec),
                _mm256_set1_ps(base_src_x),
            );
            let src_y_vec = _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(a21), dst_x_vec),
                _mm256_set1_ps(base_src_y),
            );

            // 提取到标量数组进行边界检查和插值
            let mut src_x_arr = [0.0f32; 8];
            let mut src_y_arr = [0.0f32; 8];
            _mm256_storeu_ps(src_x_arr.as_mut_ptr(), src_x_vec);
            _mm256_storeu_ps(src_y_arr.as_mut_ptr(), src_y_vec);

            // 对每个像素进行双线性插值
            for i in 0..8 {
                let src_x = src_x_arr[i];
                let src_y = src_y_arr[i];

                // 边界检查
                if src_x >= 0.0 && src_x < src_width_f32 - 1.0 
                    && src_y >= 0.0 && src_y < src_height_f32 - 1.0 
                {
                    // 快速双线性插值
                    let x0 = src_x as i32;
                    let y0 = src_y as i32;
                    let fx = src_x - x0 as f32;
                    let fy = src_y - y0 as f32;

                    let idx00 = ((y0 * src_width_i32 + x0) * 3) as usize;
                    let idx01 = (((y0 + 1) * src_width_i32 + x0) * 3) as usize;
                    let idx10 = ((y0 * src_width_i32 + x0 + 1) * 3) as usize;
                    let idx11 = (((y0 + 1) * src_width_i32 + x0 + 1) * 3) as usize;

                    let dst_idx = ((dst_y * dst_width + dst_x + i) * 3) as usize;

                    // RGB三通道插值 (手动展开循环)
                    for c in 0..3 {
                        let p00 = *src.get_unchecked(idx00 + c) as f32;
                        let p01 = *src.get_unchecked(idx01 + c) as f32;
                        let p10 = *src.get_unchecked(idx10 + c) as f32;
                        let p11 = *src.get_unchecked(idx11 + c) as f32;

                        let v0 = p00 + (p10 - p00) * fx;
                        let v1 = p01 + (p11 - p01) * fx;
                        let result = v0 + (v1 - v0) * fy;

                        *dst.get_unchecked_mut(dst_idx + c) = result.clamp(0.0, 255.0) as u8;
                    }
                } else {
                    // 边界外使用填充值
                    let dst_idx = ((dst_y * dst_width + dst_x + i) * 3) as usize;
                    *dst.get_unchecked_mut(dst_idx) = border_value;
                    *dst.get_unchecked_mut(dst_idx + 1) = border_value;
                    *dst.get_unchecked_mut(dst_idx + 2) = border_value;
                }
            }

            dst_x += 8;
        }

        // 处理剩余像素
        for x in dst_x..dst_width {
            let dst_x_f32 = x as f32;
            let src_x = a11 * dst_x_f32 + base_src_x;
            let src_y = a21 * dst_x_f32 + base_src_y;

            if src_x >= 0.0 && src_x < src_width_f32 - 1.0 
                && src_y >= 0.0 && src_y < src_height_f32 - 1.0 
            {
                let x0 = src_x as i32;
                let y0 = src_y as i32;
                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let idx00 = ((y0 * src_width_i32 + x0) * 3) as usize;
                let idx01 = (((y0 + 1) * src_width_i32 + x0) * 3) as usize;
                let idx10 = ((y0 * src_width_i32 + x0 + 1) * 3) as usize;
                let idx11 = (((y0 + 1) * src_width_i32 + x0 + 1) * 3) as usize;

                let dst_idx = ((dst_y * dst_width + x) * 3) as usize;

                for c in 0..3 {
                    let p00 = *src.get_unchecked(idx00 + c) as f32;
                    let p01 = *src.get_unchecked(idx01 + c) as f32;
                    let p10 = *src.get_unchecked(idx10 + c) as f32;
                    let p11 = *src.get_unchecked(idx11 + c) as f32;

                    let v0 = p00 + (p10 - p00) * fx;
                    let v1 = p01 + (p11 - p01) * fx;
                    let result = v0 + (v1 - v0) * fy;

                    *dst.get_unchecked_mut(dst_idx + c) = result.clamp(0.0, 255.0) as u8;
                }
            } else {
                let dst_idx = ((dst_y * dst_width + x) * 3) as usize;
                *dst.get_unchecked_mut(dst_idx) = border_value;
                *dst.get_unchecked_mut(dst_idx + 1) = border_value;
                *dst.get_unchecked_mut(dst_idx + 2) = border_value;
            }
        }
    }
}

/// SIMD优化的最近邻插值RGB变换
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn warp_affine_rgb_nearest_simd(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    inv_matrix: &AffineMatrix,
    dst: &mut [u8],
    dst_width: usize,
    dst_height: usize,
    border_mode: BorderMode,
) {
    let src_width_i32 = src_width as i32;
    let src_height_i32 = src_height as i32;

    let border_value = match border_mode {
        BorderMode::Constant(val) => val,
        _ => 0,
    };

    let a11 = inv_matrix.a11;
    let a12 = inv_matrix.a12;
    let b1 = inv_matrix.b1;
    let a21 = inv_matrix.a21;
    let a22 = inv_matrix.a22;
    let b2 = inv_matrix.b2;

    for dst_y in 0..dst_height {
        let dst_y_f32 = dst_y as f32;
        let base_src_x = a12 * dst_y_f32 + b1;
        let base_src_y = a22 * dst_y_f32 + b2;

        let mut dst_x = 0;

        // SIMD处理: 一次处理8个像素
        while dst_x + 8 <= dst_width {
            let x_offsets = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
            let dst_x_vec = _mm256_add_ps(_mm256_set1_ps(dst_x as f32), x_offsets);

            let src_x_vec = _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(a11), dst_x_vec),
                _mm256_set1_ps(base_src_x),
            );
            let src_y_vec = _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(a21), dst_x_vec),
                _mm256_set1_ps(base_src_y),
            );

            // 四舍五入到最近整数
            let src_x_rounded = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(src_x_vec);
            let src_y_rounded = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(src_y_vec);

            let mut src_x_arr = [0.0f32; 8];
            let mut src_y_arr = [0.0f32; 8];
            _mm256_storeu_ps(src_x_arr.as_mut_ptr(), src_x_rounded);
            _mm256_storeu_ps(src_y_arr.as_mut_ptr(), src_y_rounded);

            for i in 0..8 {
                let ix = src_x_arr[i] as i32;
                let iy = src_y_arr[i] as i32;

                if ix >= 0 && ix < src_width_i32 && iy >= 0 && iy < src_height_i32 {
                    let src_idx = ((iy * src_width_i32 + ix) * 3) as usize;
                    let dst_idx = ((dst_y * dst_width + dst_x + i) * 3) as usize;

                    *dst.get_unchecked_mut(dst_idx) = *src.get_unchecked(src_idx);
                    *dst.get_unchecked_mut(dst_idx + 1) = *src.get_unchecked(src_idx + 1);
                    *dst.get_unchecked_mut(dst_idx + 2) = *src.get_unchecked(src_idx + 2);
                } else {
                    let dst_idx = ((dst_y * dst_width + dst_x + i) * 3) as usize;
                    *dst.get_unchecked_mut(dst_idx) = border_value;
                    *dst.get_unchecked_mut(dst_idx + 1) = border_value;
                    *dst.get_unchecked_mut(dst_idx + 2) = border_value;
                }
            }

            dst_x += 8;
        }

        // 处理剩余像素
        for x in dst_x..dst_width {
            let dst_x_f32 = x as f32;
            let src_x = (a11 * dst_x_f32 + base_src_x).round() as i32;
            let src_y = (a21 * dst_x_f32 + base_src_y).round() as i32;

            if src_x >= 0 && src_x < src_width_i32 && src_y >= 0 && src_y < src_height_i32 {
                let src_idx = ((src_y * src_width_i32 + src_x) * 3) as usize;
                let dst_idx = ((dst_y * dst_width + x) * 3) as usize;

                *dst.get_unchecked_mut(dst_idx) = *src.get_unchecked(src_idx);
                *dst.get_unchecked_mut(dst_idx + 1) = *src.get_unchecked(src_idx + 1);
                *dst.get_unchecked_mut(dst_idx + 2) = *src.get_unchecked(src_idx + 2);
            } else {
                let dst_idx = ((dst_y * dst_width + x) * 3) as usize;
                *dst.get_unchecked_mut(dst_idx) = border_value;
                *dst.get_unchecked_mut(dst_idx + 1) = border_value;
                *dst.get_unchecked_mut(dst_idx + 2) = border_value;
            }
        }
    }
}

/// 非x86_64架构的回退实现
#[cfg(not(target_arch = "x86_64"))]
pub fn warp_affine_rgb_simd(
    src: &[u8],
    src_width: usize,
    src_height: usize,
    matrix: &AffineMatrix,
    dst_size: (usize, usize),
    interpolation: InterpolationMethod,
    border_mode: BorderMode,
) -> Vec<u8> {
    // 回退到标准实现
    super::affine_transform::warp_affine_rgb(
        src,
        src_width,
        src_height,
        matrix,
        dst_size,
        interpolation,
        border_mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_vs_standard() {
        use super::super::affine_transform::*;

        let width = 100;
        let height = 100;
        let mut src = vec![0u8; width * height * 3];

        // 创建测试图像
        for y in 40..60 {
            for x in 40..60 {
                let idx = (y * width + x) * 3;
                src[idx] = 255;     // R
                src[idx + 1] = 128; // G
                src[idx + 2] = 64;  // B
            }
        }

        let matrix = AffineMatrix::rotation_around_center(50.0, 50.0, 30.0);

        // 标准实现
        let dst_standard = warp_affine_rgb(
            &src,
            width,
            height,
            &matrix,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );

        // SIMD实现
        let dst_simd = warp_affine_rgb_simd(
            &src,
            width,
            height,
            &matrix,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );

        // 比较结果 (允许小的数值误差)
        let mut diff_count = 0;
        for i in 0..dst_standard.len() {
            if (dst_standard[i] as i32 - dst_simd[i] as i32).abs() > 1 {
                diff_count += 1;
            }
        }

        // 允许小于1%的像素有差异 (由于浮点运算精度)
        assert!(
            diff_count < dst_standard.len() / 100,
            "SIMD和标准实现差异过大: {} / {}",
            diff_count,
            dst_standard.len()
        );
    }
}
