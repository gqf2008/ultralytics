//! 色彩空间转换函数
//! YUV/NV12 到 RGBA 的高效转换实现

/// YUV420P 到 RGBA 转换
///
/// # 参数
/// - `y_plane`: Y 平面数据
/// - `u_plane`: U 平面数据
/// - `v_plane`: V 平面数据
/// - `width`: 图像宽度
/// - `height`: 图像高度
/// - `y_stride`: Y 平面的步长 (linesize)
/// - `uv_stride`: UV 平面的步长 (linesize)
///
/// # 返回
/// RGBA 格式的字节数组 (宽 × 高 × 4)
///
/// # Safety
/// 调用者必须确保指针有效且不会越界
#[inline]
pub unsafe fn yuv420p_to_rgba(
    y_plane: *const u8,
    u_plane: *const u8,
    v_plane: *const u8,
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgba_vec = vec![255u8; pixel_count * 4];

    for y in 0..(height as usize) {
        for x in 0..(width as usize) {
            // 读取 YUV 值
            let y_val = *y_plane.add(y * y_stride + x) as i32;
            let u_val = *u_plane.add((y >> 1) * uv_stride + (x >> 1)) as i32 - 128;
            let v_val = *v_plane.add((y >> 1) * uv_stride + (x >> 1)) as i32 - 128;

            // YUV 到 RGB 转换 (BT.601 系数)
            // R = Y + 1.402 * V
            // G = Y - 0.344 * U - 0.714 * V
            // B = Y + 1.772 * U
            let r = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
            let g = (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7)).clamp(0, 255) as u8;
            let b = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;

            let idx = (y * width as usize + x) * 4;
            rgba_vec[idx] = r;
            rgba_vec[idx + 1] = g;
            rgba_vec[idx + 2] = b;
            // rgba_vec[idx + 3] = 255; // Alpha 已初始化
        }
    }

    rgba_vec
}

/// NV12 到 RGBA 转换
///
/// NV12 格式: Y 平面 + 交错的 UV 平面 (UVUVUV...)
///
/// # 参数
/// - `y_plane`: Y 平面数据
/// - `uv_plane`: 交错的 UV 平面数据
/// - `width`: 图像宽度
/// - `height`: 图像高度
/// - `y_stride`: Y 平面的步长 (linesize)
/// - `uv_stride`: UV 平面的步长 (linesize)
///
/// # 返回
/// RGBA 格式的字节数组 (宽 × 高 × 4)
///
/// # Safety
/// 调用者必须确保指针有效且不会越界
#[inline]
pub unsafe fn nv12_to_rgba(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgba_vec = vec![255u8; pixel_count * 4];

    for y in 0..(height as usize) {
        for x in 0..(width as usize) {
            // 读取 Y 值
            let y_val = *y_plane.add(y * y_stride + x) as i32;

            // NV12: UV 平面交错存储 [U0,V0,U1,V1,...]
            // 对于 4:2:0 采样,每 2x2 像素块共享一个 UV 值
            let uv_idx = (y >> 1) * uv_stride + (x & !1);
            let u_val = *uv_plane.add(uv_idx) as i32 - 128;
            let v_val = *uv_plane.add(uv_idx + 1) as i32 - 128;

            // YUV 到 RGB 转换 (BT.601 系数)
            let r = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
            let g = (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7)).clamp(0, 255) as u8;
            let b = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;

            let idx = (y * width as usize + x) * 4;
            rgba_vec[idx] = r;
            rgba_vec[idx + 1] = g;
            rgba_vec[idx + 2] = b;
            // rgba_vec[idx + 3] = 255; // Alpha 已初始化
        }
    }

    rgba_vec
}

/// NV12 到 RGBA 转换 (SIMD 优化版本)
///
/// 使用 AVX2/NEON 指令集加速转换,预期性能提升 2-4x
///
/// # 要求
/// - CPU 支持 AVX2 (x86_64) 或 NEON (ARM)
/// - 对于非 8 倍数宽度,会先用 SIMD 处理对齐部分,剩余部分用标量处理
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn nv12_to_rgba_simd(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    #[cfg(target_feature = "avx2")]
    {
        nv12_to_rgba_avx2(y_plane, uv_plane, width, height, y_stride, uv_stride)
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        // 运行时检测 AVX2 支持
        if is_x86_feature_detected!("avx2") {
            nv12_to_rgba_avx2(y_plane, uv_plane, width, height, y_stride, uv_stride)
        } else {
            nv12_to_rgba(y_plane, uv_plane, width, height, y_stride, uv_stride)
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn nv12_to_rgba_simd(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    nv12_to_rgba_neon(y_plane, uv_plane, width, height, y_stride, uv_stride)
}

/// NV12 到 RGBA 转换 (AVX2 实现)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nv12_to_rgba_avx2(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let pixel_count = (width * height) as usize;
    let mut rgba_vec = vec![255u8; pixel_count * 4];

    // YUV 转换系数 (BT.601, 扩展到 16 位定点数)
    let v_r_coef = _mm256_set1_epi16(179); // 1.402 * 128
    let u_g_coef = _mm256_set1_epi16(-44); // -0.344 * 128
    let v_g_coef = _mm256_set1_epi16(-91); // -0.714 * 128
    let u_b_coef = _mm256_set1_epi16(227); // 1.772 * 128
    let uv_bias = _mm256_set1_epi16(128);
    let zero = _mm256_setzero_si256();

    let width_simd = (width / 8) * 8; // 每次处理 8 像素

    for y in 0..(height as usize) {
        let y_row = y_plane.add(y * y_stride);
        let uv_row = uv_plane.add((y >> 1) * uv_stride);
        let rgba_row = rgba_vec.as_mut_ptr().add(y * width as usize * 4);

        let mut x = 0;

        // SIMD 处理对齐部分 (8 像素/循环)
        while x < width_simd as usize {
            // 加载 8 个 Y 值
            let y_vals_u8 = _mm_loadl_epi64(y_row.add(x) as *const __m128i);
            let y_vals = _mm256_cvtepu8_epi16(y_vals_u8);

            // 加载 4 对 UV 值 (因为 4:2:0 采样)
            let uv_vals_u8 = _mm_loadl_epi64(uv_row.add(x & !1) as *const __m128i);
            let uv_vals = _mm256_cvtepu8_epi16(uv_vals_u8);

            // 分离 U 和 V 通道 (交错 -> 分离)
            let shuffle_mask = _mm256_setr_epi8(
                0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12,
                14, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let u_vals_packed = _mm256_shuffle_epi8(uv_vals, shuffle_mask);
            let u_vals = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_vals_packed)),
                uv_bias,
            );

            let shuffle_mask_v = _mm256_setr_epi8(
                1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9, 11, 13,
                15, -1, -1, -1, -1, -1, -1, -1, -1,
            );
            let v_vals_packed = _mm256_shuffle_epi8(uv_vals, shuffle_mask_v);
            let v_vals = _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_vals_packed)),
                uv_bias,
            );

            // 每个 UV 值需要复制到 2 个像素 (水平方向)
            let u_vals_dup = _mm256_unpacklo_epi16(u_vals, u_vals);
            let v_vals_dup = _mm256_unpacklo_epi16(v_vals, v_vals);

            // YUV -> RGB 转换
            // R = Y + 1.402 * V
            let r_vals = _mm256_add_epi16(
                y_vals,
                _mm256_srai_epi16(_mm256_mullo_epi16(v_vals_dup, v_r_coef), 7),
            );
            // G = Y - 0.344 * U - 0.714 * V
            let g_vals = _mm256_add_epi16(
                y_vals,
                _mm256_srai_epi16(
                    _mm256_add_epi16(
                        _mm256_mullo_epi16(u_vals_dup, u_g_coef),
                        _mm256_mullo_epi16(v_vals_dup, v_g_coef),
                    ),
                    7,
                ),
            );
            // B = Y + 1.772 * U
            let b_vals = _mm256_add_epi16(
                y_vals,
                _mm256_srai_epi16(_mm256_mullo_epi16(u_vals_dup, u_b_coef), 7),
            );

            // 饱和到 [0, 255]
            let r_u8 = _mm256_packus_epi16(r_vals, zero);
            let g_u8 = _mm256_packus_epi16(g_vals, zero);
            let b_u8 = _mm256_packus_epi16(b_vals, zero);

            // 交错 RGBA (这部分比较复杂,简化处理)
            // 由于 AVX2 交错比较复杂,这里用标量写回
            let r_arr: [u8; 32] = std::mem::transmute(r_u8);
            let g_arr: [u8; 32] = std::mem::transmute(g_u8);
            let b_arr: [u8; 32] = std::mem::transmute(b_u8);

            for i in 0..8 {
                let rgba_idx = x * 4 + i * 4;
                *rgba_row.add(rgba_idx) = r_arr[i];
                *rgba_row.add(rgba_idx + 1) = g_arr[i];
                *rgba_row.add(rgba_idx + 2) = b_arr[i];
                *rgba_row.add(rgba_idx + 3) = 255;
            }

            x += 8;
        }

        // 处理剩余像素 (标量)
        while x < width as usize {
            let y_val = *y_row.add(x) as i32;
            let uv_idx = x & !1;
            let u_val = *uv_row.add(uv_idx) as i32 - 128;
            let v_val = *uv_row.add(uv_idx + 1) as i32 - 128;

            let r = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
            let g = (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7)).clamp(0, 255) as u8;
            let b = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;

            let rgba_idx = x * 4;
            *rgba_row.add(rgba_idx) = r;
            *rgba_row.add(rgba_idx + 1) = g;
            *rgba_row.add(rgba_idx + 2) = b;
            *rgba_row.add(rgba_idx + 3) = 255;

            x += 1;
        }
    }

    rgba_vec
}

/// NV12 到 RGBA 转换 (NEON 实现)
#[cfg(target_arch = "aarch64")]
unsafe fn nv12_to_rgba_neon(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    // NEON 实现留给 ARM 平台优化
    // 当前回退到标准实现
    nv12_to_rgba(y_plane, uv_plane, width, height, y_stride, uv_stride)
}

/// RGB24 到 RGBA 转换
///
/// 为 RGB24 数据添加 alpha 通道 (设为 255 - 完全不透明)
///
/// # 参数
/// - `rgb_plane`: RGB24 数据
/// - `width`: 图像宽度
/// - `height`: 图像高度
/// - `stride`: RGB 平面的步长 (linesize)
///
/// # 返回
/// RGBA 格式的字节数组 (宽 × 高 × 4)
///
/// # Safety
/// 调用者必须确保指针有效且不会越界
#[inline]
pub unsafe fn rgb24_to_rgba(
    rgb_plane: *const u8,
    width: u32,
    height: u32,
    stride: usize,
) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgba_vec = vec![255u8; pixel_count * 4];

    for y in 0..(height as usize) {
        let rgb_row = rgb_plane.add(y * stride);
        let rgba_row_start = y * width as usize * 4;

        for x in 0..(width as usize) {
            let rgb_idx = x * 3;
            let rgba_idx = rgba_row_start + x * 4;

            // 使用 copy_nonoverlapping 批量复制 RGB 三字节
            std::ptr::copy_nonoverlapping(
                rgb_row.add(rgb_idx),
                rgba_vec.as_mut_ptr().add(rgba_idx),
                3,
            );
            // rgba_vec[rgba_idx + 3] = 255; // Alpha 已初始化
        }
    }

    rgba_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nv12_to_rgba_basic() {
        // 创建 2x2 测试图像 (最小的 NV12)
        let width = 2;
        let height = 2;

        // Y 平面: 4 个像素
        let y_data = vec![128u8; 4]; // 中等亮度

        // UV 平面: 1 对 UV 值 (对应 2x2 块)
        let uv_data = vec![128u8, 128u8]; // 中性色度

        unsafe {
            let rgba = nv12_to_rgba(
                y_data.as_ptr(),
                uv_data.as_ptr(),
                width,
                height,
                width as usize,
                2,
            );

            // 验证输出大小
            assert_eq!(rgba.len(), (width * height * 4) as usize);

            // 验证 alpha 通道
            for i in (3..rgba.len()).step_by(4) {
                assert_eq!(rgba[i], 255);
            }
        }
    }

    #[test]
    fn test_yuv420p_to_rgba_basic() {
        let width = 2;
        let height = 2;

        let y_data = vec![128u8; 4];
        let u_data = vec![128u8; 1]; // 1x1 for 4:2:0
        let v_data = vec![128u8; 1];

        unsafe {
            let rgba = yuv420p_to_rgba(
                y_data.as_ptr(),
                u_data.as_ptr(),
                v_data.as_ptr(),
                width,
                height,
                width as usize,
                1,
            );

            assert_eq!(rgba.len(), (width * height * 4) as usize);
        }
    }

    #[test]
    fn test_rgb24_to_rgba() {
        let width = 2;
        let height = 2;

        // RGB24: RGBRGBRGBRGB
        let rgb_data = vec![
            255, 0, 0, // 红色
            0, 255, 0, // 绿色
            0, 0, 255, // 蓝色
            255, 255, 0, // 黄色
        ];

        unsafe {
            let rgba = rgb24_to_rgba(rgb_data.as_ptr(), width, height, (width * 3) as usize);

            assert_eq!(rgba.len(), (width * height * 4) as usize);

            // 验证第一个像素 (红色)
            assert_eq!(rgba[0], 255); // R
            assert_eq!(rgba[1], 0); // G
            assert_eq!(rgba[2], 0); // B
            assert_eq!(rgba[3], 255); // A
        }
    }
}
