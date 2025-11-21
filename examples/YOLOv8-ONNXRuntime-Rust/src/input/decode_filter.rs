use super::decoder_manager::ACTIVE_DECODER_GENERATION;
use crate::xbus;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// FFmpeg解码过滤器模块
/// FFmpeg decode filter module
use crate::detection::types::DecodedFrame;
use ez_ffmpeg::filter::frame_filter::FrameFilter;
use ez_ffmpeg::filter::frame_filter_context::FrameFilterContext;
use ez_ffmpeg::{AVMediaType, Frame};
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// FFmpeg解码过滤器: RTSP流 → RGBA帧 (极速优化版)
#[derive(Clone)]
pub struct DecodeFilter {
    pub count: usize,
    pub last: Instant,
    pub current_fps: f64,
    pub decoder_name: String,  // 当前使用的解码器名称
    pub dropped_frames: usize, // 丢弃的帧数
    pub total_frames: usize,   // 总帧数
    pub generation: usize,     // 解码器代数ID
    buffer: Arc<Vec<u8>>,      // Arc包装避免每帧clone
}

impl DecodeFilter {
    pub fn new(generation: usize) -> Self {
        Self {
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            decoder_name: String::from("Unknown"),
            dropped_frames: 0,
            total_frames: 0,
            generation,
            buffer: Arc::new(Vec::new()),
        }
    }
}

impl FrameFilter for DecodeFilter {
    fn media_type(&self) -> AVMediaType {
        AVMediaType::AVMEDIA_TYPE_VIDEO
    }

    fn init(&mut self, _ctx: &FrameFilterContext) -> Result<(), String> {
        println!("✅ 解码线程启动");
        Ok(())
    }

    fn filter_frame(
        &mut self,
        frame: Frame,
        _ctx: &FrameFilterContext,
    ) -> Result<Option<Frame>, String> {
        // 检查解码器代数ID,如果已过期则停止解码
        let current_gen = ACTIVE_DECODER_GENERATION.load(Ordering::Relaxed);
        if self.generation != current_gen {
            println!(
                "🛑 解码器已过期 (Gen: {} != Current: {}), 停止解码",
                self.generation, current_gen
            );
            return Err("Decoder expired".to_string());
        }

        unsafe {
            self.total_frames += 1;

            // 基本检查：空帧或损坏帧
            if frame.as_ptr().is_null() || frame.is_empty() || frame.is_corrupt() {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!("⚠️ 丢弃帧 #{}: 空帧/损坏帧", self.total_frames);
                }
                return Ok(None);
            }

            let w = (*frame.as_ptr()).width as u32;
            let h = (*frame.as_ptr()).height as u32;

            // 检查分辨率合法性
            if w == 0 || h == 0 || w > 4096 || h > 4096 {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!("⚠️ 丢弃帧 #{}: 非法分辨率 {}x{}", self.total_frames, w, h);
                }
                return Ok(None);
            }

            // ✅ 关键：检查 FFmpeg 的错误标志位
            let decode_error_flags = (*frame.as_ptr()).decode_error_flags;
            // 只丢弃严重错误的帧 (缺少参考帧、无效比特流)
            if decode_error_flags & 0x03 != 0 {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 解码错误标志=0x{:02x}",
                        self.total_frames, decode_error_flags
                    );
                }
                return Ok(None);
            }

            // YUV420P数据指针
            let y_plane = (*frame.as_ptr()).data[0];
            let u_plane = (*frame.as_ptr()).data[1];
            let v_plane = (*frame.as_ptr()).data[2];
            let y_stride = (*frame.as_ptr()).linesize[0] as usize;
            let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

            if y_plane.is_null() || u_plane.is_null() || v_plane.is_null() {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!("⚠️ 丢弃帧 #{}: YUV指针为空", self.total_frames);
                }
                return Ok(None);
            }

            if y_stride < w as usize || uv_stride < (w as usize / 2) {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 步长异常 y_stride={} uv_stride={}",
                        self.total_frames, y_stride, uv_stride
                    );
                }
                return Ok(None);
            }

            self.count += 1;

            // YUV420P → RGBA (SIMD优化版 - AVX2加速)
            let pixel_count = (w * h) as usize;
            let required_size = pixel_count * 4;

            // 只在尺寸变化时重新分配Arc
            if Arc::strong_count(&self.buffer) > 1 || self.buffer.len() != required_size {
                self.buffer = Arc::new(vec![255; required_size]);
            }

            let w_usize = w as usize;
            let h_usize = h as usize;

            // 获取可变引用并使用SIMD优化的YUV转换
            let buffer = Arc::get_mut(&mut self.buffer).unwrap();

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    yuv420p_to_rgba_avx2(
                        y_plane, u_plane, v_plane, y_stride, uv_stride, buffer, w_usize, h_usize,
                    );
                } else {
                    yuv420p_to_rgba_scalar(
                        y_plane, u_plane, v_plane, y_stride, uv_stride, buffer, w_usize, h_usize,
                    );
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                yuv420p_to_rgba_scalar(
                    y_plane, u_plane, v_plane, y_stride, uv_stride, buffer, w_usize, h_usize,
                );
            }

            // 计算FPS
            if self.last.elapsed().as_secs_f64() >= 1.0 {
                let elapsed = self.last.elapsed().as_secs_f64();
                self.current_fps = self.count as f64 / elapsed;
                let drop_rate = self.dropped_frames as f64 / self.total_frames as f64 * 100.0;

                // 每秒打印一次解码统计
                println!(
                    "📺 解码统计: 解码{}帧 | 实际{:.1}fps | 总帧{} | 丢弃{} ({:.1}%)",
                    self.count, self.current_fps, self.total_frames, self.dropped_frames, drop_rate
                );

                self.last = Instant::now();
                self.count = 0;
            }

            let decoded = DecodedFrame {
                rgba_data: Arc::clone(&self.buffer), // 零拷贝共享
                width: w,
                height: h,
                decode_fps: self.current_fps,
                decoder_name: self.decoder_name.clone(),
            };

            xbus::post(decoded);

            Ok(Some(frame))
        }
    }

    fn uninit(&mut self, _ctx: &FrameFilterContext) {
        println!("✅ 解码线程退出");
    }
}

/// 标量版本YUV转换(fallback)
#[inline]
unsafe fn yuv420p_to_rgba_scalar(
    y_plane: *const u8,
    u_plane: *const u8,
    v_plane: *const u8,
    y_stride: usize,
    uv_stride: usize,
    buffer: &mut [u8],
    width: usize,
    height: usize,
) {
    let mut out_idx = 0;
    for y in 0..height {
        let y_row = y * y_stride;
        let uv_row = (y >> 1) * uv_stride;

        for x in 0..width {
            let y_val = *y_plane.add(y_row + x) as i32;
            let u_val = *u_plane.add(uv_row + (x >> 1)) as i32 - 128;
            let v_val = *v_plane.add(uv_row + (x >> 1)) as i32 - 128;

            buffer[out_idx] = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
            buffer[out_idx + 1] =
                (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7)).clamp(0, 255) as u8;
            buffer[out_idx + 2] = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;
            out_idx += 4;
        }
    }
}

/// AVX2优化版本YUV转换(16像素并行)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn yuv420p_to_rgba_avx2(
    y_plane: *const u8,
    u_plane: *const u8,
    v_plane: *const u8,
    y_stride: usize,
    uv_stride: usize,
    buffer: &mut [u8],
    width: usize,
    height: usize,
) {
    let chunks = width / 16;
    let remainder = width % 16;

    for row in 0..height {
        let y_row = row * y_stride;
        let uv_row = (row >> 1) * uv_stride;
        let out_row = row * width * 4;

        // 处理16像素对齐部分
        for chunk in 0..chunks {
            let x = chunk * 16;
            let y_ptr = y_plane.add(y_row + x);
            let u_ptr = u_plane.add(uv_row + (x >> 1));
            let v_ptr = v_plane.add(uv_row + (x >> 1));

            // 加载16个Y值
            let y_vec = _mm_loadu_si128(y_ptr as *const __m128i);

            // 加载8个U/V值(4:2:0子采样)
            let uv_vec = _mm_loadl_epi64(u_ptr as *const __m128i);
            let vv_vec = _mm_loadl_epi64(v_ptr as *const __m128i);

            // 扩展U/V到16个值(每个值重复2次)
            let u_vec_low = _mm_unpacklo_epi8(uv_vec, uv_vec);
            let v_vec_low = _mm_unpacklo_epi8(vv_vec, vv_vec);

            // 转换为16位有符号数
            let y_16_lo = _mm_cvtepu8_epi16(y_vec);
            let y_16_hi = _mm_cvtepu8_epi16(_mm_srli_si128(y_vec, 8));
            let u_16 = _mm_cvtepu8_epi16(u_vec_low);
            let v_16 = _mm_cvtepu8_epi16(v_vec_low);

            // U/V减128
            let u_bias = _mm_sub_epi16(u_16, _mm_set1_epi16(128));
            let v_bias = _mm_sub_epi16(v_16, _mm_set1_epi16(128));

            // BT.601系数(乘以128以避免浮点)
            let coef_r_v = _mm_set1_epi16(179); // 1.402 * 128
            let coef_g_u = _mm_set1_epi16(-44); // -0.344 * 128
            let coef_g_v = _mm_set1_epi16(-91); // -0.714 * 128
            let coef_b_u = _mm_set1_epi16(227); // 1.772 * 128

            // 计算RGB(低8像素)
            let r_offset_lo = _mm_srai_epi16(_mm_mullo_epi16(v_bias, coef_r_v), 7);
            let g_offset_u_lo = _mm_srai_epi16(_mm_mullo_epi16(u_bias, coef_g_u), 7);
            let g_offset_v_lo = _mm_srai_epi16(_mm_mullo_epi16(v_bias, coef_g_v), 7);
            let b_offset_lo = _mm_srai_epi16(_mm_mullo_epi16(u_bias, coef_b_u), 7);

            let r_lo = _mm_add_epi16(y_16_lo, r_offset_lo);
            let g_lo = _mm_add_epi16(y_16_lo, _mm_add_epi16(g_offset_u_lo, g_offset_v_lo));
            let b_lo = _mm_add_epi16(y_16_lo, b_offset_lo);

            // 饱和转换为u8
            let r_u8_lo = _mm_packus_epi16(r_lo, r_lo);
            let g_u8_lo = _mm_packus_epi16(g_lo, g_lo);
            let b_u8_lo = _mm_packus_epi16(b_lo, b_lo);
            let a_u8 = _mm_set1_epi8(-1); // alpha=255

            // 交错RGBA(低8像素)
            let rg_lo = _mm_unpacklo_epi8(r_u8_lo, g_u8_lo);
            let ba_lo = _mm_unpacklo_epi8(b_u8_lo, a_u8);
            let rgba_0 = _mm_unpacklo_epi16(rg_lo, ba_lo);
            let rgba_1 = _mm_unpackhi_epi16(rg_lo, ba_lo);

            // 存储低8像素
            let out_ptr = buffer.as_mut_ptr().add(out_row + x * 4);
            _mm_storeu_si128(out_ptr as *mut __m128i, rgba_0);
            _mm_storeu_si128(out_ptr.add(16) as *mut __m128i, rgba_1);

            // 处理高8像素(类似逻辑)
            let u_bias_hi = _mm_sub_epi16(
                _mm_cvtepu8_epi16(_mm_srli_si128(u_vec_low, 8)),
                _mm_set1_epi16(128),
            );
            let v_bias_hi = _mm_sub_epi16(
                _mm_cvtepu8_epi16(_mm_srli_si128(v_vec_low, 8)),
                _mm_set1_epi16(128),
            );

            let r_offset_hi = _mm_srai_epi16(_mm_mullo_epi16(v_bias_hi, coef_r_v), 7);
            let g_offset_u_hi = _mm_srai_epi16(_mm_mullo_epi16(u_bias_hi, coef_g_u), 7);
            let g_offset_v_hi = _mm_srai_epi16(_mm_mullo_epi16(v_bias_hi, coef_g_v), 7);
            let b_offset_hi = _mm_srai_epi16(_mm_mullo_epi16(u_bias_hi, coef_b_u), 7);

            let r_hi = _mm_add_epi16(y_16_hi, r_offset_hi);
            let g_hi = _mm_add_epi16(y_16_hi, _mm_add_epi16(g_offset_u_hi, g_offset_v_hi));
            let b_hi = _mm_add_epi16(y_16_hi, b_offset_hi);

            let r_u8_hi = _mm_packus_epi16(r_hi, r_hi);
            let g_u8_hi = _mm_packus_epi16(g_hi, g_hi);
            let b_u8_hi = _mm_packus_epi16(b_hi, b_hi);

            let rg_hi = _mm_unpacklo_epi8(r_u8_hi, g_u8_hi);
            let ba_hi = _mm_unpacklo_epi8(b_u8_hi, a_u8);
            let rgba_2 = _mm_unpacklo_epi16(rg_hi, ba_hi);
            let rgba_3 = _mm_unpackhi_epi16(rg_hi, ba_hi);

            _mm_storeu_si128(out_ptr.add(32) as *mut __m128i, rgba_2);
            _mm_storeu_si128(out_ptr.add(48) as *mut __m128i, rgba_3);
        }

        // 处理剩余像素
        if remainder > 0 {
            let x = chunks * 16;
            for i in 0..remainder {
                let y_val = *y_plane.add(y_row + x + i) as i32;
                let u_val = *u_plane.add(uv_row + ((x + i) >> 1)) as i32 - 128;
                let v_val = *v_plane.add(uv_row + ((x + i) >> 1)) as i32 - 128;

                let out_idx = out_row + (x + i) * 4;
                buffer[out_idx] = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
                buffer[out_idx + 1] =
                    (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7)).clamp(0, 255) as u8;
                buffer[out_idx + 2] = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;
            }
        }
    }
}
