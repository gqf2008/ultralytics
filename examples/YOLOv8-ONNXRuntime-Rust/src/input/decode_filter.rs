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
        // ✅ 检查解码器代数ID，如果已过期则停止解码
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

            // 检查YUV数据指针和步长
            let data_y = (*frame.as_ptr()).data[0];
            let data_u = (*frame.as_ptr()).data[1];
            let data_v = (*frame.as_ptr()).data[2];
            let y_stride = (*frame.as_ptr()).linesize[0] as usize;
            let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

            if data_y.is_null() || data_u.is_null() || data_v.is_null() {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!("⚠️ 丢弃帧 #{}: YUV指针为空", self.total_frames);
                }
                return Ok(None);
            }

            // ✅ 新增：步长完整性检查 - 防止数据不完整
            if y_stride < w as usize || uv_stride < (w as usize / 2) {
                self.dropped_frames += 1;
                if self.total_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 步长异常 y_stride={} (需要>={}), uv_stride={} (需要>={})",
                        self.total_frames,
                        y_stride,
                        w,
                        uv_stride,
                        w / 2
                    );
                }
                return Ok(None);
            }

            self.count += 1;

            // YUV420P → RGBA (简化版，正确处理 stride)
            let pixel_count = (w * h) as usize;
            let mut rgba_data = vec![255u8; pixel_count * 4]; // alpha=255

            // ✅ 关键：按行处理，正确使用 stride
            for row in 0..h as usize {
                for col in 0..w as usize {
                    // 读取 YUV 值 (注意使用 stride)
                    let y_val = *data_y.add(row * y_stride + col) as f32;
                    let u_val = *data_u.add((row / 2) * uv_stride + col / 2) as f32 - 128.0;
                    let v_val = *data_v.add((row / 2) * uv_stride + col / 2) as f32 - 128.0;

                    // YUV → RGB (BT.601)
                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344 * u_val - 0.714 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                    // 写入 RGBA (连续内存)
                    let idx = (row * w as usize + col) * 4;
                    rgba_data[idx] = r;
                    rgba_data[idx + 1] = g;
                    rgba_data[idx + 2] = b;
                    // alpha 已经是 255
                }
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
                rgba_data: Arc::new(rgba_data),
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
