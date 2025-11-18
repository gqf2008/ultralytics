use crate::xbus;

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
}

impl DecodeFilter {
    pub fn new() -> Self {
        Self {
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            decoder_name: String::from("Unknown"),
            dropped_frames: 0,
            total_frames: 0,
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

            // ✅ 新增：YUV数据采样检查 - 检测花屏/全黑/全白帧
            // 采样25个点检查是否有异常值(增加采样密度)
            let sample_points = [
                // 第一行
                (w / 6, h / 6),
                (w / 3, h / 6),
                (w / 2, h / 6),
                (2 * w / 3, h / 6),
                (5 * w / 6, h / 6),
                // 第二行
                (w / 6, h / 3),
                (w / 3, h / 3),
                (w / 2, h / 3),
                (2 * w / 3, h / 3),
                (5 * w / 6, h / 3),
                // 中间行
                (w / 6, h / 2),
                (w / 3, h / 2),
                (w / 2, h / 2),
                (2 * w / 3, h / 2),
                (5 * w / 6, h / 2),
                // 第四行
                (w / 6, 2 * h / 3),
                (w / 3, 2 * h / 3),
                (w / 2, 2 * h / 3),
                (2 * w / 3, 2 * h / 3),
                (5 * w / 6, 2 * h / 3),
                // 第五行
                (w / 6, 5 * h / 6),
                (w / 3, 5 * h / 6),
                (w / 2, 5 * h / 6),
                (2 * w / 3, 5 * h / 6),
                (5 * w / 6, 5 * h / 6),
            ];

            let mut y_sum = 0u32;
            let mut y_min = 255u8;
            let mut y_max = 0u8;

            for (sx, sy) in sample_points.iter() {
                let y_val = *data_y.add((sy * y_stride as u32 + sx) as usize);
                y_sum += y_val as u32;
                y_min = y_min.min(y_val);
                y_max = y_max.max(y_val);
            }

            let y_avg = (y_sum / sample_points.len() as u32) as u8;
            let y_range = y_max - y_min;

            // ✅ 只检测极端异常帧 - 降低误杀率
            // 组合条件: 同时满足低对比度+异常亮度才丢弃

            // 1. 全黑帧: 平均亮度<16 且 对比度<8
            if y_avg < 16 && y_range < 8 {
                self.dropped_frames += 1;
                if self.total_frames <= 50 || self.dropped_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 全黑帧 (Y平均={}, 范围={}, min={}, max={})",
                        self.total_frames, y_avg, y_range, y_min, y_max
                    );
                }
                return Ok(None);
            }

            // 2. 全白帧: 平均亮度>240 且 对比度<8
            if y_avg > 240 && y_range < 8 {
                self.dropped_frames += 1;
                if self.total_frames <= 50 || self.dropped_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 全白帧 (Y平均={}, 范围={}, min={}, max={})",
                        self.total_frames, y_avg, y_range, y_min, y_max
                    );
                }
                return Ok(None);
            }

            // 3. 中灰色单调帧: Y值在110-140之间 且 对比度<10 (只过滤真正的解码错误帧)
            // ⚠️ 放宽条件: 对比度<5 才算异常 (范围0-4是真正的解码错误)
            if y_avg >= 110 && y_avg <= 140 && y_range < 5 {
                self.dropped_frames += 1;
                if self.total_frames <= 50 || self.dropped_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 灰色单调帧 (Y平均={}, 范围={}, min={}, max={})",
                        self.total_frames, y_avg, y_range, y_min, y_max
                    );
                }
                return Ok(None);
            }

            // 4. 严重花屏: 对比度<3 (几乎完全单调)
            if y_range < 3 {
                self.dropped_frames += 1;
                if self.total_frames <= 50 || self.dropped_frames <= 10 {
                    println!(
                        "⚠️ 丢弃帧 #{}: 严重花屏 (Y平均={}, 范围={}, min={}, max={})",
                        self.total_frames, y_avg, y_range, y_min, y_max
                    );
                }
                return Ok(None);
            }

            // ❌ 移除关键帧检查 - 硬件解码器可能不设置此标志
            // 直接处理所有帧,依赖 decode_error_flags 来过滤损坏帧

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
                rgba_data,
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
