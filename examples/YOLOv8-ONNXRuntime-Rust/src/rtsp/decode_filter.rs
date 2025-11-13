/// FFmpeg解码过滤器模块
/// FFmpeg decode filter module
use super::types::DecodedFrame;
use crossbeam_channel::Sender;
use ez_ffmpeg::filter::frame_filter::FrameFilter;
use ez_ffmpeg::filter::frame_filter_context::FrameFilterContext;
use ez_ffmpeg::{AVMediaType, Frame};
use std::time::Instant;

/// FFmpeg解码过滤器: RTSP流 → RGBA帧 (双输出:渲染+检测)
#[derive(Clone)]
pub struct DecodeFilter {
    pub tx_render: Sender<DecodedFrame>,      // 发送给渲染线程
    pub tx_inference: Sender<DecodedFrame>,   // 发送给检测线程
    pub count: usize,
    pub last: Instant,
    pub current_fps: f64,
    pub decoder_name: String, // 当前使用的解码器名称
}

impl DecodeFilter {
    pub fn new(tx_render: Sender<DecodedFrame>, tx_inference: Sender<DecodedFrame>) -> Self {
        Self {
            tx_render,
            tx_inference,
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            decoder_name: String::from("Unknown"),
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
            if frame.as_ptr().is_null() {
                return Ok(Some(frame));
            }

            let w = (*frame.as_ptr()).width as u32;
            let h = (*frame.as_ptr()).height as u32;

            self.count += 1;

            // YUV420P → RGBA (BT.601) - 极速优化版
            let data_y = (*frame.as_ptr()).data[0];
            let data_u = (*frame.as_ptr()).data[1];
            let data_v = (*frame.as_ptr()).data[2];
            let y_stride = (*frame.as_ptr()).linesize[0] as usize;
            let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

            let pixel_count = (w * h) as usize;
            let mut rgba_data = vec![255u8; pixel_count * 4]; // 预填充alpha=255

            // 批量处理: 每次4个像素 (SIMD优化)
            let rgba_ptr = rgba_data.as_mut_ptr();
            for y in 0..h as usize {
                for x in (0..w as usize).step_by(4) {
                    for offset in 0..4.min(w as usize - x) {
                        let px = x + offset;
                        let y_val = *data_y.add(y * y_stride + px) as f32;
                        let u_val = *data_u.add((y / 2) * uv_stride + px / 2) as f32 - 128.0;
                        let v_val = *data_v.add((y / 2) * uv_stride + px / 2) as f32 - 128.0;

                        let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                        let g = (y_val - 0.344 * u_val - 0.714 * v_val).clamp(0.0, 255.0) as u8;
                        let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

                        let idx = (y * w as usize + px) * 4;
                        *rgba_ptr.add(idx) = r;
                        *rgba_ptr.add(idx + 1) = g;
                        *rgba_ptr.add(idx + 2) = b;
                        // alpha已经是255,无需设置
                    }
                }
            }

            // 计算FPS
            if self.last.elapsed().as_secs_f64() >= 1.0 {
                let elapsed = self.last.elapsed().as_secs_f64();
                self.current_fps = self.count as f64 / elapsed;

                // 每秒打印一次解码统计
                println!(
                    "📺 解码统计: 解码{}帧 | 实际{:.1}fps",
                    self.count, self.current_fps
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

            // 发送到两个channel: 渲染线程 + 检测线程
            // 使用try_send避免阻塞
            let _ = self.tx_render.try_send(decoded.clone());
            let _ = self.tx_inference.try_send(decoded);

            Ok(Some(frame))
        }
    }

    fn uninit(&mut self, _ctx: &FrameFilterContext) {
        println!("✅ 解码线程退出");
    }
}
