/// 解码模块 - 独立线程运行
/// 负责: RTSP解码 → 通过XBus发送DecodedFrame
use crate::rtsp;
use crate::rtsp_xbus::DecodedFrame;
use crate::xbus;

pub struct DecodeModule {
    rtsp_url: String,
    frame_id: u64,
}

impl DecodeModule {
    pub fn new(rtsp_url: String) -> Self {
        Self {
            rtsp_url,
            frame_id: 0,
        }
    }

    pub fn run(&mut self) {
        println!("🎬 解码模块启动");

        // 创建解码过滤器,接收回调
        let mut frame_id = 0u64;

        let filter = DecodeFilterXBus::new(|decoded_frame| {
            // 通过XBus发送解码帧
            xbus::post(decoded_frame);
        });

        // 开始解码 (这会阻塞当前线程)
        rtsp::adaptive_decode(&self.rtsp_url, filter);

        println!("❌ 解码模块退出");
    }
}

/// 解码过滤器 - 适配XBus
use ez_ffmpeg::filter::frame_filter::FrameFilter;
use ez_ffmpeg::filter::frame_filter_context::FrameFilterContext;
use ez_ffmpeg::{AVMediaType, Frame};
use std::time::Instant;

pub struct DecodeFilterXBus<F>
where
    F: Fn(DecodedFrame) + Send + 'static,
{
    callback: F,
    frame_id: u64,
    count: usize,
    last: Instant,
    current_fps: f64,
    decoder_name: String,
}

impl<F> DecodeFilterXBus<F>
where
    F: Fn(DecodedFrame) + Send + 'static,
{
    pub fn new(callback: F) -> Self {
        Self {
            callback,
            frame_id: 0,
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            decoder_name: String::from("Unknown"),
        }
    }
}

impl<F> Clone for DecodeFilterXBus<F>
where
    F: Fn(DecodedFrame) + Send + Clone + 'static,
{
    fn clone(&self) -> Self {
        Self {
            callback: self.callback.clone(),
            frame_id: self.frame_id,
            count: self.count,
            last: self.last,
            current_fps: self.current_fps,
            decoder_name: self.decoder_name.clone(),
        }
    }
}

impl<F> FrameFilter for DecodeFilterXBus<F>
where
    F: Fn(DecodedFrame) + Send + 'static,
{
    fn media_type(&self) -> AVMediaType {
        AVMediaType::AVMEDIA_TYPE_VIDEO
    }

    fn init(&mut self, _ctx: &FrameFilterContext) -> Result<(), String> {
        println!("✅ 解码线程启动");
        Ok(())
    }

    fn process(&mut self, frame: &mut Frame) -> Result<(), String> {
        self.count += 1;

        // YUV420P → RGBA转换 (极速批量转换)
        let w = frame.width();
        let h = frame.height();

        let mut rgba_data = vec![255u8; (w * h * 4) as usize]; // 预填充alpha=255

        unsafe {
            let data_y = frame.data(0);
            let data_u = frame.data(1);
            let data_v = frame.data(2);

            let y_stride = frame.linesize(0) as usize;
            let uv_stride = frame.linesize(1) as usize;

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
                    }
                }
            }

            // 计算FPS
            if self.last.elapsed().as_secs_f64() >= 1.0 {
                let elapsed = self.last.elapsed().as_secs_f64();
                self.current_fps = self.count as f64 / elapsed;

                println!(
                    "📺 解码统计: 解码{}帧 | {:.1}fps",
                    self.count, self.current_fps
                );

                self.last = Instant::now();
                self.count = 0;
            }

            // 构造消息并发送
            self.frame_id += 1;
            let decoded = DecodedFrame {
                rgba_data,
                width: w,
                height: h,
                frame_id: self.frame_id,
                decode_fps: self.current_fps,
                decoder_name: self.decoder_name.clone(),
            };

            (self.callback)(decoded);
        }

        Ok(())
    }

    fn set_codec_name(&mut self, name: &str) {
        self.decoder_name = name.to_string();
    }
}
