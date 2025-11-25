/// 桌面捕获解码器 (Rust风格封装,基于 ffmpeg-next)
use super::{DecoderError, VideoFrame};
use std::sync::Arc;
use ffmpeg::format::{input, Pixel};
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{context::Context as ScalerContext, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;

/// 桌面捕获配置
pub struct DesktopConfig {
    pub fps: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            fps: 30,
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        }
    }
}

/// 桌面捕获解码器
pub struct DesktopDecoder {
    input_context: ffmpeg::format::context::Input,
    decoder: ffmpeg::codec::decoder::Video,
    scaler: ScalerContext,
    video_stream_index: usize,
    width: u32,
    height: u32,
    // 缓冲区复用 (避免每帧分配)
    y_buffer: Vec<u8>,
    uv_buffer: Vec<u8>,
}

impl DesktopDecoder {
    /// 创建桌面捕获解码器
    pub fn new(config: DesktopConfig) -> Result<Self, DecoderError> {
        ffmpeg::init().map_err(|e| {
            DecoderError::Initialization(format!("FFmpeg initialization failed: {}", e))
        })?;

        // Windows: gdigrab, Linux: x11grab, macOS: avfoundation
        #[cfg(target_os = "windows")]
        let format_name = "gdigrab";
        #[cfg(target_os = "linux")]
        let format_name = "x11grab";
        #[cfg(target_os = "macos")]
        let format_name = "avfoundation";

        #[cfg(target_os = "windows")]
        let device_path = "desktop";
        #[cfg(target_os = "linux")]
        let device_path = format!(":0.0+{},{}", config.x, config.y);
        #[cfg(target_os = "macos")]
        let device_path = "Capture screen 0";

        println!("🖥️ 启动桌面捕获: {}", device_path);

        // 设置输入选项
        let mut options = ffmpeg::Dictionary::new();
        options.set("framerate", &config.fps.to_string());

        #[cfg(not(target_os = "windows"))]
        {
            options.set("video_size", &format!("{}x{}", config.width, config.height));
        }

        // 打开输入
        let mut input_context = input(&device_path)
            .map_err(|e| DecoderError::OpenStream(format!("Failed to open desktop: {}", e)))?;

        // 查找视频流
        let video_stream = input_context
            .streams()
            .best(Type::Video)
            .ok_or_else(|| DecoderError::NoVideoStream)?;
        let video_stream_index = video_stream.index();

        println!("✅ 桌面捕获流打开成功");

        // 获取解码器
        let context_decoder = ffmpeg::codec::context::Context::from_parameters(
            video_stream.parameters(),
        )
        .map_err(|e| DecoderError::Initialization(format!("Failed to create context: {}", e)))?;
        let decoder = context_decoder
            .decoder()
            .video()
            .map_err(|e| DecoderError::Initialization(format!("Failed to get decoder: {}", e)))?;

        let width = decoder.width();
        let height = decoder.height();

        println!("🔧 解码器: {}", decoder.codec().unwrap().name());
        println!("📐 分辨率: {}x{}", width, height);

        // 创建缩放器 (转换为 YUV420P)
        let scaler = ScalerContext::get(
            decoder.format(),
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            Flags::BILINEAR,
        )
        .map_err(|e| DecoderError::Initialization(format!("Failed to create scaler: {}", e)))?;

        // 预分配缓冲区
        let y_size = (width * height) as usize;
        let uv_size = (width * height / 2) as usize;

        Ok(Self {
            input_context,
            decoder,
            scaler,
            video_stream_index,
            width,
            height,
            y_buffer: vec![0u8; y_size],
            uv_buffer: vec![0u8; uv_size],
        })
    }

    /// 解码下一帧
    pub fn decode_next_frame(&mut self) -> Result<Option<VideoFrame>, DecoderError> {
        // 读取包
        for (stream, packet) in self.input_context.packets() {
            if stream.index() == self.video_stream_index {
                self.decoder.send_packet(&packet).map_err(|e| {
                    DecoderError::SendPacket(format!("Failed to send packet: {}", e))
                })?;

                let mut decoded_frame = Video::empty();
                while self.decoder.receive_frame(&mut decoded_frame).is_ok() {
                    // 转换为 YUV420P
                    let mut yuv_frame = Video::empty();
                    self.scaler
                        .run(&decoded_frame, &mut yuv_frame)
                        .map_err(|e| {
                            DecoderError::UnsupportedFormat(format!("Failed to scale frame: {}", e))
                        })?;

                    return self.convert_yuv_to_nv12(&yuv_frame);
                }
            }
        }

        Ok(None)
    }

    /// 转换 YUV420P 为 NV12 格式
    fn convert_yuv_to_nv12(
        &mut self,
        yuv_frame: &Video,
    ) -> Result<Option<VideoFrame>, DecoderError> {
        let width = yuv_frame.width();
        let height = yuv_frame.height();

        unsafe {
            let y_plane = yuv_frame.data(0);
            let u_plane = yuv_frame.data(1);
            let v_plane = yuv_frame.data(2);
            let y_stride = yuv_frame.stride(0);
            let u_stride = yuv_frame.stride(1);

            let y_size = (height as usize) * y_stride;

            // 复用预分配的缓冲区
            self.y_buffer.resize(y_size, 0);
            self.uv_buffer
                .resize(((height / 2) as usize) * (width as usize), 0);

            // 复制 Y 平面
            std::ptr::copy_nonoverlapping(y_plane.as_ptr(), self.y_buffer.as_mut_ptr(), y_size);

            // 交错 U 和 V 平面为 NV12 格式
            for i in 0..(height / 2) as usize {
                for j in 0..(width / 2) as usize {
                    let u_idx = i * u_stride + j;
                    let v_idx = i * u_stride + j;
                    let nv12_idx = i * (width as usize) + j * 2;

                    self.uv_buffer[nv12_idx] = u_plane[u_idx];
                    self.uv_buffer[nv12_idx + 1] = v_plane[v_idx];
                }
            }

            Ok(Some(VideoFrame {
                y_data: Arc::new(self.y_buffer.clone()),
                uv_data: Arc::new(self.uv_buffer.clone()),
                width,
                height,
                y_stride,
                uv_stride: width as usize,
            }))
        }
    }

    /// 获取视频信息
    pub fn video_info(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
