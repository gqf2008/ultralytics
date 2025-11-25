//! Intel QSV 硬件解码器封装
//!
//! 提供 Rust 风格的 API,隐藏 FFmpeg C API 的复杂性

use ffmpeg_next as ffmpeg;
use ffmpeg_next::ffi;
use std::ptr;
use std::sync::Arc;

/// 硬件加速类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareAccel {
    /// Intel QSV (通过 D3D11VA)
    Qsv,
    /// NVIDIA NVDEC
    Cuda,
    /// 软件解码
    None,
}

/// 解码器配置
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// RTSP URL
    pub url: String,
    /// 硬件加速类型
    pub hardware_accel: HardwareAccel,
    /// RTSP 传输协议
    pub rtsp_transport: RtspTransport,
    /// 网络缓冲大小 (字节)
    pub buffer_size: usize,
    /// 最大延迟 (微秒)
    pub max_delay: i32,
}

/// RTSP 传输协议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtspTransport {
    Tcp,
    Udp,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            hardware_accel: HardwareAccel::Qsv,
            rtsp_transport: RtspTransport::Tcp,
            buffer_size: 5 * 1024 * 1024, // 5MB
            max_delay: 0,
        }
    }
}

/// 解码后的视频帧 (使用 Arc 共享数据,避免克隆)
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// NV12 格式的 Y 平面数据 (共享)
    pub y_data: Arc<Vec<u8>>,
    /// NV12 格式的 UV 平面数据 (共享)
    pub uv_data: Arc<Vec<u8>>,
    /// 帧宽度
    pub width: u32,
    /// 帧高度
    pub height: u32,
    /// Y 平面步长
    pub y_stride: usize,
    /// UV 平面步长
    pub uv_stride: usize,
}

/// QSV 硬件解码器
pub struct QsvDecoder {
    input_context: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    video_stream_index: usize,
    hw_frame: ffmpeg::frame::Video,
    cpu_frame: ffmpeg::frame::Video,
    first_frame: bool,
    // 缓冲区复用 (避免每帧分配)
    y_buffer: Vec<u8>,
    uv_buffer: Vec<u8>,
}

impl QsvDecoder {
    /// 创建新的解码器
    pub fn new(config: DecoderConfig) -> Result<Self, DecoderError> {
        ffmpeg::init().map_err(|e| DecoderError::Initialization(e.to_string()))?;

        // 配置输入选项
        let mut opts = ffmpeg::Dictionary::new();
        opts.set(
            "rtsp_transport",
            match config.rtsp_transport {
                RtspTransport::Tcp => "tcp",
                RtspTransport::Udp => "udp",
            },
        );
        opts.set("buffer_size", &config.buffer_size.to_string());
        opts.set("max_delay", &config.max_delay.to_string());

        let input_context = ffmpeg::format::input_with_dictionary(&config.url, opts)
            .map_err(|e| DecoderError::OpenStream(e.to_string()))?;

        println!("✅ RTSP 流打开成功");

        // 查找视频流
        let video_stream = input_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(DecoderError::NoVideoStream)?;

        let video_stream_index = video_stream.index();
        let codec_params = video_stream.parameters();

        // 选择解码器
        let codec = match config.hardware_accel {
            HardwareAccel::Qsv => ffmpeg::decoder::find_by_name("hevc_qsv")
                .or_else(|| ffmpeg::decoder::find_by_name("h264_qsv"))
                .ok_or(DecoderError::NoHardwareDecoder)?,
            HardwareAccel::Cuda => ffmpeg::decoder::find_by_name("hevc_cuvid")
                .or_else(|| ffmpeg::decoder::find_by_name("h264_cuvid"))
                .ok_or(DecoderError::NoHardwareDecoder)?,
            HardwareAccel::None => {
                ffmpeg::decoder::find(codec_params.id()).ok_or(DecoderError::NoDecoder)?
            }
        };

        println!("🔧 解码器: {}", codec.name());

        // 创建解码器上下文
        let mut context = ffmpeg::codec::context::Context::from_parameters(codec_params)
            .map_err(|e| DecoderError::CreateContext(e.to_string()))?;

        // 配置硬件加速
        if config.hardware_accel == HardwareAccel::Qsv {
            Self::setup_qsv_hardware(&mut context)?;
        }

        context.set_threading(ffmpeg::threading::Config::count(1));
        let decoder = context
            .decoder()
            .video()
            .map_err(|e| DecoderError::CreateDecoder(e.to_string()))?;

        println!("📐 视频: {}x{}", decoder.width(), decoder.height());
        println!("🎨 像素格式: {:?}", decoder.format());

        let hw_frame = ffmpeg::frame::Video::empty();
        let cpu_frame = ffmpeg::frame::Video::empty();

        Ok(Self {
            input_context,
            decoder,
            video_stream_index,
            hw_frame,
            cpu_frame,
            first_frame: true,
            y_buffer: vec![0u8; 1920 * 1080],
            uv_buffer: vec![0u8; 1920 * 1080 / 2],
        })
    }

    /// 配置 QSV 硬件加速
    fn setup_qsv_hardware(
        context: &mut ffmpeg::codec::context::Context,
    ) -> Result<(), DecoderError> {
        unsafe {
            let mut hw_device_ctx: *mut ffi::AVBufferRef = ptr::null_mut();
            let device_type = ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA;

            let ret = ffi::av_hwdevice_ctx_create(
                &mut hw_device_ctx,
                device_type,
                ptr::null(),
                ptr::null_mut(),
                0,
            );

            if ret < 0 || hw_device_ctx.is_null() {
                return Err(DecoderError::HardwareSetup(format!(
                    "创建 D3D11VA 设备失败: {}",
                    ret
                )));
            }

            (*context.as_mut_ptr()).hw_device_ctx = ffi::av_buffer_ref(hw_device_ctx);
            ffi::av_buffer_unref(&mut hw_device_ctx);

            if (*context.as_mut_ptr()).hw_device_ctx.is_null() {
                return Err(DecoderError::HardwareSetup("引用硬件设备上下文失败".into()));
            }

            (*context.as_mut_ptr()).get_format = Some(get_format_callback);
            println!("✅ QSV 硬件加速 (D3D11VA) 已启用");
        }

        Ok(())
    }

    /// 解码下一帧
    pub fn decode_next_frame(&mut self) -> Result<Option<VideoFrame>, DecoderError> {
        loop {
            // 尝试接收已解码的帧
            if self.decoder.receive_frame(&mut self.hw_frame).is_ok() {
                return self.process_decoded_frame();
            }

            // 读取并发送新的数据包
            let mut found_packet = false;
            for (stream, packet) in self.input_context.packets() {
                if stream.index() == self.video_stream_index {
                    self.decoder
                        .send_packet(&packet)
                        .map_err(|e| DecoderError::SendPacket(e.to_string()))?;
                    found_packet = true;
                    break;
                }
            }

            if !found_packet {
                // 流结束
                return Ok(None);
            }
        }
    }

    /// 处理已解码的帧
    fn process_decoded_frame(&mut self) -> Result<Option<VideoFrame>, DecoderError> {
        let format = self.hw_frame.format();

        // 处理硬件帧
        let (final_frame, width, height) =
            if format == ffmpeg::format::Pixel::D3D11 || format == ffmpeg::format::Pixel::QSV {
                unsafe {
                    let ret = ffi::av_hwframe_transfer_data(
                        self.cpu_frame.as_mut_ptr(),
                        self.hw_frame.as_ptr(),
                        0,
                    );

                    if ret < 0 {
                        return Err(DecoderError::TransferFrame(format!(
                            "GPU→CPU 传输失败: {}",
                            ret
                        )));
                    }
                }

                (
                    &self.cpu_frame,
                    self.cpu_frame.width(),
                    self.cpu_frame.height(),
                )
            } else {
                (
                    &self.hw_frame,
                    self.hw_frame.width(),
                    self.hw_frame.height(),
                )
            };

        // 检查是否为 NV12 格式
        if final_frame.format() != ffmpeg::format::Pixel::NV12 {
            return Err(DecoderError::UnsupportedFormat(format!(
                "{:?}",
                final_frame.format()
            )));
        }

        // 提取 NV12 数据
        let y_stride = final_frame.stride(0);
        let uv_stride = final_frame.stride(1);

        let y_size = (height as usize) * (y_stride as usize);
        let uv_size = ((height / 2) as usize) * (uv_stride as usize);

        // 复用缓冲区而不是每次分配
        self.y_buffer.resize(y_size, 0);
        self.uv_buffer.resize(uv_size, 0);

        unsafe {
            std::ptr::copy_nonoverlapping(
                final_frame.data(0).as_ptr(),
                self.y_buffer.as_mut_ptr(),
                y_size,
            );
            std::ptr::copy_nonoverlapping(
                final_frame.data(1).as_ptr(),
                self.uv_buffer.as_mut_ptr(),
                uv_size,
            );
        }

        Ok(Some(VideoFrame {
            y_data: Arc::new(self.y_buffer.clone()),
            uv_data: Arc::new(self.uv_buffer.clone()),
            width,
            height,
            y_stride: y_stride as usize,
            uv_stride: uv_stride as usize,
        }))
    }

    /// 获取视频信息
    pub fn video_info(&self) -> VideoInfo {
        VideoInfo {
            width: self.decoder.width(),
            height: self.decoder.height(),
            pixel_format: format!("{:?}", self.decoder.format()),
        }
    }
}

/// 视频信息
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub pixel_format: String,
}

/// get_format 回调函数
unsafe extern "C" fn get_format_callback(
    _ctx: *mut ffi::AVCodecContext,
    pix_fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    if pix_fmts.is_null() {
        return ffi::AVPixelFormat::AV_PIX_FMT_NONE;
    }

    let mut i = 0;
    while *pix_fmts.offset(i) != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
        let fmt = *pix_fmts.offset(i);
        if fmt == ffi::AVPixelFormat::AV_PIX_FMT_QSV
            || fmt == ffi::AVPixelFormat::AV_PIX_FMT_D3D11
            || fmt == ffi::AVPixelFormat::AV_PIX_FMT_NV12
        {
            return fmt;
        }
        i += 1;
    }

    *pix_fmts
}

/// 解码器错误类型
#[derive(Debug)]
pub enum DecoderError {
    /// 初始化失败
    Initialization(String),
    /// 打开流失败
    OpenStream(String),
    /// 没有视频流
    NoVideoStream,
    /// 没有硬件解码器
    NoHardwareDecoder,
    /// 没有解码器
    NoDecoder,
    /// 创建上下文失败
    CreateContext(String),
    /// 创建解码器失败
    CreateDecoder(String),
    /// 硬件设置失败
    HardwareSetup(String),
    /// 发送数据包失败
    SendPacket(String),
    /// 帧传输失败
    TransferFrame(String),
    /// 不支持的格式
    UnsupportedFormat(String),
}

impl std::fmt::Display for DecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initialization(e) => write!(f, "FFmpeg 初始化失败: {}", e),
            Self::OpenStream(e) => write!(f, "打开流失败: {}", e),
            Self::NoVideoStream => write!(f, "找不到视频流"),
            Self::NoHardwareDecoder => write!(f, "找不到硬件解码器"),
            Self::NoDecoder => write!(f, "找不到解码器"),
            Self::CreateContext(e) => write!(f, "创建上下文失败: {}", e),
            Self::CreateDecoder(e) => write!(f, "创建解码器失败: {}", e),
            Self::HardwareSetup(e) => write!(f, "硬件加速设置失败: {}", e),
            Self::SendPacket(e) => write!(f, "发送数据包失败: {}", e),
            Self::TransferFrame(e) => write!(f, "帧传输失败: {}", e),
            Self::UnsupportedFormat(f_) => write!(f, "不支持的像素格式: {}", f_),
        }
    }
}

impl std::error::Error for DecoderError {}
