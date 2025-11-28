//! 视频解复用器模块
//!
//! 负责从 RTSP/RTMP/文件源读取视频流，进行解复用和可选的硬件解码。
//! 通过 xbus 发布事件与其他模块通信。
//!
//! ## 工作模式
//!
//! 1. **WebCodecs 模式**: 仅解复用，发布编码数据包供前端 WebCodecs 解码
//! 2. **硬件解码模式**: 使用 QSV/CUDA 硬件解码，发布解码后的帧
//!
//! ## 事件发布
//!
//! - `StreamStartedEvent`: 流开始
//! - `StreamMetadataEvent`: 流元数据（编码格式、分辨率等）
//! - `EncodedPacketEvent`: 编码数据包（WebCodecs 模式）
//! - `DecodedFrameEvent`: 解码帧（硬件解码模式）
//! - `RgbaFrameReadyEvent`: RGBA 帧就绪（通知前端读取共享内存）
//! - `StreamStoppedEvent`: 流停止
//! - `StreamErrorEvent`: 流错误

use crate::video_events::{
    DecodedFrameEvent, EncodedPacketEvent, RgbaFrameReadyEvent, StreamErrorEvent,
    StreamMetadataEvent, StreamStartedEvent, StreamStoppedEvent,
};
use crate::xbus;

use ffmpeg_next as ffmpeg;
use ffmpeg_next::codec::context::Context as CodecContext;
use ffmpeg_next::format::context::Input as FormatContext;
use ffmpeg_next::software::scaling::{context::Context as SwsContext, flag::Flags as SwsFlags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use parking_lot::RwLock;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// RGBA 帧数据
#[derive(Clone)]
pub struct RgbaFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}

/// 帧回调类型
pub type FrameCallback = Arc<dyn Fn(RgbaFrame) + Send + Sync>;

/// 解复用器配置
#[derive(Clone)]
pub struct DemuxerConfig {
    /// 视频源 URL（RTSP/RTMP/文件路径）
    pub url: String,
    /// 是否使用硬件解码
    pub use_hw_decode: bool,
    /// 硬件解码器类型: "qsv", "cuda", "d3d11va"
    pub hw_type: String,
    /// 是否发布 RGBA 帧（否则发布 NV12）
    pub output_rgba: bool,
    /// 共享内存名称前缀
    pub shm_name_prefix: String,
    /// 连接超时（秒）
    pub connect_timeout: u32,
    /// 读取超时（秒）
    pub read_timeout: u32,
    /// 帧回调函数
    pub frame_callback: Option<FrameCallback>,
}

impl Default for DemuxerConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            use_hw_decode: true,
            hw_type: "qsv".to_string(),
            output_rgba: true,
            shm_name_prefix: "yolo_frame".to_string(),
            connect_timeout: 10,
            read_timeout: 5,
            frame_callback: None,
        }
    }
}

/// 解复用器状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemuxerState {
    Idle,
    Connecting,
    Running,
    Stopping,
    Stopped,
    Error,
}

/// 视频流元数据
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub extradata: Vec<u8>,
}

/// 解复用器
pub struct Demuxer {
    config: DemuxerConfig,
    running: Arc<AtomicBool>,
    frame_count: Arc<AtomicU64>,
    packet_count: Arc<AtomicU64>,
}

impl Demuxer {
    /// 创建新的解复用器
    pub fn new(config: DemuxerConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            frame_count: Arc::new(AtomicU64::new(0)),
            packet_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 启动解复用器（在新线程中运行）
    pub fn start(&self) -> Result<(), String> {
        if self.running.load(Ordering::SeqCst) {
            return Err("Demuxer already running".to_string());
        }

        self.running.store(true, Ordering::SeqCst);
        self.frame_count.store(0, Ordering::SeqCst);
        self.packet_count.store(0, Ordering::SeqCst);

        let config = self.config.clone();
        let running = self.running.clone();
        let frame_count = self.frame_count.clone();
        let packet_count = self.packet_count.clone();

        thread::spawn(move || {
            // 发布流开始事件
            xbus::post(StreamStartedEvent {
                url: config.url.clone(),
            });

            if let Err(e) = run_demuxer_loop(config, running.clone(), frame_count, packet_count) {
                eprintln!("[Demuxer] Error: {}", e);
                xbus::post(StreamErrorEvent { message: e.clone() });
            }

            running.store(false, Ordering::SeqCst);
            xbus::post(StreamStoppedEvent {
                reason: "Demuxer stopped".to_string(),
            });
        });

        Ok(())
    }

    /// 停止解复用器
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// 检查是否正在运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// 获取已处理的帧数
    pub fn frame_count(&self) -> u64 {
        self.frame_count.load(Ordering::SeqCst)
    }

    /// 获取已处理的包数
    pub fn packet_count(&self) -> u64 {
        self.packet_count.load(Ordering::SeqCst)
    }
}

/// 解复用器主循环
fn run_demuxer_loop(
    config: DemuxerConfig,
    running: Arc<AtomicBool>,
    frame_count: Arc<AtomicU64>,
    packet_count: Arc<AtomicU64>,
) -> Result<(), String> {
    // 初始化 FFmpeg
    ffmpeg::init().map_err(|e| format!("FFmpeg init failed: {}", e))?;

    // 设置 FFmpeg 选项
    let mut opts = ffmpeg::Dictionary::new();
    opts.set("rtsp_transport", "tcp");
    opts.set(
        "stimeout",
        &(config.connect_timeout * 1_000_000).to_string(),
    );
    opts.set("timeout", &(config.read_timeout * 1_000_000).to_string());
    opts.set("buffer_size", "16777216"); //16777216
    opts.set("max_delay", "500000");
    opts.set("fflags", "nobuffer");
    opts.set("flags", "low_delay");

    // 打开输入流
    let mut ictx = ffmpeg::format::input_with_dictionary(&config.url, opts)
        .map_err(|e| format!("Failed to open {}: {}", config.url, e))?;

    // 查找视频流
    let video_stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or("No video stream found")?
        .index();

    let video_stream = ictx
        .stream(video_stream_index)
        .ok_or("Video stream not found")?;

    // 获取编解码器参数
    let codec_params = video_stream.parameters();
    let codec_id = unsafe { (*codec_params.as_ptr()).codec_id };

    let codec_name = match codec_id {
        ffmpeg::ffi::AVCodecID::AV_CODEC_ID_H264 => "h264",
        ffmpeg::ffi::AVCodecID::AV_CODEC_ID_HEVC => "hevc",
        _ => "unknown",
    };

    let width = unsafe { (*codec_params.as_ptr()).width as u32 };
    let height = unsafe { (*codec_params.as_ptr()).height as u32 };

    // 计算帧率
    let fps = {
        let rational = video_stream.avg_frame_rate();
        if rational.denominator() != 0 {
            rational.numerator() as f64 / rational.denominator() as f64
        } else {
            30.0
        }
    };

    // 获取 extradata（SPS/PPS）
    let extradata = unsafe {
        let ptr = (*codec_params.as_ptr()).extradata;
        let size = (*codec_params.as_ptr()).extradata_size as usize;
        if !ptr.is_null() && size > 0 {
            std::slice::from_raw_parts(ptr, size).to_vec()
        } else {
            Vec::new()
        }
    };

    println!(
        "[Demuxer] Stream opened: {} {}x{} @ {:.2} fps, extradata: {} bytes",
        codec_name,
        width,
        height,
        fps,
        extradata.len()
    );

    // 发布流元数据事件
    xbus::post(StreamMetadataEvent {
        codec: codec_name.to_string(),
        width,
        height,
        fps,
    });

    // 根据配置选择工作模式
    if config.use_hw_decode {
        run_hw_decode_loop(
            &mut ictx,
            video_stream_index,
            &config,
            running,
            frame_count,
            width,
            height,
        )
    } else {
        run_demux_only_loop(
            &mut ictx,
            video_stream_index,
            &extradata,
            running,
            packet_count,
        )
    }
}

/// 仅解复用模式（发布编码数据包）
fn run_demux_only_loop(
    ictx: &mut FormatContext,
    video_stream_index: usize,
    extradata: &[u8],
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
) -> Result<(), String> {
    let mut packet_id: u64 = 0;

    for (stream_idx, packet) in ictx.packets() {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        if stream_idx.index() != video_stream_index {
            continue;
        }

        let data = packet.data().map(|d| d.to_vec()).unwrap_or_default();
        if data.is_empty() {
            continue;
        }

        let is_keyframe = packet.is_key();
        let pts = packet.pts().unwrap_or(0);
        let dts = packet.dts().unwrap_or(0);

        // 转换为 Annex-B 格式
        let annexb_data = if is_keyframe && !extradata.is_empty() {
            // 关键帧前添加 extradata
            let mut full_data = extradata.to_vec();
            full_data.extend(avcc_to_annexb(&data));
            full_data
        } else {
            avcc_to_annexb(&data)
        };

        packet_id += 1;
        packet_count.fetch_add(1, Ordering::SeqCst);

        // 发布编码数据包事件
        xbus::post(EncodedPacketEvent {
            packet_id,
            data: Arc::new(annexb_data),
            is_keyframe,
            pts,
            dts,
        });
    }

    Ok(())
}

/// 硬件解码模式
fn run_hw_decode_loop(
    ictx: &mut FormatContext,
    video_stream_index: usize,
    config: &DemuxerConfig,
    running: Arc<AtomicBool>,
    frame_count: Arc<AtomicU64>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let video_stream = ictx
        .stream(video_stream_index)
        .ok_or("Video stream not found")?;

    // 获取编解码器 ID
    let codec_id = video_stream.parameters().id();

    // 根据硬件类型选择对应的硬解码器
    let hw_decoder_name = if config.use_hw_decode {
        match (codec_id, config.hw_type.as_str()) {
            (ffmpeg::codec::Id::H264, "qsv") => Some("h264_qsv"),
            (ffmpeg::codec::Id::HEVC, "qsv") => Some("hevc_qsv"),
            (ffmpeg::codec::Id::H264, "cuda") => Some("h264_cuvid"),
            (ffmpeg::codec::Id::HEVC, "cuda") => Some("hevc_cuvid"),
            _ => None,
        }
    } else {
        None
    };

    // 创建解码器
    let decoder_codec = if let Some(hw_name) = hw_decoder_name {
        ffmpeg::decoder::find_by_name(hw_name)
            .ok_or_else(|| format!("Hardware decoder '{}' not found", hw_name))?
    } else {
        ffmpeg::decoder::find(codec_id).ok_or("Decoder not found")?
    };

    println!("[Demuxer] Using decoder: {}", decoder_codec.name());

    let mut decoder_ctx = CodecContext::new_with_codec(decoder_codec);

    // QSV 解码器需要设置硬件设备上下文
    if config.use_hw_decode && config.hw_type == "qsv" {
        setup_hw_device_ctx(&mut decoder_ctx, &config.hw_type)?;
    }

    decoder_ctx
        .set_parameters(video_stream.parameters())
        .map_err(|e| format!("Set parameters failed: {}", e))?;

    let mut decoder = decoder_ctx
        .decoder()
        .video()
        .map_err(|e| format!("Open decoder failed: {}", e))?;

    // 创建颜色空间转换器（如果需要 RGBA 输出）
    // 注意：QSV 解码器输出的是 NV12 格式
    let hw_pix_fmt = if config.use_hw_decode && config.hw_type == "qsv" {
        ffmpeg::format::Pixel::NV12 // QSV 解码输出 NV12
    } else {
        ffmpeg::format::Pixel::NV12
    };

    let mut sws_ctx: Option<SwsContext> = None;
    if config.output_rgba {
        sws_ctx = Some(
            SwsContext::get(
                hw_pix_fmt,
                width,
                height,
                ffmpeg::format::Pixel::RGBA,
                width,
                height,
                SwsFlags::BILINEAR,
            )
            .map_err(|e| format!("Create scaler failed: {}", e))?,
        );
    }

    let mut frame_id: u64 = 0;
    let mut decoded_frame = VideoFrame::empty();
    let mut sw_frame = VideoFrame::empty(); // 用于 GPU->CPU 传输
    let mut rgba_frame = VideoFrame::empty();

    let shm_name = format!("{}_video", config.shm_name_prefix);
    let start_time = Instant::now();

    for (stream_idx, packet) in ictx.packets() {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        if stream_idx.index() != video_stream_index {
            continue;
        }

        // 发送数据包到解码器
        if let Err(e) = decoder.send_packet(&packet) {
            eprintln!("[Demuxer] Send packet failed: {}", e);
            continue;
        }

        // 接收解码帧
        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            frame_id += 1;
            frame_count.fetch_add(1, Ordering::SeqCst);

            if frame_id % 100 == 1 {
                println!(
                    "[Demuxer] Decoded frame #{}, format: {:?}",
                    frame_id,
                    decoded_frame.format()
                );
            }

            let timestamp = start_time.elapsed().as_millis() as u64;

            // 检查是否是硬件帧，需要传输到 CPU
            let cpu_frame = if is_hw_frame(&decoded_frame) {
                if frame_id == 1 {
                    println!("[Demuxer] Hardware frame detected, transferring to CPU...");
                }
                // GPU -> CPU 传输
                if let Err(e) = transfer_hw_frame(&decoded_frame, &mut sw_frame) {
                    eprintln!("[Demuxer] GPU->CPU transfer failed: {}", e);
                    continue;
                }
                if frame_id == 1 {
                    println!(
                        "[Demuxer] Transfer success, sw_frame format: {:?}",
                        sw_frame.format()
                    );
                }
                &sw_frame
            } else {
                if frame_id == 1 {
                    println!("[Demuxer] Software frame, no transfer needed");
                }
                &decoded_frame
            };

            if config.output_rgba {
                // 转换为 RGBA
                if let Some(ref mut scaler) = sws_ctx {
                    if scaler.run(cpu_frame, &mut rgba_frame).is_ok() {
                        if frame_id == 1 {
                            println!(
                                "[Demuxer] RGBA conversion success, size: {}x{}",
                                width, height
                            );
                        }
                        // 提取 RGBA 数据
                        let rgba_data = rgba_frame.data(0);
                        let stride = rgba_frame.stride(0) as u32;

                        // 如果 stride == width * 4，直接复制；否则需要按行复制
                        let frame_data = if stride == width * 4 {
                            rgba_data[..(width * height * 4) as usize].to_vec()
                        } else {
                            // 按行复制，去除填充
                            let mut data = Vec::with_capacity((width * height * 4) as usize);
                            for y in 0..height {
                                let start = (y * stride) as usize;
                                let end = start + (width * 4) as usize;
                                data.extend_from_slice(&rgba_data[start..end]);
                            }
                            data
                        };

                        // 调用帧回调
                        if let Some(ref callback) = config.frame_callback {
                            if frame_id == 1 {
                                println!(
                                    "[Demuxer] Calling frame callback with {} bytes",
                                    frame_data.len()
                                );
                            }
                            callback(RgbaFrame {
                                data: frame_data,
                                width,
                                height,
                                timestamp,
                            });
                        } else if frame_id == 1 {
                            println!("[Demuxer] No frame callback configured!");
                        }

                        // 发布 RGBA 帧就绪事件
                        xbus::post(RgbaFrameReadyEvent {
                            frame_id,
                            width,
                            height,
                            timestamp,
                            shm_name: shm_name.clone(),
                            offset: 0,
                            size: (width * height * 4) as usize,
                        });
                    }
                }
            } else {
                // 发布 NV12 解码帧事件
                let y_stride = cpu_frame.stride(0) as u32;
                let uv_stride = cpu_frame.stride(1) as u32;

                xbus::post(DecodedFrameEvent {
                    frame_id,
                    width,
                    height,
                    y_stride,
                    uv_stride,
                    timestamp,
                    buffer_index: (frame_id % 2) as u32,
                    shm_name: shm_name.clone(),
                });
            }
        }
    }

    Ok(())
}

/// 设置硬件设备上下文
fn setup_hw_device_ctx(ctx: &mut CodecContext, hw_type: &str) -> Result<(), String> {
    let hw_device_type = match hw_type {
        "qsv" => ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV,
        "cuda" => ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
        "d3d11va" => ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        "vaapi" => ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
        _ => return Err(format!("Unsupported hw type: {}", hw_type)),
    };

    unsafe {
        let mut hw_device_ctx: *mut ffmpeg::ffi::AVBufferRef = std::ptr::null_mut();
        let ret = ffmpeg::ffi::av_hwdevice_ctx_create(
            &mut hw_device_ctx,
            hw_device_type,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );

        if ret < 0 {
            return Err(format!(
                "Failed to create {} hw device context: {}",
                hw_type, ret
            ));
        }

        (*ctx.as_mut_ptr()).hw_device_ctx = ffmpeg::ffi::av_buffer_ref(hw_device_ctx);
        ffmpeg::ffi::av_buffer_unref(&mut hw_device_ctx);
    }

    println!("[Demuxer] Hardware device context created: {}", hw_type);
    Ok(())
}

/// 检查帧是否是硬件帧
fn is_hw_frame(frame: &VideoFrame) -> bool {
    unsafe {
        let format = (*frame.as_ptr()).format;
        // 硬件像素格式
        format == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_QSV as i32
            || format == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA as i32
            || format == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_D3D11 as i32
            || format == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_VAAPI as i32
    }
}

/// 将硬件帧传输到 CPU 内存
fn transfer_hw_frame(hw_frame: &VideoFrame, sw_frame: &mut VideoFrame) -> Result<(), String> {
    unsafe {
        let ret =
            ffmpeg::ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), hw_frame.as_ptr(), 0);
        if ret < 0 {
            return Err(format!("av_hwframe_transfer_data failed: {}", ret));
        }
    }
    Ok(())
}

/// AVCC 格式转 Annex-B 格式
fn avcc_to_annexb(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() + 32);
    let mut offset = 0;

    while offset + 4 <= data.len() {
        // AVCC 使用 4 字节长度前缀
        let nalu_size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        offset += 4;

        if offset + nalu_size > data.len() {
            break;
        }

        // Annex-B 使用 0x00000001 起始码
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&data[offset..offset + nalu_size]);

        offset += nalu_size;
    }

    if result.is_empty() {
        // 可能已经是 Annex-B 格式
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(data);
    }

    result
}

/// 解复用器句柄（用于 Tauri State 管理）
pub struct DemuxerHandle {
    demuxer: Option<Demuxer>,
}

impl DemuxerHandle {
    pub fn new() -> Self {
        Self { demuxer: None }
    }

    pub fn start(&mut self, config: DemuxerConfig) -> Result<(), String> {
        if self.demuxer.is_some() {
            self.stop();
        }

        let demuxer = Demuxer::new(config);
        demuxer.start()?;
        self.demuxer = Some(demuxer);
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(demuxer) = self.demuxer.take() {
            demuxer.stop();
        }
    }

    pub fn is_running(&self) -> bool {
        self.demuxer
            .as_ref()
            .map(|d| d.is_running())
            .unwrap_or(false)
    }

    pub fn frame_count(&self) -> u64 {
        self.demuxer.as_ref().map(|d| d.frame_count()).unwrap_or(0)
    }

    pub fn packet_count(&self) -> u64 {
        self.demuxer.as_ref().map(|d| d.packet_count()).unwrap_or(0)
    }
}

impl Default for DemuxerHandle {
    fn default() -> Self {
        Self::new()
    }
}
