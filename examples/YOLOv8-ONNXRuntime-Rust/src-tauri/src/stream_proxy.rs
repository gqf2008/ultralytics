//! 通用流媒体代理 - 使用 ffmpeg-next 支持多种协议
//!
//! 支持的协议:
//! - RTSP: rtsp://
//! - RTMP: rtmp://
//! - HTTP-FLV: http://.../xxx.flv
//! - HLS: http://.../xxx.m3u8
//! - HTTP/HTTPS 流: http://, https://
//! - 本地文件: file:// 或直接路径
//! - 摄像头设备: /dev/video0, video=xxx (Windows)

use ffmpeg_next as ffmpeg;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use tauri::ipc::{Channel, InvokeResponseBody};

/// 流媒体协议类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamProtocol {
    Rtsp,
    Rtmp,
    HttpFlv,
    Hls,
    Http,
    File,
    Device,
    Unknown,
}

impl StreamProtocol {
    /// 从 URL 自动检测协议类型
    pub fn detect(url: &str) -> Self {
        let url_lower = url.to_lowercase();

        if url_lower.starts_with("rtsp://") {
            StreamProtocol::Rtsp
        } else if url_lower.starts_with("rtmp://") || url_lower.starts_with("rtmps://") {
            StreamProtocol::Rtmp
        } else if url_lower.ends_with(".flv")
            && (url_lower.starts_with("http://") || url_lower.starts_with("https://"))
        {
            StreamProtocol::HttpFlv
        } else if url_lower.ends_with(".m3u8")
            || url_lower.contains("/hls/")
            || url_lower.contains(".m3u8?")
        {
            StreamProtocol::Hls
        } else if url_lower.starts_with("http://") || url_lower.starts_with("https://") {
            StreamProtocol::Http
        } else if url_lower.starts_with("file://")
            || url_lower.ends_with(".mp4")
            || url_lower.ends_with(".mkv")
            || url_lower.ends_with(".avi")
            || url_lower.ends_with(".mov")
            || url_lower.ends_with(".ts")
        {
            StreamProtocol::File
        } else if url_lower.starts_with("/dev/video")
            || url_lower.starts_with("video=")
            || url_lower.contains("dshow")
        {
            StreamProtocol::Device
        } else {
            StreamProtocol::Unknown
        }
    }

    /// 获取协议名称
    pub fn name(&self) -> &'static str {
        match self {
            StreamProtocol::Rtsp => "RTSP",
            StreamProtocol::Rtmp => "RTMP",
            StreamProtocol::HttpFlv => "HTTP-FLV",
            StreamProtocol::Hls => "HLS",
            StreamProtocol::Http => "HTTP",
            StreamProtocol::File => "File",
            StreamProtocol::Device => "Device",
            StreamProtocol::Unknown => "Unknown",
        }
    }
}

/// 流媒体配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// 输入 URL
    pub url: String,
    /// 强制指定协议（可选，不指定则自动检测）
    pub protocol: Option<StreamProtocol>,
    /// 传输协议（仅 RTSP: tcp/udp）
    pub transport: Option<String>,
    /// 超时时间（毫秒）
    pub timeout_ms: Option<u32>,
    /// 缓冲区大小（字节）
    pub buffer_size: Option<u32>,
    /// 是否启用硬件解码
    pub hw_decode: Option<bool>,
    /// 硬件解码器名称（如 h264_qsv, hevc_cuvid）
    pub hw_decoder: Option<String>,
    /// 是否循环播放（仅文件）
    pub loop_file: Option<bool>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            protocol: None,
            transport: Some("tcp".to_string()),
            timeout_ms: Some(5000),
            buffer_size: Some(1048576), // 1MB
            hw_decode: Some(false),
            hw_decoder: None,
            loop_file: Some(false),
        }
    }
}

/// 流媒体代理
pub struct StreamProxy {
    running: Arc<AtomicBool>,
    active_generation: Arc<AtomicUsize>,
}

impl StreamProxy {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            active_generation: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// 使用简单 URL 启动（兼容旧接口）
    pub fn start(
        &self,
        url: String,
        video_channel: Option<Channel>,
        audio_channel: Option<Channel>,
    ) {
        let config = StreamConfig {
            url,
            ..Default::default()
        };
        self.start_with_config(config, video_channel, audio_channel);
    }

    /// 使用完整配置启动
    pub fn start_with_config(
        &self,
        config: StreamConfig,
        video_channel: Option<Channel>,
        _audio_channel: Option<Channel>,
    ) {
        // 先停止旧流
        self.running.store(false, Ordering::SeqCst);

        let running = self.running.clone();
        let generation = self.active_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let active_gen = self.active_generation.clone();

        // 检测协议
        let protocol = config
            .protocol
            .unwrap_or_else(|| StreamProtocol::detect(&config.url));
        println!(
            "🎬 StreamProxy: {} 协议检测为 {} (Gen: {})",
            config.url,
            protocol.name(),
            generation
        );

        thread::spawn(move || {
            // 短暂延迟确保旧流退出
            thread::sleep(std::time::Duration::from_millis(100));
            running.store(true, Ordering::SeqCst);

            if let Err(e) = Self::decode_loop(
                &config,
                protocol,
                &running,
                &active_gen,
                generation,
                video_channel,
            ) {
                eprintln!("❌ StreamProxy 解码错误: {}", e);
            }

            running.store(false, Ordering::SeqCst);
            println!("📴 StreamProxy 解码线程退出 (Gen: {})", generation);
        });
    }

    /// 解码主循环
    fn decode_loop(
        config: &StreamConfig,
        protocol: StreamProtocol,
        running: &Arc<AtomicBool>,
        active_gen: &Arc<AtomicUsize>,
        generation: usize,
        video_channel: Option<Channel>,
    ) -> Result<(), String> {
        // 初始化 FFmpeg
        ffmpeg::init().map_err(|e| format!("FFmpeg 初始化失败: {}", e))?;

        // 构建输入选项
        let mut opts = ffmpeg::Dictionary::new();

        // 根据协议设置不同参数
        match protocol {
            StreamProtocol::Rtsp => {
                let transport = config.transport.as_deref().unwrap_or("tcp");
                opts.set("rtsp_transport", transport);
                opts.set(
                    "stimeout",
                    &format!("{}", config.timeout_ms.unwrap_or(5000) * 1000),
                ); // 微秒
                opts.set(
                    "buffer_size",
                    &format!("{}", config.buffer_size.unwrap_or(1048576)),
                );
                opts.set("max_delay", "500000");
                opts.set("reorder_queue_size", "0");
            }
            StreamProtocol::Rtmp => {
                opts.set("live", "1");
                opts.set(
                    "buffer_size",
                    &format!("{}", config.buffer_size.unwrap_or(1048576)),
                );
                opts.set(
                    "timeout",
                    &format!("{}", config.timeout_ms.unwrap_or(5000) * 1000),
                );
            }
            StreamProtocol::HttpFlv | StreamProtocol::Http => {
                opts.set(
                    "timeout",
                    &format!("{}", config.timeout_ms.unwrap_or(5000) * 1000),
                );
                opts.set("reconnect", "1");
                opts.set("reconnect_streamed", "1");
                opts.set("reconnect_delay_max", "5");
            }
            StreamProtocol::Hls => {
                opts.set("allowed_extensions", "ALL");
                opts.set(
                    "timeout",
                    &format!("{}", config.timeout_ms.unwrap_or(10000) * 1000),
                );
            }
            StreamProtocol::File => {
                // 文件不需要特殊选项
            }
            StreamProtocol::Device => {
                // Windows DirectShow 或 Linux V4L2
                #[cfg(windows)]
                opts.set("video_size", "1920x1080");
                #[cfg(unix)]
                opts.set("input_format", "mjpeg");
            }
            StreamProtocol::Unknown => {
                // 使用默认设置
            }
        }

        // 打开输入
        println!("📡 正在连接: {}", config.url);
        let mut ictx = ffmpeg::format::input_with_dictionary(&config.url, opts)
            .map_err(|e| format!("打开流失败: {} ({})", e, config.url))?;

        println!("✅ 连接成功: {}", config.url);

        // 打印流信息
        for stream in ictx.streams() {
            println!(
                "  📺 Stream #{}: {:?}",
                stream.index(),
                stream.parameters().medium()
            );
        }

        // 查找视频流
        let video_stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("找不到视频流")?;
        let video_stream_index = video_stream.index();
        let codec_params = video_stream.parameters();

        println!("🎬 视频流索引: {}", video_stream_index);

        // 创建解码器
        let codec_id = codec_params.id();
        let codec_name = match codec_id {
            ffmpeg::codec::Id::H264 => "h264",
            ffmpeg::codec::Id::HEVC => "hevc",
            ffmpeg::codec::Id::VP8 => "vp8",
            ffmpeg::codec::Id::VP9 => "vp9",
            ffmpeg::codec::Id::AV1 => "av1",
            _ => "unknown",
        };

        // 尝试硬件解码器
        let decoder = if config.hw_decode.unwrap_or(false) {
            let hw_name = config
                .hw_decoder
                .as_deref()
                .unwrap_or_else(|| match codec_id {
                    ffmpeg::codec::Id::H264 => "h264_qsv",
                    ffmpeg::codec::Id::HEVC => "hevc_qsv",
                    _ => "",
                });

            if !hw_name.is_empty() {
                match ffmpeg::decoder::find_by_name(hw_name) {
                    Some(dec) => {
                        println!("✅ 使用硬件解码器: {}", hw_name);
                        Some(dec)
                    }
                    None => {
                        println!("⚠️ 硬件解码器 {} 不可用，回退到软件解码", hw_name);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        let decoder = decoder
            .or_else(|| ffmpeg::decoder::find(codec_id))
            .ok_or(format!("找不到解码器: {:?}", codec_id))?;

        println!("🔧 使用解码器: {}", decoder.name());

        let context = ffmpeg::codec::context::Context::from_parameters(codec_params)
            .map_err(|e| format!("创建解码上下文失败: {}", e))?;
        let mut video_decoder = context
            .decoder()
            .video()
            .map_err(|e| format!("创建视频解码器失败: {}", e))?;

        let width = video_decoder.width();
        let height = video_decoder.height();
        println!("📐 视频尺寸: {}x{}", width, height);

        // 发送元数据到前端
        if let Some(ref ch) = video_channel {
            let metadata = json!({
                "type": "metadata",
                "video_codec": codec_name,
                "width": width,
                "height": height,
                "protocol": protocol.name(),
            });
            let mut bytes = vec![0xFF, 0xFE]; // 元数据标记
            bytes.extend_from_slice(metadata.to_string().as_bytes());
            let _ = ch.send(InvokeResponseBody::Raw(bytes));
        }

        // 创建色彩空间转换器（输出 NV12 给 WebCodecs）
        let mut scaler = ffmpeg::software::scaling::Context::get(
            video_decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::NV12,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| format!("创建缩放器失败: {}", e))?;

        let mut decoded_frame = ffmpeg::frame::Video::empty();
        let mut nv12_frame = ffmpeg::frame::Video::empty();
        let mut frame_count = 0u64;
        let loop_file = config.loop_file.unwrap_or(false) && protocol == StreamProtocol::File;

        // 解码循环
        'decode_loop: loop {
            if !running.load(Ordering::Relaxed) || active_gen.load(Ordering::Relaxed) != generation
            {
                break;
            }

            for (stream, packet) in ictx.packets() {
                if !running.load(Ordering::Relaxed)
                    || active_gen.load(Ordering::Relaxed) != generation
                {
                    break 'decode_loop;
                }

                if stream.index() != video_stream_index {
                    continue;
                }

                if let Err(e) = video_decoder.send_packet(&packet) {
                    eprintln!("⚠️ 发送数据包失败: {}", e);
                    continue;
                }

                while video_decoder.receive_frame(&mut decoded_frame).is_ok() {
                    // 转换为 NV12
                    if let Err(e) = scaler.run(&decoded_frame, &mut nv12_frame) {
                        eprintln!("⚠️ 色彩转换失败: {}", e);
                        continue;
                    }

                    // 提取 NV12 数据
                    let y_data = nv12_frame.data(0);
                    let uv_data = nv12_frame.data(1);
                    let y_stride = nv12_frame.stride(0);
                    let uv_stride = nv12_frame.stride(1);

                    // 打包 NV12 数据 (紧凑格式)
                    let mut nv12_packed = Vec::with_capacity((width * height * 3 / 2) as usize);

                    // Y 平面
                    for row in 0..height as usize {
                        let start = row * y_stride;
                        let end = start + width as usize;
                        nv12_packed.extend_from_slice(&y_data[start..end]);
                    }

                    // UV 平面
                    for row in 0..(height / 2) as usize {
                        let start = row * uv_stride;
                        let end = start + width as usize;
                        nv12_packed.extend_from_slice(&uv_data[start..end]);
                    }

                    // 发送到前端
                    if let Some(ref ch) = video_channel {
                        let _ = ch.send(InvokeResponseBody::Raw(nv12_packed));
                    }

                    frame_count += 1;
                    if frame_count == 1 {
                        println!("🎨 首帧解码成功 (Gen: {})", generation);
                    }
                    if frame_count % 300 == 0 {
                        println!("📺 已解码 {} 帧 (Gen: {})", frame_count, generation);
                    }
                }
            }

            // 文件播放完毕
            if loop_file {
                println!("🔄 文件循环播放");
                // 重新定位到开头
                if ictx.seek(0, ..).is_err() {
                    break;
                }
            } else {
                break;
            }
        }

        println!("📊 总计解码 {} 帧", frame_count);
        Ok(())
    }

    /// 停止流
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// 检查是否正在运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Default for StreamProxy {
    fn default() -> Self {
        Self::new()
    }
}
