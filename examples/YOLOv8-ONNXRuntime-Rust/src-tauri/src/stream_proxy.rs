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
//!
//! 注意: Rust 端只做解复用 (demux)，不做解码！
//! 压缩数据包转为 Annex-B 格式后发送给前端 WebCodecs 解码。

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
            loop_file: Some(false),
        }
    }
}

/// 将 AVCC 格式 (4字节长度前缀) 转换为 Annex-B 格式 (00 00 00 01 起始码)
/// WebCodecs 需要 Annex-B 格式
fn convert_avcc_to_annex_b(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() + 32);
    let mut i = 0;

    while i + 4 <= data.len() {
        // 读取 NAL 单元长度 (big-endian u32)
        let nal_len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;

        if nal_len == 0 || i + nal_len > data.len() {
            break;
        }

        // 添加 Annex-B 起始码
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        // 添加 NAL 数据
        result.extend_from_slice(&data[i..i + nal_len]);
        i += nal_len;
    }

    result
}

/// 检查数据是否已经是 Annex-B 格式
fn is_annex_b(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    // 检查是否以 00 00 00 01 或 00 00 01 开头
    (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        || (data[0] == 0 && data[1] == 0 && data[2] == 1)
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

            if let Err(e) = Self::demux_loop(
                &config,
                protocol,
                &running,
                &active_gen,
                generation,
                video_channel,
            ) {
                eprintln!("❌ StreamProxy 错误: {}", e);
            }

            running.store(false, Ordering::SeqCst);
            println!("📴 StreamProxy 线程退出 (Gen: {})", generation);
        });
    }

    /// 解复用主循环 - 只读取压缩数据包，不解码！
    fn demux_loop(
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

        // 获取编解码器信息 (不创建解码器！只获取参数)
        let codec_id = codec_params.id();

        // codec_type: 简单类型名 (h264/hevc/vp9 等)
        // codec_string: WebCodecs 需要的完整 codec string
        let (codec_type, codec_string) = match codec_id {
            ffmpeg::codec::Id::H264 => ("h264", "avc1.640028"), // H.264 High Profile Level 4.0
            ffmpeg::codec::Id::HEVC => ("hevc", "hev1.1.6.L93.B0"), // HEVC Main Profile
            ffmpeg::codec::Id::VP8 => ("vp8", "vp8"),
            ffmpeg::codec::Id::VP9 => ("vp9", "vp09.00.10.08"),
            ffmpeg::codec::Id::AV1 => ("av1", "av01.0.01M.08"),
            _ => ("unknown", "unknown"),
        };

        // 从 codec_params 获取宽高
        let (width, height) = unsafe {
            let par = codec_params.as_ptr();
            ((*par).width as u32, (*par).height as u32)
        };

        println!(
            "📐 视频尺寸: {}x{}, 编码: {} ({})",
            width, height, codec_type, codec_string
        );

        // 获取 extradata (SPS/PPS for H.264, VPS/SPS/PPS for HEVC)
        let extradata = unsafe {
            let par = codec_params.as_ptr();
            if !(*par).extradata.is_null() && (*par).extradata_size > 0 {
                let slice =
                    std::slice::from_raw_parts((*par).extradata, (*par).extradata_size as usize);
                Some(slice.to_vec())
            } else {
                None
            }
        };

        // 发送元数据到前端
        if let Some(ref ch) = video_channel {
            let metadata = json!({
                "type": "metadata",
                "codec_type": codec_type,        // 简单类型: h264, hevc, vp9...
                "video_codec": codec_string,     // WebCodecs codec string
                "width": width,
                "height": height,
                "protocol": protocol.name(),
            });
            let mut bytes = vec![0xFF, 0xFE]; // 元数据标记
            bytes.extend_from_slice(metadata.to_string().as_bytes());
            let _ = ch.send(InvokeResponseBody::Raw(bytes));
        }

        // 如果有 extradata，先发送 (包含 SPS/PPS)
        // 对于 H.264/HEVC 容器格式 (MP4/FLV/RTMP)，extradata 是 AVCC/HVCC 格式
        // 需要转换为 Annex-B 格式
        if let Some(extra) = &extradata {
            if let Some(ref ch) = video_channel {
                let annex_b_extra = parse_extradata_to_annex_b(extra, codec_id);
                if !annex_b_extra.is_empty() {
                    println!("📦 发送 extradata (SPS/PPS): {} bytes", annex_b_extra.len());
                    let _ = ch.send(InvokeResponseBody::Raw(annex_b_extra));
                }
            }
        }

        let mut packet_count = 0u64;
        let loop_file = config.loop_file.unwrap_or(false) && protocol == StreamProtocol::File;

        // 解复用循环 - 只读取压缩包，不解码！
        'demux_loop: loop {
            if !running.load(Ordering::Relaxed) || active_gen.load(Ordering::Relaxed) != generation
            {
                break;
            }

            for (stream, packet) in ictx.packets() {
                if !running.load(Ordering::Relaxed)
                    || active_gen.load(Ordering::Relaxed) != generation
                {
                    break 'demux_loop;
                }

                if stream.index() != video_stream_index {
                    continue;
                }

                // 获取压缩数据
                let data = packet.data().unwrap_or(&[]);
                if data.is_empty() {
                    continue;
                }

                // 转换为 Annex-B 格式 (如果需要)
                let annex_b_data = if is_annex_b(data) {
                    // 已经是 Annex-B (如 RTSP/TS)
                    data.to_vec()
                } else {
                    // AVCC 格式 (如 MP4/FLV/RTMP)，需要转换
                    convert_avcc_to_annex_b(data)
                };

                if annex_b_data.is_empty() {
                    continue;
                }

                // 直接发送压缩数据到前端 WebCodecs
                if let Some(ref ch) = video_channel {
                    let _ = ch.send(InvokeResponseBody::Raw(annex_b_data));
                }

                packet_count += 1;
                if packet_count == 1 {
                    println!("📦 首个视频包已发送 (Gen: {})", generation);
                }
                if packet_count % 300 == 0 {
                    println!("📦 已发送 {} 个视频包 (Gen: {})", packet_count, generation);
                }
            }

            // 文件播放完毕
            if loop_file {
                println!("🔄 文件循环播放");
                if ictx.seek(0, ..).is_err() {
                    break;
                }
            } else {
                break;
            }
        }

        println!("📊 总计发送 {} 个视频包", packet_count);
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

/// 解析 extradata (AVCC/HVCC 格式) 转换为 Annex-B 格式的 SPS/PPS
fn parse_extradata_to_annex_b(extradata: &[u8], codec_id: ffmpeg::codec::Id) -> Vec<u8> {
    match codec_id {
        ffmpeg::codec::Id::H264 => parse_avcc_extradata(extradata),
        ffmpeg::codec::Id::HEVC => parse_hvcc_extradata(extradata),
        _ => Vec::new(),
    }
}

/// 解析 H.264 AVCC extradata -> Annex-B SPS/PPS
fn parse_avcc_extradata(data: &[u8]) -> Vec<u8> {
    if data.len() < 7 {
        return Vec::new();
    }

    // 检查是否已经是 Annex-B 格式
    if is_annex_b(data) {
        return data.to_vec();
    }

    // AVCC 格式:
    // byte 0: version (always 1)
    // byte 1: profile
    // byte 2: profile compatibility
    // byte 3: level
    // byte 4: 6 bits reserved (111111) + 2 bits NAL size length minus 1
    // byte 5: 3 bits reserved (111) + 5 bits number of SPS
    if data[0] != 1 {
        // 可能不是 AVCC 格式，直接返回
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(data.len() + 16);
    let mut offset = 5;

    // 解析 SPS
    let num_sps = (data[offset] & 0x1F) as usize;
    offset += 1;

    for _ in 0..num_sps {
        if offset + 2 > data.len() {
            break;
        }
        let sps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + sps_len > data.len() {
            break;
        }

        // 添加 Annex-B 起始码 + SPS
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&data[offset..offset + sps_len]);
        offset += sps_len;
    }

    // 解析 PPS
    if offset >= data.len() {
        return result;
    }
    let num_pps = data[offset] as usize;
    offset += 1;

    for _ in 0..num_pps {
        if offset + 2 > data.len() {
            break;
        }
        let pps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + pps_len > data.len() {
            break;
        }

        // 添加 Annex-B 起始码 + PPS
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&data[offset..offset + pps_len]);
        offset += pps_len;
    }

    result
}

/// 解析 HEVC HVCC extradata -> Annex-B VPS/SPS/PPS
fn parse_hvcc_extradata(data: &[u8]) -> Vec<u8> {
    if data.len() < 23 {
        return Vec::new();
    }

    // 检查是否已经是 Annex-B 格式
    if is_annex_b(data) {
        return data.to_vec();
    }

    // HVCC 格式检查
    // byte 0: configurationVersion (always 1)
    if data[0] != 1 {
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(data.len() + 32);

    // HVCC header 是 22 bytes，然后是 NAL unit arrays
    let mut offset = 22;

    // numOfArrays
    if offset >= data.len() {
        return result;
    }
    let num_arrays = data[offset] as usize;
    offset += 1;

    for _ in 0..num_arrays {
        if offset + 3 > data.len() {
            break;
        }

        // array_completeness (1 bit) + reserved (1 bit) + NAL_unit_type (6 bits)
        let _nal_type = data[offset] & 0x3F;
        offset += 1;

        // numNalus
        let num_nalus = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        for _ in 0..num_nalus {
            if offset + 2 > data.len() {
                break;
            }

            let nal_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + nal_len > data.len() {
                break;
            }

            // 添加 Annex-B 起始码 + NAL
            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend_from_slice(&data[offset..offset + nal_len]);
            offset += nal_len;
        }
    }

    result
}
