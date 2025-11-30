//! 统一流代理模块
//!
//! 根据 URL 协议自动选择最佳后端：
//! - RTSP: retina (纯 Rust，极快)
//! - HTTP-FLV: 纯 Rust FLV 解析器 (极快)
//! - MP4/RTMP/其他: FFmpeg (通用，已优化)

use crate::flv_demuxer::{avcc_to_annexb, FlvDemuxer, VideoCodec};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tauri::ipc::{Channel, InvokeResponseBody};

// ==================== 公共类型 ====================

/// 流元数据
#[derive(Clone, Debug, Serialize)]
pub struct StreamMetadata {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub extradata: Vec<u8>,
}

/// 完整的流媒体信息 (显示在 UI 面板上)
#[derive(Clone, Debug, Default, Serialize)]
pub struct StreamInfo {
    // 流基本信息
    pub protocol: String, // rtsp, http-flv, rtmp, file
    pub backend: String,  // retina, flv_parser, ffmpeg
    pub url: String,

    // 视频信息
    pub video_codec: String,      // H.264, HEVC, VP9 等
    pub video_codec_full: String, // avc1.640028 完整编码字符串
    pub video_width: u32,
    pub video_height: u32,
    pub video_fps: f64,
    pub video_bitrate: u64,    // bps
    pub video_profile: String, // High, Main, Baseline
    pub video_level: String,   // 4.0, 5.1 等

    // 音频信息
    pub audio_codec: String,       // AAC, MP3, OPUS, PCM_ALAW 等
    pub audio_sample_rate: u32,    // 44100, 48000 等
    pub audio_channels: u8,        // 1=单声道, 2=立体声
    pub audio_bits_per_sample: u8, // 16, 24, 32
    pub audio_bitrate: u64,        // bps

    // 统计信息 (实时更新)
    pub total_packets: u64,
    pub total_bytes: u64,
    pub keyframes: u64,
    pub dropped_frames: u64,
    pub start_time: u64, // Unix 时间戳毫秒
}

/// 编码数据包
#[derive(Clone, Debug)]
pub struct EncodedPacket {
    pub data: Vec<u8>,
    pub is_keyframe: bool,
    pub pts: i64,
    pub dts: i64,
    pub packet_id: u64,
}

/// 流配置
#[derive(Clone, Debug, Deserialize)]
pub struct StreamConfig {
    pub url: String,
}

/// 流代理 (自动选择后端)
pub struct StreamProxy {
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
}

impl StreamProxy {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            packet_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn packet_count(&self) -> u64 {
        self.packet_count.load(Ordering::Relaxed)
    }

    /// 启动流 (自动选择后端)
    pub fn start(&self, config: StreamConfig, on_data: Channel<InvokeResponseBody>) {
        // 先停止旧流
        self.running.store(false, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(100));

        self.running.store(true, Ordering::SeqCst);
        self.packet_count.store(0, Ordering::SeqCst);

        let running = self.running.clone();
        let packet_count = self.packet_count.clone();
        let url = config.url.clone();

        // 根据 URL 选择后端
        let url_lower = url.to_lowercase();
        let use_retina = url_lower.starts_with("rtsp://") || url_lower.starts_with("rtsps://");
        let use_flv =
            url_lower.contains(".flv") || url_lower.contains("flv?") || url_lower.contains("/flv/");

        if use_retina {
            println!("🚀 使用 retina 后端 (RTSP)");
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime");

                rt.block_on(async move {
                    if let Err(e) =
                        run_retina_stream(url, running, packet_count, on_data.clone()).await
                    {
                        let error_msg = format!("RTSP 连接失败: {}", e);
                        eprintln!("❌ {}", error_msg);
                        send_error(&on_data, &error_msg);
                    }
                });
            });
        } else if use_flv {
            println!("🚀 使用纯 Rust FLV 后端 (HTTP-FLV)");
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime");

                rt.block_on(async move {
                    if let Err(e) =
                        run_flv_stream(url, running, packet_count, on_data.clone()).await
                    {
                        let error_msg = format!("FLV 连接失败: {}", e);
                        eprintln!("❌ {}", error_msg);
                        send_error(&on_data, &error_msg);
                    }
                });
            });
        } else {
            println!("🚀 使用 FFmpeg 后端 (MP4/RTMP/其他)");
            std::thread::spawn(move || {
                if let Err(e) = run_ffmpeg_stream(url, running, packet_count, on_data.clone()) {
                    let error_msg = format!("FFmpeg 连接失败: {}", e);
                    eprintln!("❌ {}", error_msg);
                    send_error(&on_data, &error_msg);
                }
            });
        }
    }

    /// 停止流
    pub fn stop(&self) {
        println!("⏹ 停止流");
        self.running.store(false, Ordering::SeqCst);
    }
}

// ==================== retina 后端 (RTSP) ====================

use futures_util::StreamExt;
use retina::client::{Credentials, PlayOptions, Session, SessionOptions, SetupOptions, Transport};
use retina::codec::CodecItem;
use url::Url;

async fn run_retina_stream(
    url: String,
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
    on_data: Channel<InvokeResponseBody>,
) -> Result<(), String> {
    println!("🔌 [retina] 连接: {}", url);

    // 解析 URL
    let mut parsed_url = Url::parse(&url).map_err(|e| format!("URL 解析失败: {}", e))?;

    // 提取认证信息
    let creds = if !parsed_url.username().is_empty() {
        let username = parsed_url.username().to_string();
        let password = parsed_url.password().unwrap_or("").to_string();
        let _ = parsed_url.set_username("");
        let _ = parsed_url.set_password(None);
        Some(Credentials { username, password })
    } else {
        None
    };

    // 创建会话
    let mut session_opts = SessionOptions::default();
    if let Some(c) = creds {
        session_opts = session_opts.creds(Some(c));
    }

    let setup_opts = SetupOptions::default().transport(Transport::Tcp(Default::default()));

    let mut session = Session::describe(parsed_url.clone(), session_opts)
        .await
        .map_err(|e| format!("DESCRIBE 失败: {}", e))?;

    // 查找视频流和音频流
    let mut video_idx = None;
    let mut codec_str = "unknown".to_string();
    let mut video_codec_name = "Unknown".to_string();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut extradata = Vec::new();
    let mut fps = 25.0f64;

    // 音频信息
    let mut audio_codec = String::new();
    let mut audio_sample_rate = 0u32;
    let mut audio_channels = 0u8;

    for (idx, stream) in session.streams().iter().enumerate() {
        if stream.media() == "video" && video_idx.is_none() {
            video_idx = Some(idx);

            let encoding = stream.encoding_name().to_uppercase();
            video_codec_name = encoding.clone();
            codec_str = match encoding.as_str() {
                "H264" => "avc1.640028".to_string(),
                "H265" | "HEVC" => "hvc1.1.6.L93.B0".to_string(),
                _ => format!("unknown:{}", encoding),
            };

            if let Some(params) = stream.parameters() {
                if let retina::codec::ParametersRef::Video(vp) = params {
                    width = vp.pixel_dimensions().0;
                    height = vp.pixel_dimensions().1;
                    extradata = vp.extra_data().to_vec();
                    // 尝试从 SDP 获取帧率
                    if let Some(fr) = vp.frame_rate() {
                        fps = fr.0 as f64 / fr.1 as f64;
                    }
                }
            }

            println!(
                "📹 [retina] 视频流: {} {}x{}, extradata: {} bytes",
                codec_str,
                width,
                height,
                extradata.len()
            );
        } else if stream.media() == "audio" && audio_codec.is_empty() {
            let encoding = stream.encoding_name().to_uppercase();
            audio_codec = encoding.clone();
            if let Some(params) = stream.parameters() {
                if let retina::codec::ParametersRef::Audio(ap) = params {
                    audio_sample_rate = ap.clock_rate();
                    // retina 不直接提供声道数，根据编码类型猜测
                    audio_channels = match encoding.as_str() {
                        "PCMA" | "PCMU" => 1, // G.711 通常是单声道
                        _ => 2,               // 默认立体声
                    };
                }
            }
            println!(
                "🎵 [retina] 音频流: {} {}Hz {}ch",
                audio_codec, audio_sample_rate, audio_channels
            );
        }
    }

    let video_idx = video_idx.ok_or("未找到视频流")?;

    // 发送元数据 (WebCodecs 配置)
    send_metadata(
        &on_data,
        &StreamMetadata {
            codec: codec_str.clone(),
            width,
            height,
            fps,
            extradata: extradata.clone(),
        },
    );

    // 发送完整流信息 (UI 面板显示)
    let stream_info = StreamInfo {
        protocol: "RTSP".to_string(),
        backend: "retina".to_string(),
        url: url.clone(),
        video_codec: video_codec_name,
        video_codec_full: codec_str,
        video_width: width,
        video_height: height,
        video_fps: fps,
        video_bitrate: 0, // RTSP 通常不提供码率信息
        video_profile: String::new(),
        video_level: String::new(),
        audio_codec,
        audio_sample_rate,
        audio_channels,
        audio_bits_per_sample: 16,
        audio_bitrate: 0,
        total_packets: 0,
        total_bytes: 0,
        keyframes: 0,
        dropped_frames: 0,
        start_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    };
    send_stream_info(&on_data, &stream_info);

    // 设置并播放
    session
        .setup(video_idx, setup_opts)
        .await
        .map_err(|e| format!("SETUP 失败: {}", e))?;

    let playing = session
        .play(PlayOptions::default())
        .await
        .map_err(|e| format!("PLAY 失败: {}", e))?;

    let mut demuxed = playing
        .demuxed()
        .map_err(|e| format!("demuxed 失败: {}", e))?;

    println!("▶️ [retina] 开始接收...");

    while running.load(Ordering::Relaxed) {
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), demuxed.next()).await {
            Ok(Some(Ok(CodecItem::VideoFrame(frame)))) => {
                let count = packet_count.fetch_add(1, Ordering::Relaxed) + 1;

                let raw_data = frame.data();
                let is_keyframe = frame.is_random_access_point();
                let pts = frame.timestamp().elapsed() as i64;

                let annex_b_data = convert_to_annex_b(raw_data);

                if count <= 3 || count % 100 == 0 {
                    println!(
                        "📦 [retina] #{}: keyframe={}, {} bytes",
                        count,
                        is_keyframe,
                        annex_b_data.len()
                    );
                }

                send_packet(
                    &on_data,
                    &EncodedPacket {
                        data: annex_b_data,
                        is_keyframe,
                        pts,
                        dts: pts,
                        packet_id: count,
                    },
                );
            }
            Ok(Some(Ok(_))) => {}
            Ok(Some(Err(e))) => {
                eprintln!("❌ [retina] 错误: {}", e);
                break;
            }
            Ok(None) => {
                println!("📴 [retina] 流结束");
                break;
            }
            Err(_) => {
                eprintln!("⏱ [retina] 超时");
                break;
            }
        }
    }

    running.store(false, Ordering::SeqCst);
    println!(
        "🛑 [retina] 停止，共 {} 包",
        packet_count.load(Ordering::Relaxed)
    );

    Ok(())
}

// ==================== 纯 Rust FLV 后端 (HTTP-FLV) ====================

async fn run_flv_stream(
    url: String,
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
    on_data: Channel<InvokeResponseBody>,
) -> Result<(), String> {
    // 自动重连循环
    let mut reconnect_count = 0;
    const MAX_RECONNECTS: u32 = 5; // 最多重连 5 次
    let mut last_error = String::new();

    while running.load(Ordering::Relaxed) && reconnect_count < MAX_RECONNECTS {
        if reconnect_count > 0 {
            let msg = format!("第 {} 次重连... ({})", reconnect_count, last_error);
            println!("🔄 [FLV] {}", msg);
            send_error(&on_data, &msg);
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        }

        match run_flv_stream_once(&url, &running, &packet_count, &on_data).await {
            Ok(()) => {
                // 正常结束（用户停止）
                return Ok(());
            }
            Err(e) => {
                if !running.load(Ordering::Relaxed) {
                    // 用户主动停止
                    return Ok(());
                }
                last_error = e.clone();
                eprintln!("⚠️ [FLV] 连接断开: {}", e);
                reconnect_count += 1;
            }
        }
    }

    // 达到最大重连次数，返回错误
    if reconnect_count >= MAX_RECONNECTS {
        let msg = format!("连接失败，已重试 {} 次: {}", MAX_RECONNECTS, last_error);
        return Err(msg);
    }

    Ok(())
}

/// FLV 流单次连接
async fn run_flv_stream_once(
    url: &str,
    running: &Arc<AtomicBool>,
    packet_count: &Arc<AtomicU64>,
    on_data: &Channel<InvokeResponseBody>,
) -> Result<(), String> {
    println!("🔌 [FLV] 连接: {}", url);

    // 使用 reqwest 获取 HTTP 流
    // 注意：不要设置 read_timeout，它会导致 chunked 流中断！
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10)) // 只设置连接超时
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        // 不设置 timeout 和 read_timeout！流式传输需要持续读取
        .build()
        .map_err(|e| format!("创建 HTTP 客户端失败: {}", e))?;

    let response = client
        .get(url)
        .header("Connection", "keep-alive")
        .header("Accept", "*/*")
        // 模拟常见播放器的 User-Agent
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .header("Referer", url)  // 有些服务器检查 Referer
        .send()
        .await
        .map_err(|e| format!("HTTP 请求失败: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP 状态码: {}", response.status()));
    }

    // 打印响应头以便调试 chunked
    let is_chunked = response
        .headers()
        .get("transfer-encoding")
        .map(|v| v.to_str().unwrap_or("").contains("chunked"))
        .unwrap_or(false);

    println!(
        "✅ [FLV] HTTP 连接成功, chunked={}, Content-Type: {:?}",
        is_chunked,
        response.headers().get("content-type")
    );

    let mut stream = response.bytes_stream();
    let mut demuxer = FlvDemuxer::new();
    let mut metadata_sent = false;
    let start_time = std::time::Instant::now();
    let mut last_data_time = std::time::Instant::now();
    let url_for_info = url.to_string();

    println!("▶️ [FLV] 开始接收...");

    while running.load(Ordering::Relaxed) {
        use futures_util::StreamExt;

        // 使用较长的超时，只是为了检测连接是否真的断开
        match tokio::time::timeout(tokio::time::Duration::from_secs(30), stream.next()).await {
            Ok(Some(Ok(chunk))) => {
                last_data_time = std::time::Instant::now();
                demuxer.feed(&chunk);

                // 解析所有可用的视频帧
                while let Some(frame) = demuxer.next_video_frame() {
                    // 发送元数据 (只发一次)
                    if !metadata_sent && demuxer.metadata().width > 0 {
                        let meta = demuxer.metadata();
                        let video_codec_name = match meta.video_codec {
                            Some(VideoCodec::AVC) => "H.264".to_string(),
                            Some(VideoCodec::HEVC) => "HEVC".to_string(),
                            _ => "Unknown".to_string(),
                        };
                        let codec_str = match meta.video_codec {
                            Some(VideoCodec::AVC) => "avc1.640028".to_string(),
                            Some(VideoCodec::HEVC) => "hvc1.1.6.L93.B0".to_string(),
                            _ => "unknown".to_string(),
                        };
                        let fps = if meta.fps > 0.0 { meta.fps } else { 25.0 };

                        send_metadata(
                            &on_data,
                            &StreamMetadata {
                                codec: codec_str.clone(),
                                width: meta.width,
                                height: meta.height,
                                fps,
                                extradata: meta.extradata.clone(),
                            },
                        );

                        // 发送完整流信息 (UI 面板显示)
                        let audio_codec_name = match meta.audio_codec {
                            Some(crate::flv_demuxer::AudioCodec::AAC) => "AAC".to_string(),
                            Some(crate::flv_demuxer::AudioCodec::MP3) => "MP3".to_string(),
                            Some(crate::flv_demuxer::AudioCodec::PCM_ALAW) => {
                                "PCM A-Law".to_string()
                            }
                            Some(crate::flv_demuxer::AudioCodec::PCM_MULAW) => {
                                "PCM μ-Law".to_string()
                            }
                            _ => String::new(),
                        };

                        let stream_info = StreamInfo {
                            protocol: "HTTP-FLV".to_string(),
                            backend: "flv_parser".to_string(),
                            url: url_for_info.clone(),
                            video_codec: video_codec_name,
                            video_codec_full: codec_str,
                            video_width: meta.width,
                            video_height: meta.height,
                            video_fps: fps,
                            video_bitrate: (meta.video_bitrate * 1000) as u64,
                            video_profile: String::new(),
                            video_level: String::new(),
                            audio_codec: audio_codec_name,
                            audio_sample_rate: meta.audio_sample_rate,
                            audio_channels: meta.audio_channels,
                            audio_bits_per_sample: 16,
                            audio_bitrate: (meta.audio_bitrate * 1000) as u64,
                            total_packets: 0,
                            total_bytes: 0,
                            keyframes: 0,
                            dropped_frames: 0,
                            start_time: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0),
                        };
                        send_stream_info(&on_data, &stream_info);

                        metadata_sent = true;
                    }

                    // 跳过序列头帧 (已经通过 metadata 发送了)
                    if frame.is_sequence_header {
                        continue;
                    }

                    let count = packet_count.fetch_add(1, Ordering::Relaxed) + 1;

                    // AVCC 转 Annex-B
                    let annexb_data = avcc_to_annexb(&frame.data, 4);

                    if count <= 5 || count % 100 == 0 {
                        // 调试：检查前几个字节
                        let preview_before: Vec<u8> = frame.data.iter().take(16).cloned().collect();
                        let preview_after: Vec<u8> = annexb_data.iter().take(16).cloned().collect();
                        println!(
                            "📦 [FLV] #{}: keyframe={}, {} -> {} bytes, pts={}",
                            count,
                            frame.is_keyframe,
                            frame.data.len(),
                            annexb_data.len(),
                            frame.pts
                        );
                        if count <= 3 {
                            println!("   原始前16字节: {:02x?}", preview_before);
                            println!("   转换后前16字节: {:02x?}", preview_after);
                        }
                    }

                    send_packet(
                        &on_data,
                        &EncodedPacket {
                            data: annexb_data,
                            is_keyframe: frame.is_keyframe,
                            pts: frame.pts,
                            dts: frame.dts,
                            packet_id: count,
                        },
                    );
                }
            }
            Ok(Some(Err(e))) => {
                let err_str = e.to_string();
                let elapsed = start_time.elapsed().as_secs_f32();
                let count = packet_count.load(Ordering::Relaxed);

                // 区分正常结束和异常
                if err_str.contains("decoding response body")
                    || err_str.contains("connection closed")
                    || err_str.contains("reset by peer")
                    || err_str.contains("end of file")
                {
                    println!("📴 [FLV] 连接关闭 (已运行 {:.1}s, {} 包)", elapsed, count);
                    // 返回错误触发重连
                    return Err("连接关闭".to_string());
                } else {
                    eprintln!(
                        "❌ [FLV] 读取错误: {} (已运行 {:.1}s, {} 包)",
                        e, elapsed, count
                    );
                    return Err(err_str);
                }
            }
            Ok(None) => {
                let elapsed = start_time.elapsed().as_secs_f32();
                let idle_secs = last_data_time.elapsed().as_secs_f32();
                println!(
                    "📴 [FLV] 流结束 (已运行 {:.1}s, 最后数据 {:.1}s 前)",
                    elapsed, idle_secs
                );
                // 流结束也触发重连（直播流可能暂时中断）
                return Err("流结束".to_string());
            }
            Err(_) => {
                let elapsed = start_time.elapsed().as_secs_f32();
                let idle_secs = last_data_time.elapsed().as_secs_f32();
                eprintln!(
                    "⏱ [FLV] 读取超时 30s (已运行 {:.1}s, 最后数据 {:.1}s 前)",
                    elapsed, idle_secs
                );
                return Err("读取超时".to_string());
            }
        }
    }

    // 用户主动停止
    let total_time = start_time.elapsed().as_secs_f32();
    println!(
        "🛑 [FLV] 用户停止，共 {} 包，运行 {:.1}s",
        packet_count.load(Ordering::Relaxed),
        total_time
    );

    Ok(())
}

// ==================== FFmpeg 后端 (MP4/RTMP/其他) ====================

use ffmpeg_next as ffmpeg;

fn run_ffmpeg_stream(
    url: String,
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
    on_data: Channel<InvokeResponseBody>,
) -> Result<(), String> {
    println!("🔌 [FFmpeg] 连接: {}", url);

    // 初始化 FFmpeg
    ffmpeg::init().map_err(|e| format!("FFmpeg 初始化失败: {}", e))?;

    let url_lower = url.to_lowercase();

    // 优化选项，加快连接速度
    let mut options = ffmpeg::Dictionary::new();

    // 关键优化：减少分析时间！
    options.set("analyzeduration", "1000000"); // 1s
    options.set("probesize", "65536"); // 64KB

    // 低延迟设置
    options.set("fflags", "nobuffer");

    // 协议特定设置
    if url_lower.contains(".flv") || url_lower.contains("flv?") {
        options.set("flv_metadata", "1");
    }
    if url_lower.starts_with("rtsp://") {
        options.set("rtsp_transport", "tcp");
        options.set("stimeout", "5000000");
    }
    // RTMP: 不设置额外选项，使用 FFmpeg 默认行为

    println!("🔗 [FFmpeg] 打开: {}", url);

    // 打开输入
    let mut ictx = ffmpeg::format::input_with_dictionary(&url, options)
        .map_err(|e| format!("打开流失败: {}", e))?;

    // 查找视频流
    let video_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or("未找到视频流")?;

    let video_idx = video_stream.index();
    let codec_params = video_stream.parameters();

    // 获取编解码器信息
    let codec_id = unsafe { (*codec_params.as_ptr()).codec_id };
    let width = unsafe { (*codec_params.as_ptr()).width as u32 };
    let height = unsafe { (*codec_params.as_ptr()).height as u32 };

    let is_h264 = codec_id == ffmpeg::ffi::AVCodecID::AV_CODEC_ID_H264;
    let is_hevc = codec_id == ffmpeg::ffi::AVCodecID::AV_CODEC_ID_HEVC;

    let codec_str = if is_h264 {
        "avc1.640028".to_string()
    } else if is_hevc {
        "hvc1.1.6.L93.B0".to_string()
    } else {
        format!("unknown:{:?}", codec_id)
    };

    // 获取 extradata
    let extradata = unsafe {
        let ptr = (*codec_params.as_ptr()).extradata;
        let size = (*codec_params.as_ptr()).extradata_size as usize;
        if !ptr.is_null() && size > 0 {
            std::slice::from_raw_parts(ptr, size).to_vec()
        } else {
            Vec::new()
        }
    };

    // 计算帧率
    let fps = video_stream.avg_frame_rate();
    let fps_val = if fps.denominator() > 0 {
        fps.numerator() as f64 / fps.denominator() as f64
    } else {
        25.0
    };

    println!(
        "📹 [FFmpeg] 视频流: {} {}x{} @ {:.2} fps, extradata: {} bytes",
        codec_str,
        width,
        height,
        fps_val,
        extradata.len()
    );

    // 获取视频码率
    let video_bitrate = unsafe { (*codec_params.as_ptr()).bit_rate as u64 };

    // 查找音频流信息
    let mut audio_codec = String::new();
    let mut audio_sample_rate = 0u32;
    let mut audio_channels = 0u8;
    let mut audio_bits = 0u8;
    let mut audio_bitrate = 0u64;

    if let Some(audio_stream) = ictx.streams().best(ffmpeg::media::Type::Audio) {
        let audio_params = audio_stream.parameters();
        let audio_codec_id = unsafe { (*audio_params.as_ptr()).codec_id };
        audio_sample_rate = unsafe { (*audio_params.as_ptr()).sample_rate as u32 };
        audio_channels = unsafe { (*audio_params.as_ptr()).ch_layout.nb_channels as u8 };
        audio_bitrate = unsafe { (*audio_params.as_ptr()).bit_rate as u64 };

        // 获取音频位深
        let sample_fmt = unsafe { (*audio_params.as_ptr()).format };
        audio_bits = match sample_fmt {
            0 => 8,  // AV_SAMPLE_FMT_U8
            1 => 16, // AV_SAMPLE_FMT_S16
            2 => 32, // AV_SAMPLE_FMT_S32
            3 => 32, // AV_SAMPLE_FMT_FLT
            4 => 64, // AV_SAMPLE_FMT_DBL
            _ => 16, // 默认 16 位
        };

        audio_codec = match audio_codec_id {
            ffmpeg::ffi::AVCodecID::AV_CODEC_ID_AAC => "AAC".to_string(),
            ffmpeg::ffi::AVCodecID::AV_CODEC_ID_MP3 => "MP3".to_string(),
            ffmpeg::ffi::AVCodecID::AV_CODEC_ID_OPUS => "Opus".to_string(),
            ffmpeg::ffi::AVCodecID::AV_CODEC_ID_PCM_ALAW => "PCM A-Law".to_string(),
            ffmpeg::ffi::AVCodecID::AV_CODEC_ID_PCM_MULAW => "PCM μ-Law".to_string(),
            _ => format!("{:?}", audio_codec_id),
        };

        println!(
            "🎵 [FFmpeg] 音频流: {} {}Hz {}ch {}bits {}bps",
            audio_codec, audio_sample_rate, audio_channels, audio_bits, audio_bitrate
        );
    }

    // 确定协议类型
    let protocol = if url_lower.starts_with("rtsp://") {
        "RTSP"
    } else if url_lower.starts_with("rtmp://") {
        "RTMP"
    } else if url_lower.contains(".flv") || url_lower.contains("flv?") {
        "HTTP-FLV"
    } else if url_lower.contains(".mp4") || url_lower.contains(".mkv") || url_lower.contains(".avi")
    {
        "File"
    } else {
        "HTTP"
    };

    // 获取视频编码名称
    let video_codec_name = if is_h264 {
        "H.264".to_string()
    } else if is_hevc {
        "HEVC".to_string()
    } else {
        format!("{:?}", codec_id)
    };

    // 发送元数据
    send_metadata(
        &on_data,
        &StreamMetadata {
            codec: codec_str.clone(),
            width,
            height,
            fps: fps_val,
            extradata: extradata.clone(),
        },
    );

    // 发送完整流信息 (UI 面板显示)
    let stream_info = StreamInfo {
        protocol: protocol.to_string(),
        backend: "FFmpeg".to_string(),
        url: url.clone(),
        video_codec: video_codec_name,
        video_codec_full: codec_str,
        video_width: width,
        video_height: height,
        video_fps: fps_val,
        video_bitrate,
        video_profile: String::new(),
        video_level: String::new(),
        audio_codec,
        audio_sample_rate,
        audio_channels,
        audio_bits_per_sample: audio_bits,
        audio_bitrate,
        total_packets: 0,
        total_bytes: 0,
        keyframes: 0,
        dropped_frames: 0,
        start_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    };
    send_stream_info(&on_data, &stream_info);

    println!("▶️ [FFmpeg] 开始接收...");

    // 读取数据包
    for (stream, packet) in ictx.packets() {
        if !running.load(Ordering::Relaxed) {
            break;
        }

        if stream.index() != video_idx {
            continue;
        }

        let count = packet_count.fetch_add(1, Ordering::Relaxed) + 1;

        let data = packet.data().unwrap_or(&[]).to_vec();
        let pts = packet.pts().unwrap_or(0);
        let dts = packet.dts().unwrap_or(pts);

        // 检测关键帧
        let is_keyframe = packet.is_key()
            || (is_h264 && detect_h264_keyframe(&data))
            || (is_hevc && detect_hevc_keyframe(&data));

        if count <= 3 || count % 100 == 0 {
            println!(
                "📦 [FFmpeg] #{}: keyframe={}, {} bytes",
                count,
                is_keyframe,
                data.len()
            );
        }

        send_packet(
            &on_data,
            &EncodedPacket {
                data,
                is_keyframe,
                pts,
                dts,
                packet_id: count,
            },
        );
    }

    running.store(false, Ordering::SeqCst);
    println!(
        "🛑 [FFmpeg] 停止，共 {} 包",
        packet_count.load(Ordering::Relaxed)
    );

    Ok(())
}

// ==================== 辅助函数 ====================

/// AVCC → Annex-B 转换
fn convert_to_annex_b(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() + 32);
    let mut i = 0;

    while i + 4 <= data.len() {
        let nal_len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;

        if nal_len == 0 || i + nal_len > data.len() {
            break;
        }

        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&data[i..i + nal_len]);
        i += nal_len;
    }

    if result.is_empty() && !data.is_empty() {
        return data.to_vec();
    }

    result
}

/// H.264 关键帧检测
fn detect_h264_keyframe(data: &[u8]) -> bool {
    let mut i = 0;
    while i + 4 < data.len() {
        // Annex-B
        if data[i] == 0 && data[i + 1] == 0 {
            let (start, skip) = if data[i + 2] == 0 && data[i + 3] == 1 {
                (i + 4, 4)
            } else if data[i + 2] == 1 {
                (i + 3, 3)
            } else {
                i += 1;
                continue;
            };

            if start < data.len() {
                let nal_type = data[start] & 0x1F;
                if nal_type == 5 || nal_type == 7 || nal_type == 8 {
                    return true;
                }
            }
            i += skip;
        } else {
            i += 1;
        }
    }

    // AVCC
    let mut j = 0;
    while j + 4 < data.len() {
        let nal_len = u32::from_be_bytes([data[j], data[j + 1], data[j + 2], data[j + 3]]) as usize;
        if nal_len > 0 && j + 4 < data.len() {
            let nal_type = data[j + 4] & 0x1F;
            if nal_type == 5 || nal_type == 7 || nal_type == 8 {
                return true;
            }
        }
        j += 4 + nal_len;
        if nal_len == 0 {
            break;
        }
    }

    false
}

/// HEVC 关键帧检测
fn detect_hevc_keyframe(data: &[u8]) -> bool {
    let mut i = 0;
    while i + 4 < data.len() {
        // Annex-B
        if data[i] == 0 && data[i + 1] == 0 {
            let (start, skip) = if data[i + 2] == 0 && data[i + 3] == 1 {
                (i + 4, 4)
            } else if data[i + 2] == 1 {
                (i + 3, 3)
            } else {
                i += 1;
                continue;
            };

            if start < data.len() {
                let nal_type = (data[start] >> 1) & 0x3F;
                // IDR: 16-21, VPS: 32, SPS: 33, PPS: 34
                if (16..=21).contains(&nal_type) || (32..=34).contains(&nal_type) {
                    return true;
                }
            }
            i += skip;
        } else {
            i += 1;
        }
    }

    // AVCC
    let mut j = 0;
    while j + 4 < data.len() {
        let nal_len = u32::from_be_bytes([data[j], data[j + 1], data[j + 2], data[j + 3]]) as usize;
        if nal_len > 0 && j + 4 < data.len() {
            let nal_type = (data[j + 4] >> 1) & 0x3F;
            if (16..=21).contains(&nal_type) || (32..=34).contains(&nal_type) {
                return true;
            }
        }
        j += 4 + nal_len;
        if nal_len == 0 {
            break;
        }
    }

    false
}

/// 发送元数据 (JSON)
fn send_metadata(channel: &Channel<InvokeResponseBody>, metadata: &StreamMetadata) {
    #[derive(Serialize)]
    struct VideoConfigMessage {
        r#type: String,
        codec: String,
        width: u32,
        height: u32,
        extradata: Vec<u8>,
    }

    let msg = VideoConfigMessage {
        r#type: "video_config".to_string(),
        codec: metadata.codec.clone(),
        width: metadata.width,
        height: metadata.height,
        extradata: metadata.extradata.clone(),
    };

    let json = serde_json::to_string(&msg).unwrap_or_default();
    let _ = channel.send(InvokeResponseBody::Json(json));

    println!(
        "📤 发送元数据: {} {}x{}, extradata: {} bytes",
        metadata.codec,
        metadata.width,
        metadata.height,
        metadata.extradata.len()
    );
}

/// 发送完整流信息 (JSON) - 用于 UI 面板显示
fn send_stream_info(channel: &Channel<InvokeResponseBody>, info: &StreamInfo) {
    #[derive(Serialize)]
    struct StreamInfoMessage {
        r#type: String,
        #[serde(flatten)]
        info: StreamInfo,
    }

    let msg = StreamInfoMessage {
        r#type: "stream_info".to_string(),
        info: info.clone(),
    };

    let json = serde_json::to_string(&msg).unwrap_or_default();
    let _ = channel.send(InvokeResponseBody::Json(json));

    println!(
        "📤 发送流信息: {} {} {}x{}@{}fps",
        info.protocol, info.video_codec, info.video_width, info.video_height, info.video_fps
    );
}

/// 发送数据包 (二进制)
fn send_packet(channel: &Channel<InvokeResponseBody>, packet: &EncodedPacket) {
    let header_size = 30;
    let total_size = header_size + packet.data.len();
    let mut buffer = vec![0u8; total_size];

    buffer[0] = 1; // video
    buffer[1] = if packet.is_keyframe { 1 } else { 0 };
    buffer[2..10].copy_from_slice(&packet.pts.to_le_bytes());
    buffer[10..18].copy_from_slice(&packet.dts.to_le_bytes());
    buffer[18..26].copy_from_slice(&packet.packet_id.to_le_bytes());
    buffer[26..30].copy_from_slice(&(packet.data.len() as u32).to_le_bytes());
    buffer[30..].copy_from_slice(&packet.data);

    let _ = channel.send(InvokeResponseBody::Raw(buffer));
}

/// 发送错误消息 (JSON) - 通知前端连接或解码错误
fn send_error(channel: &Channel<InvokeResponseBody>, error: &str) {
    #[derive(Serialize)]
    struct ErrorMessage {
        r#type: String,
        error: String,
    }

    let msg = ErrorMessage {
        r#type: "error".to_string(),
        error: error.to_string(),
    };

    let json = serde_json::to_string(&msg).unwrap_or_default();
    let _ = channel.send(InvokeResponseBody::Json(json));

    eprintln!("📤 发送错误: {}", error);
}
