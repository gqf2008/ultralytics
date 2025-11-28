//! Pure Rust RTSP Proxy - using retina crate (no FFmpeg dependency)
//!
//! 使用 retina 纯 Rust 实现 RTSP 客户端，无需 FFmpeg 依赖。
//! 发送编码数据给前端 WebCodecs 解码。

use futures_util::StreamExt;
use retina::client::{Credentials, PlayOptions, Session, SessionOptions, SetupOptions, Transport};
use retina::codec::CodecItem;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tauri::ipc::{Channel, InvokeResponseBody};
use url::Url;

/// 流元数据
#[derive(Clone, Debug, Serialize)]
pub struct StreamMetadata {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub extradata: Vec<u8>,
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

/// RTSP 代理配置
#[derive(Clone, Debug, Deserialize)]
pub struct RtspConfig {
    pub url: String,
}

/// RTSP 代理 (使用 retina 纯 Rust 实现)
pub struct RtspProxy {
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
}

impl RtspProxy {
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

    /// 启动 RTSP 流
    pub fn start(&self, config: RtspConfig, on_data: Channel<InvokeResponseBody>) {
        // 先停止旧流
        self.running.store(false, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(100));

        self.running.store(true, Ordering::SeqCst);
        self.packet_count.store(0, Ordering::SeqCst);

        let running = self.running.clone();
        let packet_count = self.packet_count.clone();
        let url = config.url.clone();

        // 在新 tokio runtime 中运行
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");

            rt.block_on(async move {
                if let Err(e) = run_rtsp_stream(url, running, packet_count, on_data).await {
                    eprintln!("❌ RTSP 流错误: {}", e);
                }
            });
        });
    }

    /// 停止 RTSP 流
    pub fn stop(&self) {
        println!("⏹ 停止 RTSP 流");
        self.running.store(false, Ordering::SeqCst);
    }
}

/// 运行 RTSP 流
async fn run_rtsp_stream(
    url: String,
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
    on_data: Channel<InvokeResponseBody>,
) -> Result<(), String> {
    println!("🔌 连接 RTSP: {}", url);

    // 解析 URL
    let mut parsed_url = Url::parse(&url).map_err(|e| format!("URL 解析失败: {}", e))?;

    // 提取认证信息
    let creds = if !parsed_url.username().is_empty() {
        let username = parsed_url.username().to_string();
        let password = parsed_url.password().unwrap_or("").to_string();
        // 清除 URL 中的凭据
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

    let mut session = Session::describe(parsed_url, session_opts)
        .await
        .map_err(|e| format!("DESCRIBE 失败: {}", e))?;

    // 查找视频流
    let mut video_idx = None;
    let mut codec_str = "unknown".to_string();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut extradata = Vec::new();

    for (idx, stream) in session.streams().iter().enumerate() {
        if stream.media() == "video" {
            video_idx = Some(idx);

            // 获取编码格式
            let encoding = stream.encoding_name().to_uppercase();
            codec_str = match encoding.as_str() {
                "H264" => "avc1.640028".to_string(), // H.264 High Profile Level 4.0
                "H265" | "HEVC" => "hvc1.1.6.L93.B0".to_string(), // HEVC Main Profile
                _ => format!("unknown:{}", encoding),
            };

            // 获取参数
            if let Some(params) = stream.parameters() {
                if let retina::codec::ParametersRef::Video(vp) = params {
                    width = vp.pixel_dimensions().0;
                    height = vp.pixel_dimensions().1;

                    // 获取 extradata (SPS/PPS for H.264, VPS/SPS/PPS for HEVC)
                    extradata = vp.extra_data().to_vec();
                }
            }

            println!(
                "📹 发现视频流 #{}: {} {}x{}, extradata: {} bytes",
                idx,
                codec_str,
                width,
                height,
                extradata.len()
            );
            break;
        }
    }

    let video_idx = video_idx.ok_or("未找到视频流")?;

    // 发送元数据
    let metadata = StreamMetadata {
        codec: codec_str.clone(),
        width,
        height,
        fps: 25.0, // retina 可能不提供 fps，使用默认值
        extradata: extradata.clone(),
    };

    send_metadata(&on_data, &metadata);

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

    println!("▶️ 开始接收视频流...");

    // 读取视频帧
    while running.load(Ordering::Relaxed) {
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), demuxed.next()).await {
            Ok(Some(Ok(CodecItem::VideoFrame(frame)))) => {
                let count = packet_count.fetch_add(1, Ordering::Relaxed) + 1;

                // 获取帧数据
                let raw_data = frame.data();
                let is_keyframe = frame.is_random_access_point();
                let pts = frame.timestamp().elapsed() as i64;

                // retina 输出 AVC/AVCC 格式 (4字节长度前缀)
                // 需要转换为 Annex-B 格式 (00 00 00 01 起始码)
                let annex_b_data = convert_to_annex_b(raw_data);

                if count <= 3 || count % 100 == 0 {
                    println!(
                        "📦 帧 #{}: keyframe={}, size={} -> {} bytes, pts={}",
                        count,
                        is_keyframe,
                        raw_data.len(),
                        annex_b_data.len(),
                        pts
                    );
                }

                // 发送数据包
                let packet = EncodedPacket {
                    data: annex_b_data,
                    is_keyframe,
                    pts,
                    dts: pts,
                    packet_id: count,
                };

                send_packet(&on_data, &packet);
            }
            Ok(Some(Ok(_))) => {
                // 其他类型的 CodecItem (音频等)，忽略
            }
            Ok(Some(Err(e))) => {
                eprintln!("❌ 读取帧错误: {}", e);
                break;
            }
            Ok(None) => {
                println!("📴 流结束");
                break;
            }
            Err(_) => {
                eprintln!("⏱ 读取超时");
                break;
            }
        }
    }

    running.store(false, Ordering::SeqCst);
    println!(
        "🛑 RTSP 流停止，共接收 {} 个数据包",
        packet_count.load(Ordering::Relaxed)
    );

    Ok(())
}

/// 将 AVC/AVCC 格式 (4字节长度前缀) 转换为 Annex-B 格式 (00 00 00 01 起始码)
fn convert_to_annex_b(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() + 32);
    let mut i = 0;

    while i + 4 <= data.len() {
        // 读取 NAL 单元长度 (big-endian u32)
        let nal_len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;

        if nal_len == 0 || i + nal_len > data.len() {
            // 数据不完整，停止
            break;
        }

        // 添加 Annex-B 起始码
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        // 添加 NAL 数据
        result.extend_from_slice(&data[i..i + nal_len]);
        i += nal_len;
    }

    // 如果转换失败，返回原始数据（可能已经是 Annex-B）
    if result.is_empty() && !data.is_empty() {
        return data.to_vec();
    }

    result
}

/// 发送元数据 (JSON 格式)
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

/// 发送数据包 (二进制格式)
/// 格式：
/// [0]     : type (1=video, 2=audio)
/// [1]     : is_keyframe (0/1)
/// [2..10] : pts (i64, little-endian)
/// [10..18]: dts (i64, little-endian)
/// [18..26]: packet_id (u64, little-endian)
/// [26..30]: data_len (u32, little-endian)
/// [30..]  : data
fn send_packet(channel: &Channel<InvokeResponseBody>, packet: &EncodedPacket) {
    let header_size = 30;
    let total_size = header_size + packet.data.len();
    let mut buffer = vec![0u8; total_size];

    buffer[0] = 1; // type: video
    buffer[1] = if packet.is_keyframe { 1 } else { 0 };
    buffer[2..10].copy_from_slice(&packet.pts.to_le_bytes());
    buffer[10..18].copy_from_slice(&packet.dts.to_le_bytes());
    buffer[18..26].copy_from_slice(&packet.packet_id.to_le_bytes());
    buffer[26..30].copy_from_slice(&(packet.data.len() as u32).to_le_bytes());
    buffer[30..].copy_from_slice(&packet.data);

    let _ = channel.send(InvokeResponseBody::Raw(buffer));
}
