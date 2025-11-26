//! Pure Rust RTSP Proxy - using retina crate (no FFmpeg dependency)

use futures_util::{SinkExt, StreamExt};
use retina::client::{Credentials, PlayOptions, Session, SessionOptions, SetupOptions, Transport};
use retina::codec::CodecItem;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::ipc::{Channel, InvokeResponseBody};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

/// 将 AVC/AVCC 格式 (4字节长度前缀) 转换为 Annex-B 格式 (00 00 00 01 起始码)
/// WebCodecs 需要 Annex-B 格式
fn convert_to_annex_b(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;

    while i + 4 <= data.len() {
        // 读取 NAL 单元长度 (big-endian u32)
        let nal_len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        i += 4;

        if i + nal_len > data.len() {
            // 数据不完整，跳过
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

pub struct RtspProxy {
    tx: broadcast::Sender<Vec<u8>>,
    running: Arc<AtomicBool>,
    active_generation: Arc<AtomicUsize>,
}

impl RtspProxy {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            tx,
            running: Arc::new(AtomicBool::new(false)),
            active_generation: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn start(
        &self,
        url: String,
        video_channel: Option<Channel>,
        _audio_channel: Option<Channel>,
    ) {
        // 先停止旧流
        self.running.store(false, Ordering::SeqCst);

        let tx = self.tx.clone();
        let running = self.running.clone();
        // 增加 generation，新流使用新值
        let generation = self.active_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let active_gen = self.active_generation.clone();

        // 等一小段时间让旧任务退出
        let url_clone = url.clone();
        tokio::spawn(async move {
            // 短暂延迟确保旧流退出
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            running.store(true, Ordering::SeqCst);

            println!("RTSP Proxy connecting: {} Gen:{}", url_clone, generation);

            let mut parsed_url = match Url::parse(&url_clone) {
                Ok(u) => u,
                Err(e) => {
                    eprintln!("Invalid URL: {}", e);
                    return;
                }
            };

            // 从 URL 中提取凭据，retina 要求凭据单独传递
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

            let mut session_opts = SessionOptions::default();
            if let Some(c) = creds {
                session_opts = session_opts.creds(Some(c));
            }
            let setup_opts = SetupOptions::default().transport(Transport::Tcp(Default::default()));

            let mut session = match Session::describe(parsed_url, session_opts).await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("DESCRIBE failed: {}", e);
                    return;
                }
            };

            let mut video_idx_opt = None;
            let mut codec_str = "unknown";
            let mut w = 0u32;
            let mut h = 0u32;

            for (idx, stream) in session.streams().iter().enumerate() {
                if stream.media() == "video" {
                    video_idx_opt = Some(idx);
                    codec_str = match stream.encoding_name().to_uppercase().as_str() {
                        "H264" => "h264",
                        "H265" | "HEVC" => "hevc",
                        _ => "unknown",
                    };
                    if let Some(params) = stream.parameters() {
                        if let retina::codec::ParametersRef::Video(vp) = params {
                            w = vp.pixel_dimensions().0;
                            h = vp.pixel_dimensions().1;
                        }
                    }
                }
            }

            let video_idx = match video_idx_opt {
                Some(i) => i,
                None => {
                    eprintln!("No video stream");
                    return;
                }
            };

            println!("Video: {} {}x{}", codec_str, w, h);

            if let Some(ref c) = video_channel {
                let metadata = json!({
                    "type": "metadata",
                    "video_codec": codec_str,
                    "width": w,
                    "height": h
                });
                let mut bytes = vec![0xFF, 0xFE];
                bytes.extend_from_slice(metadata.to_string().as_bytes());
                let _ = c.send(InvokeResponseBody::Raw(bytes));
            }

            if session.setup(video_idx, setup_opts).await.is_err() {
                eprintln!("SETUP failed");
                return;
            }

            let playing = match session.play(PlayOptions::default()).await {
                Ok(p) => p,
                Err(_) => {
                    eprintln!("PLAY failed");
                    return;
                }
            };

            let mut demuxed = match playing.demuxed() {
                Ok(d) => d,
                Err(_) => {
                    eprintln!("demuxed failed");
                    return;
                }
            };

            while running.load(Ordering::Relaxed)
                && active_gen.load(Ordering::Relaxed) == generation
            {
                match demuxed.next().await {
                    Some(Ok(CodecItem::VideoFrame(frame))) => {
                        // retina 输出 AVC 格式 (4字节长度前缀)
                        // WebCodecs 需要 Annex-B 格式 (00 00 00 01 起始码)
                        let raw_data = frame.into_data();
                        let annex_b_data = convert_to_annex_b(&raw_data);

                        let mut pkt = vec![0xAA];
                        pkt.extend_from_slice(&annex_b_data);
                        let _ = tx.send(pkt);
                        if let Some(ref c) = video_channel {
                            let _ = c.send(InvokeResponseBody::Raw(annex_b_data.clone()));
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) | None => break,
                }
            }
            running.store(false, Ordering::SeqCst);
        });
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub async fn run_server(&self, port: u16) {
        let addr = format!("127.0.0.1:{}", port);
        let listener = TcpListener::bind(&addr).await.expect("bind failed");
        println!("WS Proxy on ws://{}", addr);
        loop {
            if let Ok((stream, _)) = listener.accept().await {
                let mut rx = self.tx.subscribe();
                tokio::spawn(async move {
                    if let Ok(ws) = accept_async(stream).await {
                        let (mut w, _) = ws.split();
                        while let Ok(data) = rx.recv().await {
                            if w.send(Message::Binary(data)).await.is_err() {
                                break;
                            }
                        }
                    }
                });
            }
        }
    }
}
