use ffmpeg_next as ffmpeg;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::ipc::{Channel, InvokeResponseBody};
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

pub struct RtspProxy {
    tx: broadcast::Sender<Vec<u8>>,
    running: Arc<Mutex<bool>>,
    active_generation: Arc<AtomicUsize>,
}

impl RtspProxy {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100); // Buffer 100 packets
        Self {
            tx,
            running: Arc::new(Mutex::new(false)),
            active_generation: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn start(&self, url: String, channel: Option<Channel>) {
        let tx = self.tx.clone();
        let running = self.running.clone();

        // Increment generation to invalidate previous threads
        let generation = self.active_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let active_gen = self.active_generation.clone();

        *running.lock().unwrap() = true;

        thread::spawn(move || {
            println!("🔌 RTSP Proxy connecting to: {} (Gen: {})", url, generation);

            if let Err(e) = ffmpeg::init() {
                eprintln!("❌ FFmpeg init failed: {}", e);
                return;
            }

            // 设置 RTSP 选项
            let mut opts = ffmpeg::Dictionary::new();
            opts.set("rtsp_transport", "tcp");
            opts.set("buffer_size", "67108864");
            opts.set("rtsp_flags", "prefer_tcp");

            match ffmpeg::format::input_with_dictionary(&url, opts) {
                Ok(mut context) => {
                    println!("✅ RTSP Connected. Stream info:");

                    // Find video stream
                    let video_stream = context
                        .streams()
                        .best(ffmpeg::media::Type::Video);

                    if let Some(stream) = video_stream {
                        let video_idx = stream.index();
                        
                        // 获取视频流参数
                        let params = stream.parameters();
                        let codec_id = params.id();
                        
                        // 从 codec context 获取分辨率
                        let decoder = ffmpeg::codec::context::Context::from_parameters(params)
                            .and_then(|ctx| ctx.decoder())
                            .and_then(|dec| dec.video());
                        
                        let (width, height) = if let Ok(video_ctx) = decoder {
                            (video_ctx.width(), video_ctx.height())
                        } else {
                            (0, 0)
                        };
                        
                        // 判断编码格式
                        let codec_name = match codec_id {
                            ffmpeg::codec::Id::H264 => "h264",
                            ffmpeg::codec::Id::H265 | ffmpeg::codec::Id::HEVC => "hevc",
                            _ => "unknown",
                        };
                        
                        println!("   Video Stream Index: {}", video_idx);
                        println!("   Codec: {} ({:?})", codec_name, codec_id);
                        println!("   Resolution: {}x{}", width, height);
                        
                        // 发送元数据到前端 (通过特殊标记)
                        if let Some(ref c) = channel {
                            let metadata = json!({
                                "type": "metadata",
                                "codec": codec_name,
                                "width": width,
                                "height": height
                            });
                            let metadata_str = metadata.to_string();
                            let mut metadata_bytes = vec![0xFF, 0xFE]; // 魔数标记
                            metadata_bytes.extend_from_slice(metadata_str.as_bytes());
                            let _ = c.send(InvokeResponseBody::Raw(metadata_bytes));
                            println!("📤 Metadata sent: {}", metadata_str);
                        }
                        println!("   Video Stream Index: {}", video_idx);
                        println!("🚀 Proxying packets...");

                        for (stream, packet) in context.packets() {
                            // Check if we are still the active generation
                            if active_gen.load(Ordering::Relaxed) != generation {
                                println!("🛑 Stopping obsolete proxy thread (Gen: {})", generation);
                                break;
                            }

                            if !*running.lock().unwrap() {
                                break;
                            }

                            if stream.index() == video_idx {
                                if let Some(data) = packet.data() {
                                    let data_vec = data.to_vec();

                                    // Send to WebSocket subscribers
                                    let _ = tx.send(data_vec.clone());

                                    // Send to Tauri Channel (IPC)
                                    if let Some(ref c) = channel {
                                        // Tauri v2 Channel send
                                        let _ = c.send(InvokeResponseBody::Raw(data_vec));
                                    }
                                }
                            }
                        }
                    } else {
                        eprintln!("❌ No video stream found");
                    }
                }
                Err(e) => eprintln!("❌ Failed to open RTSP stream: {}", e),
            }

            // Only set running to false if we are the active generation
            // This prevents a race where a new thread starts (running=true)
            // and this old thread exits and sets running=false.
            if active_gen.load(Ordering::Relaxed) == generation {
                println!("🛑 RTSP Proxy stopped (Gen: {})", generation);
                *running.lock().unwrap() = false;
            }
        });
    }

    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }

    pub async fn run_server(&self, port: u16) {
        let addr = format!("127.0.0.1:{}", port);
        let listener = TcpListener::bind(&addr)
            .await
            .expect("Failed to bind WS port");
        println!("📡 WebSocket Proxy running on ws://{}", addr);

        loop {
            if let Ok((stream, _)) = listener.accept().await {
                let mut rx = self.tx.subscribe();
                tokio::spawn(async move {
                    if let Ok(ws_stream) = accept_async(stream).await {
                        let (mut write, _) = ws_stream.split();
                        while let Ok(data) = rx.recv().await {
                            if write.send(Message::Binary(data)).await.is_err() {
                                break;
                            }
                        }
                    }
                });
            }
        }
    }
}
