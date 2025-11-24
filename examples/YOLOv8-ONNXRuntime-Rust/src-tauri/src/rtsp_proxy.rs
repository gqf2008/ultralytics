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
                    let video_stream = context.streams().best(ffmpeg::media::Type::Video);
                    // Find audio stream
                    let audio_stream = context.streams().best(ffmpeg::media::Type::Audio);

                    let video_idx = video_stream.as_ref().map(|s| s.index());
                    let audio_idx = audio_stream.as_ref().map(|s| s.index());

                    if let Some(stream) = video_stream {
                        // 获取视频流参数
                        let params = stream.parameters();
                        let codec_id = params.id();

                        // 从 codec context 获取分辨率
                        let decoder = ffmpeg::codec::context::Context::from_parameters(params)
                            .map(|ctx| ctx.decoder())
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

                        // 获取音频流信息
                        let (audio_codec, sample_rate, channels) =
                            if let Some(audio_stream) = audio_stream {
                                let audio_params = audio_stream.parameters();
                                let audio_codec_id = audio_params.id();
                                let audio_codec_name = match audio_codec_id {
                                    ffmpeg::codec::Id::AAC => "aac",
                                    ffmpeg::codec::Id::MP3 => "mp3",
                                    ffmpeg::codec::Id::OPUS => "opus",
                                    ffmpeg::codec::Id::PCM_MULAW => "pcm_mulaw",
                                    ffmpeg::codec::Id::PCM_ALAW => "pcm_alaw",
                                    _ => "unknown",
                                };

                                let decoder =
                                    ffmpeg::codec::context::Context::from_parameters(audio_params)
                                        .map(|ctx| ctx.decoder())
                                        .and_then(|dec| dec.audio());

                                let (rate, ch) = if let Ok(audio_ctx) = decoder {
                                    (audio_ctx.rate(), audio_ctx.channels())
                                } else {
                                    (0, 0)
                                };

                                (audio_codec_name, rate, ch as u32)
                            } else {
                                ("none", 0, 0)
                            };

                        // 打印漂亮的 TUI 风格信息框
                        Self::print_stream_info(
                            generation,
                            &url,
                            codec_name,
                            width,
                            height,
                            video_idx,
                            audio_codec,
                            sample_rate,
                            channels,
                            audio_idx,
                        );

                        // 通过 Channel 发送元数据(只发送视频相关)
                        if let Some(ref c) = channel {
                            let metadata = json!({
                                "type": "metadata",
                                "video_codec": codec_name,
                                "width": width,
                                "height": height
                            });
                            let metadata_str = metadata.to_string();
                            let mut metadata_bytes = vec![0xFF, 0xFE]; // 魔数标记
                            metadata_bytes.extend_from_slice(metadata_str.as_bytes());
                            let _ = c.send(InvokeResponseBody::Raw(metadata_bytes));
                        }
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

                            let stream_idx = stream.index();

                            // 只处理视频包
                            if video_idx == Some(stream_idx) {
                                if let Some(data) = packet.data() {
                                    let data_vec = data.to_vec();

                                    // Send to WebSocket subscribers (带类型标记)
                                    let mut packet_with_type = vec![0xAA];
                                    packet_with_type.extend_from_slice(&data_vec);
                                    let _ = tx.send(packet_with_type);

                                    // Send to Channel (纯视频数据)
                                    if let Some(ref c) = channel {
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

    /// 打印漂亮的 TUI 风格流信息框
    fn print_stream_info(
        generation: usize,
        url: &str,
        codec: &str,
        width: u32,
        height: u32,
        stream_idx: Option<usize>,
        audio_codec: &str,
        sample_rate: u32,
        channels: u32,
        audio_idx: Option<usize>,
    ) {
        let codec_display = codec.to_uppercase();
        let resolution = format!("{}x{}", width, height);
        let transport = "TCP";
        let audio_info = if audio_codec != "none" {
            format!(
                "{} {}Hz {}ch",
                audio_codec.to_uppercase(),
                sample_rate,
                channels
            )
        } else {
            "No Audio".to_string()
        };

        println!("\n╔═══════════════════════════════════════════════════════════════════╗");
        println!("║                     🎬 RTSP STREAM CONNECTED                      ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║                                                                   ║");
        println!("║  📌 Generation:  {:<48}║", format!("#{}", generation));
        println!("║  🔗 URL:         {:<48}║", Self::truncate_string(url, 48));
        println!("║                                                                   ║");
        println!("╔═══════════════════════════════════════════════════════════════════╗");
        println!("║  📺 VIDEO STREAM INFO                                             ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║                                                                   ║");
        println!("║  🎞️  Codec:       {:<48}║", codec_display);
        println!("║  📐 Resolution:  {:<48}║", resolution);
        if let Some(idx) = stream_idx {
            println!("║  🔢 Stream ID:   {:<48}║", idx);
        }
        println!("║                                                                   ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  🔊 AUDIO STREAM INFO                                             ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║                                                                   ║");
        println!("║  🎵 Codec:       {:<48}║", audio_info);
        if let Some(idx) = audio_idx {
            println!("║  🔢 Stream ID:   {:<48}║", idx);
        }
        println!("║                                                                   ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  🌐 TRANSPORT                                                     ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║                                                                   ║");
        println!("║  🚀 Protocol:    {:<48}║", transport);
        println!("║  📦 Buffer:      {:<48}║", "64 MB");
        println!("║                                                                   ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  ✅ Status:      STREAMING ACTIVE                                 ║");
        println!("╚═══════════════════════════════════════════════════════════════════╝\n");
    }
    /// 截断字符串到指定长度
    fn truncate_string(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
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
