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

// PCM A-law 解码表
static ALAW_TABLE: [i16; 256] = [
    -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736, -7552, -7296, -8064, -7808, -6528,
    -6272, -7040, -6784, -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368, -3776, -3648,
    -4032, -3904, -3264, -3136, -3520, -3392, -22016, -20992, -24064, -23040, -17920, -16896,
    -19968, -18944, -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136, -11008, -10496,
    -12032, -11520, -8960, -8448, -9984, -9472, -15104, -14592, -16128, -15616, -13056, -12544,
    -14080, -13568, -344, -328, -376, -360, -280, -264, -312, -296, -472, -456, -504, -488, -408,
    -392, -440, -424, -88, -72, -120, -104, -24, -8, -56, -40, -216, -200, -248, -232, -152, -136,
    -184, -168, -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184, -1888, -1824, -2016, -1952,
    -1632, -1568, -1760, -1696, -688, -656, -752, -720, -560, -528, -624, -592, -944, -912, -1008,
    -976, -816, -784, -880, -848, 5504, 5248, 6016, 5760, 4480, 4224, 4992, 4736, 7552, 7296, 8064,
    7808, 6528, 6272, 7040, 6784, 2752, 2624, 3008, 2880, 2240, 2112, 2496, 2368, 3776, 3648, 4032,
    3904, 3264, 3136, 3520, 3392, 22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944, 30208,
    29184, 32256, 31232, 26112, 25088, 28160, 27136, 11008, 10496, 12032, 11520, 8960, 8448, 9984,
    9472, 15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568, 344, 328, 376, 360, 280, 264,
    312, 296, 472, 456, 504, 488, 408, 392, 440, 424, 88, 72, 120, 104, 24, 8, 56, 40, 216, 200,
    248, 232, 152, 136, 184, 168, 1376, 1312, 1504, 1440, 1120, 1056, 1248, 1184, 1888, 1824, 2016,
    1952, 1632, 1568, 1760, 1696, 688, 656, 752, 720, 560, 528, 624, 592, 944, 912, 1008, 976, 816,
    784, 880, 848,
];

// PCM μ-law 解码表
static MULAW_TABLE: [i16; 256] = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956, -23932, -22908, -21884, -20860,
    -19836, -18812, -17788, -16764, -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316, -7932, -7676, -7420, -7164, -6908,
    -6652, -6396, -6140, -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092, -3900, -3772,
    -3644, -3516, -3388, -3260, -3132, -3004, -2876, -2748, -2620, -2492, -2364, -2236, -2108,
    -1980, -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436, -1372, -1308, -1244, -1180,
    -1116, -1052, -988, -924, -876, -844, -812, -780, -748, -716, -684, -652, -620, -588, -556,
    -524, -492, -460, -428, -396, -372, -356, -340, -324, -308, -292, -276, -260, -244, -228, -212,
    -196, -180, -164, -148, -132, -120, -112, -104, -96, -88, -80, -72, -64, -56, -48, -40, -32,
    -24, -16, -8, 0, 32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956, 23932, 22908, 21884,
    20860, 19836, 18812, 17788, 16764, 15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316, 7932, 7676, 7420, 7164, 6908, 6652, 6396,
    6140, 5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092, 3900, 3772, 3644, 3516, 3388, 3260, 3132,
    3004, 2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980, 1884, 1820, 1756, 1692, 1628, 1564, 1500,
    1436, 1372, 1308, 1244, 1180, 1116, 1052, 988, 924, 876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396, 372, 356, 340, 324, 308, 292, 276, 260, 244, 228, 212,
    196, 180, 164, 148, 132, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0,
];

/// 解码 PCM A-law 到 PCM16 (Little Endian)
fn decode_pcm_alaw(data: &[u8]) -> Vec<u8> {
    let mut pcm16 = Vec::with_capacity(data.len() * 2);
    for &byte in data {
        let sample = ALAW_TABLE[byte as usize];
        pcm16.extend_from_slice(&sample.to_le_bytes());
    }
    pcm16
}

/// 解码 PCM μ-law 到 PCM16 (Little Endian)
fn decode_pcm_mulaw(data: &[u8]) -> Vec<u8> {
    let mut pcm16 = Vec::with_capacity(data.len() * 2);
    for &byte in data {
        let sample = MULAW_TABLE[byte as usize];
        pcm16.extend_from_slice(&sample.to_le_bytes());
    }
    pcm16
}

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

    pub fn start(
        &self,
        url: String,
        video_channel: Option<Channel>,
        audio_channel: Option<Channel>,
    ) {
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

                        // 通过 video_channel 发送元数据
                        if let Some(ref c) = video_channel {
                            let metadata = json!({
                                "type": "metadata",
                                "video_codec": codec_name,
                                "width": width,
                                "height": height,
                                "audio_codec": audio_codec,
                                "sample_rate": sample_rate,
                                "channels": channels
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

                            // 处理视频包
                            if video_idx == Some(stream_idx) {
                                if let Some(data) = packet.data() {
                                    let data_vec = data.to_vec();

                                    // Send to WebSocket subscribers (带类型标记)
                                    let mut packet_with_type = vec![0xAA];
                                    packet_with_type.extend_from_slice(&data_vec);
                                    let _ = tx.send(packet_with_type);

                                    // Send to video_channel (纯视频数据,不需要类型标记)
                                    if let Some(ref c) = video_channel {
                                        let _ = c.send(InvokeResponseBody::Raw(data_vec));
                                    }
                                }
                            }
                            // 处理音频包
                            else if audio_idx == Some(stream_idx) {
                                if let Some(data) = packet.data() {
                                    let data_vec = data.to_vec();

                                    // 如果是 PCM A-law 或 μ-law,在后端解码成 PCM16
                                    let audio_data = match audio_codec {
                                        "pcm_alaw" => decode_pcm_alaw(&data_vec),
                                        "pcm_mulaw" => decode_pcm_mulaw(&data_vec),
                                        _ => data_vec.clone(),
                                    };

                                    // Send to WebSocket subscribers (带类型标记)
                                    let mut packet_with_type = vec![0xBB];
                                    packet_with_type.extend_from_slice(&audio_data);
                                    let _ = tx.send(packet_with_type);

                                    // Send to audio_channel (已解码的 PCM16 数据)
                                    if let Some(ref c) = audio_channel {
                                        let _ = c.send(InvokeResponseBody::Raw(audio_data));
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
