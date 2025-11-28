//! 视频解复用器模块
//!
//! 只负责从 RTSP/RTMP/文件源读取视频流，提取编码数据包。
//! 解码工作由 decoder.rs 模块或前端 WebCodecs 完成。

use ffmpeg_next as ffmpeg;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

/// 编码数据包
#[derive(Clone)]
pub struct EncodedPacket {
    /// 数据 (Annex-B 格式)
    pub data: Vec<u8>,
    /// 是否是关键帧
    pub is_keyframe: bool,
    /// PTS
    pub pts: i64,
    /// DTS
    pub dts: i64,
    /// 包序号
    pub packet_id: u64,
}

/// 数据包回调类型
pub type PacketCallback = Arc<dyn Fn(EncodedPacket) + Send + Sync>;

/// 视频流元数据
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    /// 编解码器: "h264", "hevc"
    pub codec: String,
    /// 宽度
    pub width: u32,
    /// 高度
    pub height: u32,
    /// 帧率
    pub fps: f64,
    /// extradata (SPS/PPS)
    pub extradata: Vec<u8>,
}

/// 元数据回调类型
pub type MetadataCallback = Arc<dyn Fn(StreamMetadata) + Send + Sync>;

/// 解复用器配置
#[derive(Clone)]
pub struct DemuxerConfig {
    /// 视频源 URL（RTSP/RTMP/文件路径）
    pub url: String,
    /// 连接超时（秒）
    pub connect_timeout: u32,
    /// 读取超时（秒）
    pub read_timeout: u32,
    /// 数据包回调
    pub packet_callback: Option<PacketCallback>,
    /// 元数据回调
    pub metadata_callback: Option<MetadataCallback>,
}

impl Default for DemuxerConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            connect_timeout: 10,
            read_timeout: 5,
            packet_callback: None,
            metadata_callback: None,
        }
    }
}

/// 解复用器
pub struct Demuxer {
    config: DemuxerConfig,
    running: Arc<AtomicBool>,
    packet_count: Arc<AtomicU64>,
}

impl Demuxer {
    /// 创建新的解复用器
    pub fn new(config: DemuxerConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            packet_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 启动解复用器（在新线程中运行）
    pub fn start(&self) -> Result<(), String> {
        if self.running.load(Ordering::SeqCst) {
            return Err("Demuxer already running".to_string());
        }

        self.running.store(true, Ordering::SeqCst);
        self.packet_count.store(0, Ordering::SeqCst);

        let config = self.config.clone();
        let running = self.running.clone();
        let packet_count = self.packet_count.clone();

        thread::spawn(move || {
            if let Err(e) = run_demuxer_loop(config, running.clone(), packet_count) {
                eprintln!("[Demuxer] Error: {}", e);
            }
            running.store(false, Ordering::SeqCst);
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

    /// 获取已处理的包数
    pub fn packet_count(&self) -> u64 {
        self.packet_count.load(Ordering::SeqCst)
    }
}

/// 解复用器主循环
fn run_demuxer_loop(
    config: DemuxerConfig,
    running: Arc<AtomicBool>,
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
    opts.set("buffer_size", "16777216");
    opts.set("max_delay", "500000");
    opts.set("fflags", "nobuffer");
    opts.set("flags", "low_delay");

    println!("[Demuxer] Connecting to: {}", config.url);

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

    // 发送元数据
    if let Some(ref callback) = config.metadata_callback {
        callback(StreamMetadata {
            codec: codec_name.to_string(),
            width,
            height,
            fps,
            extradata: extradata.clone(),
        });
    }

    // 解复用循环
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
            let mut full_data = extradata_to_annexb(&extradata);
            full_data.extend(avcc_to_annexb(&data));
            full_data
        } else {
            avcc_to_annexb(&data)
        };

        packet_id += 1;
        packet_count.fetch_add(1, Ordering::SeqCst);

        // 发送数据包
        if let Some(ref callback) = config.packet_callback {
            callback(EncodedPacket {
                data: annexb_data,
                is_keyframe,
                pts,
                dts,
                packet_id,
            });
        }
    }

    println!("[Demuxer] Stream ended, total packets: {}", packet_id);
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

/// 将 extradata (AVCC 格式) 转换为 Annex-B 格式
fn extradata_to_annexb(extradata: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    if extradata.len() < 7 {
        return result;
    }

    // AVCC extradata 格式:
    // [0] version
    // [1] profile
    // [2] compatibility
    // [3] level
    // [4] NALU length size - 1 (masked with 0x03)
    // [5] number of SPS (masked with 0x1F)
    // [6..] SPS data
    // [...] number of PPS
    // [...] PPS data

    let mut offset = 5;

    // 读取 SPS
    if offset < extradata.len() {
        let num_sps = extradata[offset] & 0x1F;
        offset += 1;

        for _ in 0..num_sps {
            if offset + 2 > extradata.len() {
                break;
            }
            let sps_len = u16::from_be_bytes([extradata[offset], extradata[offset + 1]]) as usize;
            offset += 2;

            if offset + sps_len > extradata.len() {
                break;
            }

            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend_from_slice(&extradata[offset..offset + sps_len]);
            offset += sps_len;
        }
    }

    // 读取 PPS
    if offset < extradata.len() {
        let num_pps = extradata[offset];
        offset += 1;

        for _ in 0..num_pps {
            if offset + 2 > extradata.len() {
                break;
            }
            let pps_len = u16::from_be_bytes([extradata[offset], extradata[offset + 1]]) as usize;
            offset += 2;

            if offset + pps_len > extradata.len() {
                break;
            }

            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend_from_slice(&extradata[offset..offset + pps_len]);
            offset += pps_len;
        }
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

    pub fn packet_count(&self) -> u64 {
        self.demuxer.as_ref().map(|d| d.packet_count()).unwrap_or(0)
    }
}

impl Default for DemuxerHandle {
    fn default() -> Self {
        Self::new()
    }
}
