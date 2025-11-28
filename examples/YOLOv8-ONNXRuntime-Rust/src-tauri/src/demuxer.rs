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

    let is_hevc = codec_id == ffmpeg::ffi::AVCodecID::AV_CODEC_ID_HEVC;

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

    // 打印 extradata 头部用于调试格式
    let extradata_header: Vec<u8> = extradata.iter().take(16).cloned().collect();
    println!(
        "[Demuxer] Stream opened: {} {}x{} @ {:.2} fps, extradata: {} bytes, header: {:?}",
        codec_name,
        width,
        height,
        fps,
        extradata.len(),
        extradata_header
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

    println!("[Demuxer] Starting packet loop...");

    for (stream_idx, packet) in ictx.packets() {
        if !running.load(Ordering::SeqCst) {
            println!("[Demuxer] Stop signal received");
            break;
        }

        if stream_idx.index() != video_stream_index {
            continue;
        }

        let data = packet.data().map(|d| d.to_vec()).unwrap_or_default();
        if data.is_empty() {
            continue;
        }

        // 检测关键帧：对于 HEVC，手动检查 NAL unit type
        // 因为某些 RTSP 流中 FFmpeg 的 is_key() 可能不准确
        let is_keyframe = if codec_name == "hevc" {
            detect_hevc_keyframe(&data) || packet.is_key()
        } else {
            detect_h264_keyframe(&data) || packet.is_key()
        };

        let pts = packet.pts().unwrap_or(0);
        let dts = packet.dts().unwrap_or(0);

        // 直接发送 Annex-B 格式数据给前端 WebCodecs
        // 前端会进行必要的格式转换

        packet_id += 1;
        packet_count.fetch_add(1, Ordering::SeqCst);

        // 首包和关键帧打印详细信息
        if packet_id == 1 || (is_keyframe && packet_id <= 5) {
            let data_header: Vec<u8> = data.iter().take(16).cloned().collect();
            println!(
                "[Demuxer] Packet #{}: keyframe={}, size={} bytes, header: {:?}",
                packet_id,
                is_keyframe,
                data.len(),
                data_header
            );
        } else if packet_id % 100 == 0 {
            println!(
                "[Demuxer] Packet #{}: keyframe={}, size={} bytes",
                packet_id,
                is_keyframe,
                data.len()
            );
        }

        // 发送数据包 (原始 AVCC 格式)
        if let Some(ref callback) = config.packet_callback {
            callback(EncodedPacket {
                data: data, // 直接使用原始数据
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

/// 检测 HEVC (H.265) 数据中是否包含 IDR 帧 (关键帧)
/// 支持 Annex-B 格式 (00 00 00 01 起始码)
fn detect_hevc_keyframe(data: &[u8]) -> bool {
    // HEVC NAL unit type 位于第一个字节的 bit 1-6
    // IDR 帧类型:
    // - 19 (IDR_W_RADL): IDR with RADL pictures
    // - 20 (IDR_N_LP): IDR without leading pictures
    // - 16-18 (BLA): Broken Link Access
    // VPS/SPS/PPS 也算作关键帧的一部分
    // - 32 (VPS): Video Parameter Set
    // - 33 (SPS): Sequence Parameter Set
    // - 34 (PPS): Picture Parameter Set

    let mut i = 0;
    while i + 4 < data.len() {
        // 查找 Annex-B 起始码
        if data[i] == 0 && data[i + 1] == 0 {
            let start_code_len = if data[i + 2] == 0 && data[i + 3] == 1 {
                4
            } else if data[i + 2] == 1 {
                3
            } else {
                i += 1;
                continue;
            };

            let nalu_start = i + start_code_len;
            if nalu_start >= data.len() {
                break;
            }

            // HEVC NAL unit type: (byte >> 1) & 0x3F
            let nal_type = (data[nalu_start] >> 1) & 0x3F;

            // IDR 帧或参数集
            if nal_type >= 16 && nal_type <= 21 {
                // BLA_W_LP(16), BLA_W_RADL(17), BLA_N_LP(18), IDR_W_RADL(19), IDR_N_LP(20), CRA(21)
                return true;
            }
            if nal_type >= 32 && nal_type <= 34 {
                // VPS(32), SPS(33), PPS(34)
                return true;
            }

            i = nalu_start;
        }
        i += 1;
    }
    false
}

/// 检测 H.264 数据中是否包含 IDR 帧 (关键帧)
fn detect_h264_keyframe(data: &[u8]) -> bool {
    // H.264 NAL unit type 位于第一个字节的低 5 位
    // IDR 帧类型: 5 (IDR slice)
    // SPS: 7, PPS: 8

    let mut i = 0;
    while i + 4 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            let start_code_len = if data[i + 2] == 0 && data[i + 3] == 1 {
                4
            } else if data[i + 2] == 1 {
                3
            } else {
                i += 1;
                continue;
            };

            let nalu_start = i + start_code_len;
            if nalu_start >= data.len() {
                break;
            }

            let nal_type = data[nalu_start] & 0x1F;

            // IDR slice (5) 或 SPS (7) 或 PPS (8)
            if nal_type == 5 || nal_type == 7 || nal_type == 8 {
                return true;
            }

            i = nalu_start;
        }
        i += 1;
    }
    false
}

/// 长度前缀格式 (AVCC/HVCC) 转 Annex-B 格式
///
/// AVCC/HVCC 数据包格式: [4字节长度][NALU数据][4字节长度][NALU数据]...
/// Annex-B 格式: [0x00000001][NALU数据][0x00000001][NALU数据]...
#[allow(dead_code)]
fn length_prefixed_to_annexb(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len() + 32);
    let mut offset = 0;

    while offset + 4 <= data.len() {
        // 读取 4 字节长度前缀 (大端序)
        let nalu_size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        offset += 4;

        if nalu_size == 0 || offset + nalu_size > data.len() {
            break;
        }

        // 添加 Annex-B 起始码
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&data[offset..offset + nalu_size]);

        offset += nalu_size;
    }

    // 如果没有解析出任何 NALU，可能已经是 Annex-B 格式
    if result.is_empty() && !data.is_empty() {
        // 检查是否已经有起始码
        if data.len() >= 4 && (data[0..3] == [0, 0, 1] || data[0..4] == [0, 0, 0, 1]) {
            return data.to_vec();
        }
        // 否则添加起始码
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(data);
    }

    result
}

/// H.264 AVCC extradata 转 Annex-B 格式
///
/// AVCC extradata 格式:
/// - [0]: version (always 1)
/// - [1]: AVCProfileIndication
/// - [2]: profile_compatibility  
/// - [3]: AVCLevelIndication
/// - [4]: lengthSizeMinusOne (NAL unit length size - 1, masked with 0x03)
/// - [5]: numOfSPS (masked with 0x1F)
/// - [6..]: SPS entries (each: 2-byte length + SPS data)
/// - [...]: numOfPPS
/// - [...]: PPS entries (each: 2-byte length + PPS data)
fn avcc_extradata_to_annexb(extradata: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    if extradata.len() < 7 {
        return result;
    }

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

/// HEVC HVCC extradata 转 Annex-B 格式
///
/// HVCC extradata 格式:
/// - [0]: configurationVersion (always 1)
/// - [1-22]: 各种 profile/level 信息
/// - [21]: lengthSizeMinusOne (NAL unit length size - 1, masked with 0x03)
/// - [22]: numOfArrays (NALU array 数量)
/// - [23..]: NALU arrays, each:
///   - [0]: array_completeness(1) + reserved(1) + NAL_unit_type(6)
///   - [1-2]: numNalus (2 bytes, big-endian)
///   - [3..]: NALU entries (each: 2-byte length + NALU data)
fn hvcc_extradata_to_annexb(extradata: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    // HVCC 最小长度是 23 字节 header
    if extradata.len() < 23 {
        println!("[HVCC] extradata too short: {} bytes", extradata.len());
        return result;
    }

    let num_arrays = extradata[22] as usize;
    let mut offset = 23;

    println!(
        "[HVCC] Parsing {} NALU arrays from {} bytes extradata",
        num_arrays,
        extradata.len()
    );

    for array_idx in 0..num_arrays {
        if offset + 3 > extradata.len() {
            println!("[HVCC] Array {} truncated at offset {}", array_idx, offset);
            break;
        }

        let nal_type = extradata[offset] & 0x3F;
        let num_nalus = u16::from_be_bytes([extradata[offset + 1], extradata[offset + 2]]) as usize;
        offset += 3;

        println!(
            "[HVCC] Array {}: NAL type={}, count={}",
            array_idx, nal_type, num_nalus
        );

        for nalu_idx in 0..num_nalus {
            if offset + 2 > extradata.len() {
                println!("[HVCC] NALU {} length truncated", nalu_idx);
                break;
            }

            let nalu_len = u16::from_be_bytes([extradata[offset], extradata[offset + 1]]) as usize;
            offset += 2;

            if offset + nalu_len > extradata.len() {
                println!(
                    "[HVCC] NALU {} data truncated: need {}, have {}",
                    nalu_idx,
                    nalu_len,
                    extradata.len() - offset
                );
                break;
            }

            // 添加 Annex-B 起始码
            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend_from_slice(&extradata[offset..offset + nalu_len]);
            offset += nalu_len;
        }
    }

    println!("[HVCC] Generated {} bytes Annex-B data", result.len());
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
