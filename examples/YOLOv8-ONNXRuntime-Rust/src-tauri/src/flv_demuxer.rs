//! 纯 Rust FLV 解复用器
//!
//! 支持 HTTP-FLV 流，无需 FFmpeg 依赖
//! FLV 格式参考: https://en.wikipedia.org/wiki/Flash_Video

use amf::{Amf0Value, Version};
use std::io::Cursor;

/// FLV Tag 类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TagType {
    Audio = 8,
    Video = 9,
    Script = 18,
}

impl TryFrom<u8> for TagType {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            8 => Ok(TagType::Audio),
            9 => Ok(TagType::Video),
            18 => Ok(TagType::Script),
            _ => Err("Unknown tag type"),
        }
    }
}

/// 视频编解码器 ID
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VideoCodec {
    H263 = 2,
    ScreenVideo = 3,
    VP6 = 4,
    VP6Alpha = 5,
    ScreenVideo2 = 6,
    AVC = 7,   // H.264
    HEVC = 12, // H.265 (Enhanced FLV, 或部分厂商扩展)
    // Enhanced FLV (ISO 14496-12 FourCC)
    EnhancedHEVC = 0x68766331, // 'hvc1'
    EnhancedAV1 = 0x61763031,  // 'av01'
}

impl TryFrom<u8> for VideoCodec {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(VideoCodec::H263),
            3 => Ok(VideoCodec::ScreenVideo),
            4 => Ok(VideoCodec::VP6),
            5 => Ok(VideoCodec::VP6Alpha),
            6 => Ok(VideoCodec::ScreenVideo2),
            7 => Ok(VideoCodec::AVC),
            12 => Ok(VideoCodec::HEVC),
            _ => Err("Unknown video codec"),
        }
    }
}

/// AVC 包类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AvcPacketType {
    SequenceHeader = 0, // SPS/PPS
    NaluData = 1,       // NAL 单元
    EndOfSequence = 2,
}

/// 视频帧类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameType {
    Keyframe = 1,
    InterFrame = 2,
    DisposableInterFrame = 3,
    GeneratedKeyframe = 4,
    VideoInfoCommand = 5,
}

impl TryFrom<u8> for FrameType {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(FrameType::Keyframe),
            2 => Ok(FrameType::InterFrame),
            3 => Ok(FrameType::DisposableInterFrame),
            4 => Ok(FrameType::GeneratedKeyframe),
            5 => Ok(FrameType::VideoInfoCommand),
            _ => Err("Unknown frame type"),
        }
    }
}

/// FLV 视频帧
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub codec: VideoCodec,
    pub frame_type: FrameType,
    pub is_keyframe: bool,
    pub pts: i64,
    pub dts: i64,
    pub data: Vec<u8>, // AVCC 格式数据
    pub is_sequence_header: bool,
}

/// FLV 元数据
#[derive(Debug, Clone, Default)]
pub struct FlvMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub video_codec: Option<VideoCodec>,
    pub extradata: Vec<u8>, // SPS/PPS (AVCC 格式)
}

/// FLV 解复用器
pub struct FlvDemuxer {
    buffer: Vec<u8>,
    position: usize,
    metadata: FlvMetadata,
    header_parsed: bool,
}

impl FlvDemuxer {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024 * 1024), // 1MB 缓冲
            position: 0,
            metadata: FlvMetadata::default(),
            header_parsed: false,
        }
    }

    /// 获取元数据
    pub fn metadata(&self) -> &FlvMetadata {
        &self.metadata
    }

    /// 添加数据到缓冲区
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// 解析下一个视频帧
    pub fn next_video_frame(&mut self) -> Option<VideoFrame> {
        // 解析 FLV 头
        if !self.header_parsed {
            if !self.parse_header() {
                return None;
            }
        }

        // 循环查找视频 tag
        loop {
            let tag = self.parse_tag()?;

            if tag.tag_type == TagType::Video {
                return self.parse_video_tag(&tag);
            } else if tag.tag_type == TagType::Script {
                // 解析 script data (onMetaData)
                self.parse_script_data(&tag.data);
            }
            // 跳过音频 tag
        }
    }

    /// 解析 FLV 头 (9 bytes)
    fn parse_header(&mut self) -> bool {
        if self.buffer.len() < self.position + 13 {
            return false;
        }

        // FLV 签名: "FLV"
        if &self.buffer[self.position..self.position + 3] != b"FLV" {
            eprintln!("❌ 无效的 FLV 签名");
            return false;
        }

        let version = self.buffer[self.position + 3];
        let flags = self.buffer[self.position + 4];
        let header_size = u32::from_be_bytes([
            self.buffer[self.position + 5],
            self.buffer[self.position + 6],
            self.buffer[self.position + 7],
            self.buffer[self.position + 8],
        ]) as usize;

        let has_video = (flags & 0x01) != 0;
        let has_audio = (flags & 0x04) != 0;

        println!(
            "📼 FLV v{}: video={}, audio={}, header_size={}",
            version, has_video, has_audio, header_size
        );

        // 跳过 header + PreviousTagSize0 (4 bytes)
        self.position += header_size + 4;
        self.header_parsed = true;

        true
    }

    /// 解析 FLV Tag
    fn parse_tag(&mut self) -> Option<FlvTag> {
        // Tag header: 11 bytes
        if self.buffer.len() < self.position + 11 {
            return None;
        }

        let tag_type = self.buffer[self.position];
        let data_size = u32::from_be_bytes([
            0,
            self.buffer[self.position + 1],
            self.buffer[self.position + 2],
            self.buffer[self.position + 3],
        ]) as usize;

        // 时间戳 (24位) + 时间戳扩展 (8位)
        let timestamp_lower = u32::from_be_bytes([
            0,
            self.buffer[self.position + 4],
            self.buffer[self.position + 5],
            self.buffer[self.position + 6],
        ]);
        let timestamp_ext = self.buffer[self.position + 7] as u32;
        let timestamp = ((timestamp_ext << 24) | timestamp_lower) as i64;

        // StreamID (3 bytes, 总是 0)
        // self.position + 8..11

        // 检查是否有足够数据
        let total_size = 11 + data_size + 4; // header + data + PreviousTagSize
        if self.buffer.len() < self.position + total_size {
            return None;
        }

        let data = self.buffer[self.position + 11..self.position + 11 + data_size].to_vec();

        // 跳过整个 tag
        self.position += total_size;

        // 清理已处理的数据 (保持缓冲区不会无限增长)
        if self.position > 1024 * 1024 {
            self.buffer.drain(..self.position);
            self.position = 0;
        }

        Some(FlvTag {
            tag_type: TagType::try_from(tag_type).ok()?,
            timestamp,
            data,
        })
    }

    /// 解析视频 Tag
    fn parse_video_tag(&mut self, tag: &FlvTag) -> Option<VideoFrame> {
        if tag.data.is_empty() {
            return None;
        }

        // 第一个字节: FrameType (4 bits) + CodecID (4 bits)
        let first_byte = tag.data[0];
        let frame_type_val = (first_byte >> 4) & 0x0F;
        let codec_id = first_byte & 0x0F;

        // Enhanced FLV 检测 (帧类型为 5 = ExVideoTagHeader)
        // 或者检测 codec_id = 0 + FourCC
        let is_enhanced_flv = frame_type_val == 5 || (codec_id == 0 && tag.data.len() >= 5);

        if is_enhanced_flv {
            return self.parse_enhanced_video_tag(tag);
        }

        let frame_type = FrameType::try_from(frame_type_val).ok()?;
        let codec = VideoCodec::try_from(codec_id).ok()?;
        let is_keyframe = frame_type == FrameType::Keyframe;

        // 对于 AVC/HEVC，有额外的头信息
        if codec == VideoCodec::AVC || codec == VideoCodec::HEVC {
            if tag.data.len() < 5 {
                return None;
            }

            let avc_packet_type = tag.data[1];
            let composition_time = i32::from_be_bytes([0, tag.data[2], tag.data[3], tag.data[4]]);

            let pts = tag.timestamp + composition_time as i64;
            let dts = tag.timestamp;

            // 视频数据从第 5 字节开始
            let video_data = tag.data[5..].to_vec();

            if avc_packet_type == AvcPacketType::SequenceHeader as u8 {
                // 这是 SPS/PPS 数据 (AVCC/HVCC 格式)
                self.metadata.extradata = video_data.clone();
                self.metadata.video_codec = Some(codec);

                // 尝试解析分辨率
                if let Some((w, h)) = parse_avcc_resolution(&video_data, codec) {
                    self.metadata.width = w;
                    self.metadata.height = h;
                }

                println!(
                    "📦 收到 {} 序列头: {} bytes, {}x{}",
                    if codec == VideoCodec::AVC {
                        "AVC"
                    } else {
                        "HEVC"
                    },
                    self.metadata.extradata.len(),
                    self.metadata.width,
                    self.metadata.height
                );

                return Some(VideoFrame {
                    codec,
                    frame_type,
                    is_keyframe: true,
                    pts,
                    dts,
                    data: video_data,
                    is_sequence_header: true,
                });
            } else if avc_packet_type == AvcPacketType::NaluData as u8 {
                // 普通视频帧 (AVCC 格式)
                return Some(VideoFrame {
                    codec,
                    frame_type,
                    is_keyframe,
                    pts,
                    dts,
                    data: video_data,
                    is_sequence_header: false,
                });
            }
        }

        None
    }

    /// 解析 Enhanced FLV 视频 Tag (支持 HEVC/AV1)
    fn parse_enhanced_video_tag(&mut self, tag: &FlvTag) -> Option<VideoFrame> {
        if tag.data.len() < 5 {
            return None;
        }

        let first_byte = tag.data[0];
        let frame_type_val = (first_byte >> 4) & 0x0F;
        let packet_type = first_byte & 0x0F;

        // Enhanced FLV: 第 2-5 字节是 FourCC
        let fourcc = &tag.data[1..5];
        let fourcc_str = String::from_utf8_lossy(fourcc);

        // 判断编码类型
        let codec = if fourcc == b"hvc1" || fourcc == b"hev1" {
            VideoCodec::HEVC
        } else if fourcc == b"avc1" {
            VideoCodec::AVC
        } else if fourcc == b"av01" {
            println!("⚠️ AV1 编码暂不支持");
            return None;
        } else {
            println!("⚠️ 未知 FourCC: {} ({:02x?})", fourcc_str, fourcc);
            return None;
        };

        let is_keyframe = frame_type_val == 1; // 1 = keyframe
        let frame_type = if is_keyframe {
            FrameType::Keyframe
        } else {
            FrameType::InterFrame
        };

        // Enhanced FLV packet types:
        // 0 = PacketTypeSequenceStart (类似 SequenceHeader)
        // 1 = PacketTypeCodedFrames
        // 2 = PacketTypeSequenceEnd
        // 3 = PacketTypeCodedFramesX (没有 CompositionTime)
        // 4 = PacketTypeMetadata
        // 5 = PacketTypeMPEG2TSSequenceStart

        match packet_type {
            0 => {
                // SequenceStart - 包含配置记录 (HVCC/AVCC)
                let config_data = tag.data[5..].to_vec();

                self.metadata.extradata = config_data.clone();
                self.metadata.video_codec = Some(codec);

                // 解析 HVCC 获取分辨率
                if codec == VideoCodec::HEVC {
                    if let Some((w, h)) = parse_hvcc_resolution(&config_data) {
                        self.metadata.width = w;
                        self.metadata.height = h;
                    }
                } else if let Some((w, h)) = parse_avcc_resolution(&config_data, codec) {
                    self.metadata.width = w;
                    self.metadata.height = h;
                }

                println!(
                    "📦 收到 Enhanced FLV {} 序列头: {} bytes, {}x{}, FourCC={}",
                    if codec == VideoCodec::HEVC {
                        "HEVC"
                    } else {
                        "AVC"
                    },
                    self.metadata.extradata.len(),
                    self.metadata.width,
                    self.metadata.height,
                    fourcc_str
                );

                Some(VideoFrame {
                    codec,
                    frame_type,
                    is_keyframe: true,
                    pts: tag.timestamp,
                    dts: tag.timestamp,
                    data: config_data,
                    is_sequence_header: true,
                })
            }
            1 => {
                // CodedFrames - 包含 CompositionTime (3 bytes) + 数据
                if tag.data.len() < 8 {
                    return None;
                }

                let composition_time =
                    i32::from_be_bytes([0, tag.data[5], tag.data[6], tag.data[7]]);
                let video_data = tag.data[8..].to_vec();

                let pts = tag.timestamp + composition_time as i64;
                let dts = tag.timestamp;

                Some(VideoFrame {
                    codec,
                    frame_type,
                    is_keyframe,
                    pts,
                    dts,
                    data: video_data,
                    is_sequence_header: false,
                })
            }
            3 => {
                // CodedFramesX - 没有 CompositionTime
                let video_data = tag.data[5..].to_vec();

                Some(VideoFrame {
                    codec,
                    frame_type,
                    is_keyframe,
                    pts: tag.timestamp,
                    dts: tag.timestamp,
                    data: video_data,
                    is_sequence_header: false,
                })
            }
            2 => {
                // SequenceEnd
                println!("📼 Enhanced FLV: 序列结束");
                None
            }
            _ => {
                println!("⚠️ Enhanced FLV: 未知 packet_type={}", packet_type);
                None
            }
        }
    }

    /// 解析 Script Data (onMetaData) - 使用 amf 库
    fn parse_script_data(&mut self, data: &[u8]) {
        let mut cursor = Cursor::new(data);

        // 读取第一个 AMF0 值 (通常是 "onMetaData" 字符串)
        let name = match amf::Value::read_from(&mut cursor, Version::Amf0) {
            Ok(v) => v,
            Err(e) => {
                println!("⚠️ AMF 解析失败 (name): {}", e);
                return;
            }
        };

        // 检查是否是 onMetaData
        let name_str = match &name {
            amf::Value::Amf0(Amf0Value::String(s)) => s.as_str(),
            _ => "",
        };

        if name_str != "onMetaData" {
            println!("📋 FLV Script: {} (跳过)", name_str);
            return;
        }

        // 读取元数据对象
        let metadata = match amf::Value::read_from(&mut cursor, Version::Amf0) {
            Ok(v) => v,
            Err(e) => {
                println!("⚠️ AMF 解析失败 (metadata): {}", e);
                return;
            }
        };

        // 提取值
        match &metadata {
            amf::Value::Amf0(Amf0Value::EcmaArray { entries })
            | amf::Value::Amf0(Amf0Value::Object { entries, .. }) => {
                for pair in entries {
                    match (pair.key.as_str(), &pair.value) {
                        ("width", Amf0Value::Number(n)) => {
                            self.metadata.width = *n as u32;
                        }
                        ("height", Amf0Value::Number(n)) => {
                            self.metadata.height = *n as u32;
                        }
                        ("framerate", Amf0Value::Number(n))
                        | ("videoframerate", Amf0Value::Number(n)) => {
                            self.metadata.fps = *n;
                        }
                        ("duration", Amf0Value::Number(n)) => {
                            println!("📊 FLV 时长: {:.2} 秒", n);
                        }
                        ("videocodecid", Amf0Value::Number(n)) => {
                            println!("📊 视频编码: {}", *n as u32);
                        }
                        ("audiocodecid", Amf0Value::Number(n)) => {
                            println!("📊 音频编码: {}", *n as u32);
                        }
                        _ => {}
                    }
                }
            }
            _ => {
                println!("⚠️ onMetaData 格式未知: {:?}", metadata);
            }
        }

        if self.metadata.width > 0 && self.metadata.height > 0 {
            println!(
                "📊 FLV Metadata: {}x{} @ {:.2} fps",
                self.metadata.width, self.metadata.height, self.metadata.fps
            );
        }
    }

    /// 清空缓冲区
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.position = 0;
        self.header_parsed = false;
    }
}

/// FLV Tag 结构
struct FlvTag {
    tag_type: TagType,
    timestamp: i64,
    data: Vec<u8>,
}

/// 从 AVCC extradata 解析分辨率
fn parse_avcc_resolution(data: &[u8], codec: VideoCodec) -> Option<(u32, u32)> {
    if codec == VideoCodec::AVC {
        // AVCC 格式: configurationVersion(1) + profile(1) + compatibility(1) + level(1) + ...
        if data.len() < 8 {
            println!("⚠️ AVCC extradata 太短: {} bytes", data.len());
            return None;
        }

        // 打印前几个字节帮助调试
        let preview: Vec<u8> = data.iter().take(16.min(data.len())).cloned().collect();
        println!(
            "🔍 AVCC extradata 前{}字节: {:02x?}",
            preview.len(),
            preview
        );

        // 跳过 AVCC header，找到 SPS
        let num_sps = data[5] & 0x1F;
        if num_sps == 0 {
            println!("⚠️ AVCC 没有 SPS");
            return None;
        }

        let sps_len = u16::from_be_bytes([data[6], data[7]]) as usize;
        println!("🔍 SPS 数量: {}, 第一个 SPS 长度: {}", num_sps, sps_len);

        if data.len() < 8 + sps_len {
            println!("⚠️ AVCC extradata 不足以包含 SPS");
            return None;
        }

        let sps = &data[8..8 + sps_len];
        let sps_preview: Vec<u8> = sps.iter().take(16.min(sps.len())).cloned().collect();
        println!(
            "🔍 SPS 数据前{}字节: {:02x?}",
            sps_preview.len(),
            sps_preview
        );

        parse_h264_sps_resolution(sps)
    } else if codec == VideoCodec::HEVC {
        parse_hvcc_resolution(data)
    } else {
        None
    }
}

/// 从 HVCC extradata 解析 HEVC 分辨率
fn parse_hvcc_resolution(data: &[u8]) -> Option<(u32, u32)> {
    // HVCC 格式比 AVCC 更复杂
    // https://github.com/niclet/HEVCBitstreamAnalyzer

    if data.len() < 23 {
        println!("⚠️ HVCC extradata 太短: {} bytes", data.len());
        return None;
    }

    let preview: Vec<u8> = data.iter().take(24.min(data.len())).cloned().collect();
    println!(
        "🔍 HVCC extradata 前{}字节: {:02x?}",
        preview.len(),
        preview
    );

    // HVCC header:
    // [0] configurationVersion
    // [1] general_profile_space (2) + general_tier_flag (1) + general_profile_idc (5)
    // [2-5] general_profile_compatibility_flags
    // [6-11] general_constraint_indicator_flags
    // [12] general_level_idc
    // [13-14] min_spatial_segmentation_idc (12 bits)
    // [15] parallelismType (2 bits)
    // [16] chroma_format_idc (2 bits)
    // [17] bit_depth_luma_minus8 (3 bits)
    // [18] bit_depth_chroma_minus8 (3 bits)
    // [19-20] avgFrameRate
    // [21] constantFrameRate (2) + numTemporalLayers (3) + temporalIdNested (1) + lengthSizeMinusOne (2)
    // [22] numOfArrays

    let num_arrays = data[22];
    let mut pos = 23;

    // 遍历 NAL 单元数组，找到 SPS
    for _ in 0..num_arrays {
        if pos + 3 > data.len() {
            break;
        }

        let nal_type = data[pos] & 0x3F;
        let num_nalus = u16::from_be_bytes([data[pos + 1], data[pos + 2]]) as usize;
        pos += 3;

        for _ in 0..num_nalus {
            if pos + 2 > data.len() {
                break;
            }

            let nal_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;

            if pos + nal_len > data.len() {
                break;
            }

            // SPS NAL type = 33
            if nal_type == 33 && nal_len > 2 {
                let sps_data = &data[pos..pos + nal_len];
                if let Some(res) = parse_hevc_sps_resolution(sps_data) {
                    return Some(res);
                }
            }

            pos += nal_len;
        }
    }

    println!("⚠️ HVCC 未找到 SPS");
    None
}

/// 解析 HEVC SPS 获取分辨率 (简化版)
fn parse_hevc_sps_resolution(sps: &[u8]) -> Option<(u32, u32)> {
    // HEVC SPS 解析比较复杂
    // NAL header: 2 bytes
    if sps.len() < 15 {
        println!("⚠️ HEVC SPS 太短: {} bytes", sps.len());
        return None;
    }

    let sps_preview: Vec<u8> = sps.iter().take(20.min(sps.len())).cloned().collect();
    println!("🔍 HEVC SPS 数据: {:02x?}", sps_preview);

    // 移除 emulation prevention bytes (00 00 03 -> 00 00)
    let rbsp = remove_emulation_prevention(&sps[2..]); // 跳过 2 字节 NAL header

    let mut reader = BitReader::new(&rbsp);

    // sps_video_parameter_set_id (4 bits)
    reader.read_bits(4)?;
    // sps_max_sub_layers_minus1 (3 bits)
    let max_sub_layers = reader.read_bits(3)? as usize;
    // sps_temporal_id_nesting_flag (1 bit)
    reader.read_bits(1)?;

    // profile_tier_level(sps_max_sub_layers_minus1)
    // general_profile_space (2) + general_tier_flag (1) + general_profile_idc (5)
    reader.read_bits(8)?;
    // general_profile_compatibility_flag[32]
    reader.read_bits(32)?;
    // general_progressive_source_flag + general_interlaced_source_flag +
    // general_non_packed_constraint_flag + general_frame_only_constraint_flag +
    // general_max_12bit_constraint_flag + ... (48 bits total for general constraints)
    reader.read_bits(32)?;
    reader.read_bits(16)?;
    // general_level_idc (8)
    reader.read_bits(8)?;

    // sub_layer_profile_present_flag[i] and sub_layer_level_present_flag[i]
    // 每个 sub layer 2 bits
    if max_sub_layers > 0 {
        for _ in 0..max_sub_layers {
            reader.read_bits(2)?; // profile_present + level_present flags
        }
        // 填充到 8 的倍数
        if max_sub_layers < 8 {
            for _ in max_sub_layers..8 {
                reader.read_bits(2)?; // reserved_zero_2bits
            }
        }
    }

    // 对于每个 sub_layer (如果 profile/level present)
    // 这里简化跳过，因为通常 max_sub_layers = 0 或 1

    // sps_seq_parameter_set_id
    reader.read_exp_golomb()?;
    // chroma_format_idc
    let chroma_format = reader.read_exp_golomb()?;
    if chroma_format == 3 {
        reader.read_bits(1)?; // separate_colour_plane_flag
    }

    // pic_width_in_luma_samples
    let width = reader.read_exp_golomb()?;
    // pic_height_in_luma_samples
    let height = reader.read_exp_golomb()?;

    if width > 0 && height > 0 && width < 16384 && height < 16384 {
        println!("📐 HEVC SPS 解析成功: {}x{}", width, height);
        Some((width, height))
    } else {
        println!("⚠️ HEVC SPS 解析得到无效分辨率: {}x{}", width, height);
        None
    }
}

/// 移除 emulation prevention bytes (00 00 03 -> 00 00)
/// H.264/HEVC 码流中为了避免起始码冲突，会在 00 00 后插入 03
fn remove_emulation_prevention(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        // 检测 00 00 03 序列
        if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            // 写入 00 00，跳过 03
            result.push(0x00);
            result.push(0x00);
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    result
}

/// H.264 SPS 分辨率解析 (使用 Exp-Golomb)
fn parse_h264_sps_resolution(sps: &[u8]) -> Option<(u32, u32)> {
    if sps.len() < 4 {
        return None;
    }

    // 跳过 NAL header (1 byte)
    let nal_type = sps[0] & 0x1F;
    if nal_type != 7 {
        return None;
    }

    // 使用 BitReader 解析
    let mut reader = BitReader::new(&sps[1..]);

    // profile_idc
    let profile_idc = reader.read_bits(8)?;
    // constraint_set_flags + reserved
    reader.read_bits(8)?;
    // level_idc
    reader.read_bits(8)?;
    // seq_parameter_set_id
    reader.read_exp_golomb()?;

    // 高级 profile 有更多字段
    if profile_idc == 100
        || profile_idc == 110
        || profile_idc == 122
        || profile_idc == 244
        || profile_idc == 44
        || profile_idc == 83
        || profile_idc == 86
        || profile_idc == 118
        || profile_idc == 128
    {
        // chroma_format_idc
        let chroma_format = reader.read_exp_golomb()?;
        if chroma_format == 3 {
            reader.read_bits(1)?; // separate_colour_plane_flag
        }
        reader.read_exp_golomb()?; // bit_depth_luma
        reader.read_exp_golomb()?; // bit_depth_chroma
        reader.read_bits(1)?; // qpprime_y_zero_transform_bypass_flag

        let scaling_matrix_present = reader.read_bits(1)?;
        if scaling_matrix_present == 1 {
            let count = if chroma_format != 3 { 8 } else { 12 };
            for _ in 0..count {
                let present = reader.read_bits(1)?;
                if present == 1 {
                    // 跳过 scaling list
                    let size = if count <= 6 { 16 } else { 64 };
                    for _ in 0..size {
                        reader.read_exp_golomb_signed()?;
                    }
                }
            }
        }
    }

    // log2_max_frame_num
    reader.read_exp_golomb()?;
    // pic_order_cnt_type
    let poc_type = reader.read_exp_golomb()?;
    if poc_type == 0 {
        reader.read_exp_golomb()?;
    } else if poc_type == 1 {
        reader.read_bits(1)?;
        reader.read_exp_golomb_signed()?;
        reader.read_exp_golomb_signed()?;
        let num = reader.read_exp_golomb()?;
        for _ in 0..num {
            reader.read_exp_golomb_signed()?;
        }
    }

    // max_num_ref_frames
    reader.read_exp_golomb()?;
    // gaps_in_frame_num_value_allowed_flag
    reader.read_bits(1)?;

    // pic_width_in_mbs_minus1
    let width_mbs = reader.read_exp_golomb()? + 1;
    // pic_height_in_map_units_minus1
    let height_mbs = reader.read_exp_golomb()? + 1;

    // frame_mbs_only_flag
    let frame_mbs_only = reader.read_bits(1)?;
    if frame_mbs_only == 0 {
        reader.read_bits(1)?; // mb_adaptive_frame_field_flag
    }

    // direct_8x8_inference_flag
    reader.read_bits(1)?;

    // frame_cropping_flag
    let crop_flag = reader.read_bits(1)?;
    let (crop_left, crop_right, crop_top, crop_bottom) = if crop_flag == 1 {
        (
            reader.read_exp_golomb()?,
            reader.read_exp_golomb()?,
            reader.read_exp_golomb()?,
            reader.read_exp_golomb()?,
        )
    } else {
        (0, 0, 0, 0)
    };

    // 计算实际分辨率
    let width = width_mbs * 16 - crop_left * 2 - crop_right * 2;
    let height = (2 - frame_mbs_only as u32) * height_mbs * 16 - crop_top * 2 - crop_bottom * 2;

    println!(
        "📐 SPS 解析: {}x{} (mbs: {}x{}, crop: {}/{}/{}/{})",
        width, height, width_mbs, height_mbs, crop_left, crop_right, crop_top, crop_bottom
    );

    Some((width, height))
}

/// 简单的位读取器
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, n: u8) -> Option<u32> {
        let mut result = 0u32;
        for _ in 0..n {
            if self.byte_pos >= self.data.len() {
                return None;
            }
            let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
            result = (result << 1) | bit as u32;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Some(result)
    }

    fn read_exp_golomb(&mut self) -> Option<u32> {
        let mut leading_zeros = 0u8;
        while self.read_bits(1)? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return None;
            }
        }
        if leading_zeros == 0 {
            return Some(0);
        }
        let suffix = self.read_bits(leading_zeros)?;
        Some((1 << leading_zeros) - 1 + suffix)
    }

    fn read_exp_golomb_signed(&mut self) -> Option<i32> {
        let v = self.read_exp_golomb()?;
        if v == 0 {
            Some(0)
        } else if v & 1 == 1 {
            Some(((v + 1) / 2) as i32)
        } else {
            Some(-((v / 2) as i32))
        }
    }
}

/// AVCC 转 Annex-B
pub fn avcc_to_annexb(data: &[u8], nal_length_size: usize) -> Vec<u8> {
    // 首先检查是否已经是 Annex-B 格式
    if data.len() >= 4
        && ((data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
            || (data[0] == 0 && data[1] == 0 && data[2] == 1))
    {
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(data.len() + 64);
    let mut i = 0;
    let mut nal_count = 0;

    while i + nal_length_size <= data.len() {
        // 读取 NAL 长度
        let nal_len = match nal_length_size {
            4 => u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize,
            3 => u32::from_be_bytes([0, data[i], data[i + 1], data[i + 2]]) as usize,
            2 => u16::from_be_bytes([data[i], data[i + 1]]) as usize,
            1 => data[i] as usize,
            _ => break,
        };

        i += nal_length_size;

        // 检查 NAL 长度是否合理
        if nal_len == 0 {
            // 可能是填充，跳过
            continue;
        }

        if nal_len > data.len() - i || nal_len > 10 * 1024 * 1024 {
            // NAL 长度不合理，可能格式有问题
            break;
        }

        // 添加 Annex-B 起始码
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&data[i..i + nal_len]);

        i += nal_len;
        nal_count += 1;
    }

    // 如果没解析出任何 NAL，可能数据格式不对，直接返回原数据
    if result.is_empty() && !data.is_empty() {
        // 尝试以原始数据返回 (加上起始码)
        let mut fallback = Vec::with_capacity(data.len() + 4);
        fallback.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        fallback.extend_from_slice(data);
        return fallback;
    }

    result
}

/// AVCC extradata 转 Annex-B (SPS/PPS)
pub fn avcc_extradata_to_annexb(extradata: &[u8]) -> Vec<u8> {
    if extradata.len() < 7 {
        return Vec::new();
    }

    let mut result = Vec::new();

    // AVCC 格式:
    // [0] configurationVersion = 1
    // [1] AVCProfileIndication
    // [2] profile_compatibility
    // [3] AVCLevelIndication
    // [4] lengthSizeMinusOne (NAL length size - 1, 低 2 位)
    // [5] numOfSequenceParameterSets (低 5 位)

    let num_sps = extradata[5] & 0x1F;

    let mut pos = 6;

    // 读取 SPS
    for _ in 0..num_sps {
        if pos + 2 > extradata.len() {
            break;
        }
        let sps_len = u16::from_be_bytes([extradata[pos], extradata[pos + 1]]) as usize;
        pos += 2;

        if pos + sps_len > extradata.len() {
            break;
        }

        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        result.extend_from_slice(&extradata[pos..pos + sps_len]);
        pos += sps_len;
    }

    // 读取 PPS
    if pos < extradata.len() {
        let num_pps = extradata[pos];
        pos += 1;

        for _ in 0..num_pps {
            if pos + 2 > extradata.len() {
                break;
            }
            let pps_len = u16::from_be_bytes([extradata[pos], extradata[pos + 1]]) as usize;
            pos += 2;

            if pos + pps_len > extradata.len() {
                break;
            }

            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend_from_slice(&extradata[pos..pos + pps_len]);
            pos += pps_len;
        }
    }

    result
}

/// HVCC extradata 转 Annex-B (VPS/SPS/PPS)
pub fn hvcc_extradata_to_annexb(extradata: &[u8]) -> Vec<u8> {
    if extradata.len() < 23 {
        return Vec::new();
    }

    let mut result = Vec::new();

    // HVCC header 长度为 22 bytes + numOfArrays
    // [22] numOfArrays
    let num_arrays = extradata[22];
    let mut pos = 23;

    for _ in 0..num_arrays {
        if pos + 3 > extradata.len() {
            break;
        }

        // array_completeness (1) + reserved (1) + NAL_unit_type (6)
        let _nal_type = extradata[pos] & 0x3F;
        let num_nalus = u16::from_be_bytes([extradata[pos + 1], extradata[pos + 2]]) as usize;
        pos += 3;

        for _ in 0..num_nalus {
            if pos + 2 > extradata.len() {
                break;
            }

            let nal_len = u16::from_be_bytes([extradata[pos], extradata[pos + 1]]) as usize;
            pos += 2;

            if pos + nal_len > extradata.len() {
                break;
            }

            // 添加 Annex-B 起始码
            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend_from_slice(&extradata[pos..pos + nal_len]);
            pos += nal_len;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avcc_to_annexb() {
        // 模拟 AVCC 数据: [长度(4)][NAL数据]
        let avcc = vec![
            0x00, 0x00, 0x00, 0x05, // 长度 = 5
            0x67, 0x64, 0x00, 0x1f, 0x00, // NAL 数据
        ];

        let annexb = avcc_to_annexb(&avcc, 4);
        assert_eq!(&annexb[0..4], &[0x00, 0x00, 0x00, 0x01]);
        assert_eq!(&annexb[4..], &[0x67, 0x64, 0x00, 0x1f, 0x00]);
    }
}
