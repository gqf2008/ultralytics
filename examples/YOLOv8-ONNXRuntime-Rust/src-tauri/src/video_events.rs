//! 视频管道事件定义
//!
//! 用于 Demuxer、Decoder、LlmInference 之间通过 xbus 通信

use std::sync::Arc;

/// 流元数据事件 - Demuxer 发布
#[derive(Debug, Clone)]
pub struct StreamMetadataEvent {
    /// 编码格式: "h264" 或 "hevc"
    pub codec: String,
    /// 视频宽度
    pub width: u32,
    /// 视频高度
    pub height: u32,
    /// 帧率
    pub fps: f64,
}

/// 编码数据包事件 - Demuxer 发布, Decoder 订阅
#[derive(Debug, Clone)]
pub struct EncodedPacketEvent {
    /// 数据包 ID
    pub packet_id: u64,
    /// 编码数据 (Annex-B 格式)
    pub data: Arc<Vec<u8>>,
    /// 是否关键帧
    pub is_keyframe: bool,
    /// PTS (显示时间戳)
    pub pts: i64,
    /// DTS (解码时间戳)
    pub dts: i64,
}

/// 解码帧事件 - Decoder 发布, LlmInference/Frontend 订阅
///
/// 帧数据存放在共享内存中，这里只传递元信息
#[derive(Debug, Clone)]
pub struct DecodedFrameEvent {
    /// 帧 ID
    pub frame_id: u64,
    /// 宽度
    pub width: u32,
    /// 高度
    pub height: u32,
    /// Y 平面步长
    pub y_stride: u32,
    /// UV 平面步长
    pub uv_stride: u32,
    /// 时间戳 (毫秒)
    pub timestamp: u64,
    /// 共享内存中的缓冲区索引 (双缓冲: 0 或 1)
    pub buffer_index: u32,
    /// 共享内存名称
    pub shm_name: String,
}

/// 流开始事件
#[derive(Debug, Clone)]
pub struct StreamStartedEvent {
    pub url: String,
}

/// 流停止事件
#[derive(Debug, Clone)]
pub struct StreamStoppedEvent {
    pub reason: String,
}

/// 流错误事件
#[derive(Debug, Clone)]
pub struct StreamErrorEvent {
    pub message: String,
}

/// RGBA 帧就绪事件 - 通知前端可以读取共享内存
#[derive(Debug, Clone)]
pub struct RgbaFrameReadyEvent {
    /// 帧 ID
    pub frame_id: u64,
    /// 宽度
    pub width: u32,
    /// 高度
    pub height: u32,
    /// 时间戳 (毫秒)
    pub timestamp: u64,
    /// 共享内存名称
    pub shm_name: String,
    /// 数据偏移量
    pub offset: usize,
    /// 数据大小
    pub size: usize,
}
