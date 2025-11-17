/// 视频处理流水线 (Video Processing Pipeline)
///
/// 三线程架构,通过XBus消息总线通信:
/// - Decoder:  视频解码 (独立线程)
/// - Detector: 目标检测 (独立线程)
/// - Renderer: 渲染显示 (主线程)
pub mod decoder;
pub mod detector;
pub mod renderer;

use crate::rtsp::types;

// ========== XBus消息类型定义 ==========

/// 解码帧 (解码模块 → 渲染模块 + 检测模块)
#[derive(Clone, Debug)]
pub struct DecodedFrame {
    pub rgba_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frame_id: u64, // 帧序号
    pub decode_fps: f64,
    pub decoder_name: String,
}

/// 检测结果 (检测模块 → 渲染模块)
#[derive(Clone, Debug)]
pub struct DetectionResult {
    pub frame_id: u64, // 对应的帧序号
    pub bboxes: Vec<types::BBox>,
    pub keypoints: Vec<types::PoseKeypoints>,
    pub inference_fps: f64,
    pub inference_ms: f64,
}

/// 渲染统计 (渲染模块 → 其他模块,用于监控)
#[derive(Clone, Debug)]
pub struct RenderStats {
    pub render_fps: f64,
    pub frame_count: u64,
}

/// 系统控制
#[derive(Clone, Debug)]
pub enum SystemControl {
    PauseDecode,
    ResumeDecode,
    Shutdown,
    SwitchTracker(String),
}

/// 错误消息
#[derive(Clone, Debug)]
pub struct Error {
    pub module: String,
    pub error: String,
}
