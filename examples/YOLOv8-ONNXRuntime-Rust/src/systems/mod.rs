/// 系统架构 (System Architecture)
/// 
/// 两个独立工作线程 + 一个主渲染线程:
/// - Acquisition System: 视频采集系统 (独立线程)
/// - Detection System:   智能检测系统 (独立线程)
/// - Sentinel:           数字卫兵主程序 (主线程/渲染)

pub mod acquisition;
pub mod detection;

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
