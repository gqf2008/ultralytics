/// RTSP实时检测模块
/// RTSP Real-time Detection Module
///
/// 三线程架构(通过xbus通信):
/// 1. 解码线程: FFmpeg RTSP解码 → 发送DecodedFrame
/// 2. 检测线程: 接收DecodedFrame → 检测+跟踪 → 发送DetectionResult
/// 3. 渲染线程: 接收DecodedFrame + DetectionResult → GPU渲染
pub mod bytetrack;
pub mod decode_filter;
pub mod decoder;
pub mod inference;
pub mod renderer;
pub mod tracker;
pub mod types;

// ========== 公共常量 ==========

pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;
pub const INF_SIZE: u32 = 320; // YOLOv8推理输入尺寸

// ========== 重新导出常用类型 ==========

pub use bytetrack::{ByteTrackedPerson, ByteTracker};
pub use decode_filter::DecodeFilter;
pub use decoder::adaptive_decode;
pub use inference::inference_thread;
pub use renderer::{TrackerType, YoloApp};
pub use tracker::{PersonTracker, TrackPoint, TrackedPerson};
// 导出旧架构的类型(用于yolov8-rtsp程序)
pub use types::{BBox, DecodedFrame, InferredFrame, PoseKeypoints, ResizedFrame};
