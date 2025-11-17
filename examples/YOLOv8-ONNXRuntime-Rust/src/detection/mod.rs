//! 检测系统 (Detection System)
//!
//! 独立工作线程,负责智能分析
//! - Detector: 目标检测
//! - Tracker:  目标追踪

pub mod bytetrack;
pub mod deepsort;
pub mod detector;
pub mod tracker;
pub mod types;

// Re-exports
pub use bytetrack::{ByteTrackedPerson, ByteTracker};
pub use deepsort::{PersonTracker, TrackedPerson};
pub use detector::Detector;
pub use tracker::{compute_iou, id_to_color, KalmanBoxFilter, TrackPoint, TrackedObject, Tracker};
pub use types::{
    BBox, DecodedFrame, InferredFrame, PoseKeypoints, ResizedFrame, TrackerType, INF_SIZE,
};
