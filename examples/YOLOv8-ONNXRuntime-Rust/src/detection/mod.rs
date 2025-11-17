/// 检测系统 (Detection System)
///
/// 独立工作线程,负责智能分析
/// - Detector: 目标检测
/// - Pose:     姿态估计
/// - Tracker:  目标追踪
pub mod detector;

pub use detector::Detector;
