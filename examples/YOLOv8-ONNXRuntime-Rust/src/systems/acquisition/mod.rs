/// 采集系统 (Acquisition System)
/// 
/// 独立工作线程,负责视频流采集与预处理
/// - Decoder: 视频解码
/// - Filter:  帧过滤与预处理

pub mod decoder;

pub use decoder::Decoder;
