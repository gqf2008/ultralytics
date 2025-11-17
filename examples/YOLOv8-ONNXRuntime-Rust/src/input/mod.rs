/// 视频输入系统 (Video Input System)
///
/// 独立工作线程,负责视频流解码与预处理
/// - Decoder: 视频解码器
/// - Filter:  帧过滤与预处理
pub mod decode_filter;
pub mod decoder;

pub use decode_filter::DecodeFilter;
pub use decoder::{adaptive_decode, Decoder};
