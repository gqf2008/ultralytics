/// 视频输入系统 (Video Input System)
///
/// 独立工作线程,负责视频流解码与预处理
/// - Decoder: RTSP主动拉流解码器 (VLC级别画质优化)
/// - RtmpDecoder: RTMP被动接收推流解码器 (嵌入式服务器)
/// - Filter:  帧过滤与预处理
/// - DecoderManager: 解码器管理器 (支持动态热切换)
pub mod decode_filter;
pub mod decoder;
pub mod decoder_manager;

pub use decode_filter::DecodeFilter;
pub use decoder::{adaptive_decode, Decoder, InputSource};
pub use decoder_manager::{get_video_devices, switch_decoder_source, should_stop, DecoderManager, VideoDevice};
