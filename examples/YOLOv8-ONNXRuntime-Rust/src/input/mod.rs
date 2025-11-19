/// 视频输入系统 (Video Input System)
///
/// 独立工作线程,负责视频流解码与预处理
/// - Decoder: RTSP主动拉流解码器 (VLC级别画质优化)
/// - CameraDecoder: 本地摄像头解码器 (DirectShow/AVFoundation/V4L2)
/// - Filter:  帧过滤与预处理
/// - DecoderManager: 解码器管理器 (支持动态热切换)
pub mod decode_filter;
pub mod decoder;
pub mod camera;
pub mod desktop;
pub mod decoder_manager;

pub use decode_filter::DecodeFilter;
pub use decoder::{adaptive_decode, Decoder};
pub use camera::{CameraDecoder, get_camera_devices};
pub use desktop::DesktopDecoder;
pub use decoder_manager::{get_video_devices, switch_decoder_source, should_stop, DecoderManager, VideoDevice, InputSource};
