/// 视频输入系统 (Video Input System)
///
/// 独立工作线程,负责视频流解码与预处理
/// - QsvRtspDecoder: QSV硬件加速RTSP解码器 (双线程架构,Rust风格封装)
/// - CameraInput: 摄像头输入解码器 (ffmpeg-next)
/// - DesktopInput: 桌面捕获解码器 (ffmpeg-next)
/// - DecoderManager: 解码器管理器 (支持动态热切换)
pub mod camera_input;
pub mod decoder_manager;
pub mod desktop_input;
pub mod qsv_rtsp;

pub use camera_input::CameraInput;
pub use decoder_manager::{
    should_stop, switch_decoder_source, DecoderManager, InputSource, ACTIVE_DECODER_GENERATION,
};
pub use desktop_input::DesktopInput;
pub use qsv_rtsp::QsvRtspDecoder;
