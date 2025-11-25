/// 解码器管理器 - 支持所有输入源 (使用 ffmpeg-next)
use std::sync::atomic::{AtomicUsize, Ordering};

/// 全局活跃解码器代数ID (用于平滑切换)
pub static ACTIVE_DECODER_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// 输入源类型
#[derive(Debug, Clone)]
pub enum InputSource {
    Rtsp(String),          // RTSP流
    Camera(usize, String), // 本地摄像头 (索引, 名称)
    Desktop,               // 桌面捕获
}

/// 解码器管理器（简化版）
pub struct DecoderManager;

impl DecoderManager {
    pub fn new(_initial_source: InputSource) -> Self {
        Self
    }
}

/// 切换输入源 - 在新线程中启动解码器
pub fn switch_decoder_source(source: InputSource) {
    println!("\n🔄 ============ 切换输入源 ============");

    use super::{CameraInput, DesktopInput, QsvRtspDecoder};
    use std::thread;

    // 1. 增加代数ID，使旧解码器失效
    let new_gen = ACTIVE_DECODER_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    println!("🔄 切换解码器代数: {} -> {}", new_gen - 1, new_gen);

    match source {
        InputSource::Rtsp(url) => {
            println!("📹 新输入源: RTSP流");
            println!("   地址: {}", url);
            println!("⚡ 使用 QSV RTSP 解码器 (ffmpeg-next 封装,双线程)");

            thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let mut decoder = QsvRtspDecoder::new(url, new_gen);
                decoder.run();
            });
        }
        InputSource::Camera(index, name) => {
            println!("📷 新输入源: 本地摄像头");
            println!("   设备索引: {}", index);
            println!("   设备名称: {}", name);

            thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(1000));
                let mut camera = CameraInput::new(index, name, new_gen);
                camera.run();
            });
        }
        InputSource::Desktop => {
            println!("🖥️ 新输入源: 桌面捕获");

            thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let mut desktop = DesktopInput::new(new_gen);
                desktop.run();
            });
        }
    }

    println!("✅ 解码器已在后台线程启动");
    println!("========================================\n");
}

pub fn should_stop() -> bool {
    false // 占位函数
}
