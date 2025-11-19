/// 解码器管理器 - 支持动态切换输入源
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

/// 视频设备信息
#[derive(Debug, Clone)]
pub struct VideoDevice {
    pub name: String,
    pub index: usize,
}

/// 解码器管理器（简化版）
pub struct DecoderManager;

impl DecoderManager {
    pub fn new(_initial_source: InputSource) -> Self {
        Self
    }
}

/// 切换输入源 - 在新线程中启动解码器
pub fn switch_decoder_source(source: InputSource, preference: super::decoder::DecoderPreference) {
    println!("\n🔄 ============ 切换输入源 ============");

    use super::{CameraDecoder, Decoder, DesktopDecoder};
    use std::thread;

    // 1. 增加代数ID，使旧解码器失效
    let new_gen = ACTIVE_DECODER_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    println!("🔄 切换解码器代数: {} -> {}", new_gen - 1, new_gen);

    match source {
        InputSource::Rtsp(url) => {
            println!("📹 新输入源: RTSP流");
            println!("   地址: {}", url);

            thread::spawn(move || {
                // 等待旧解码器退出
                std::thread::sleep(std::time::Duration::from_millis(500));
                let mut decoder = Decoder::new(url, new_gen, preference);
                decoder.run();
            });
        }
        InputSource::Camera(index, name) => {
            println!("📷 新输入源: 本地摄像头");
            println!("   设备索引: {}", index);
            println!("   设备名称: {}", name);

            thread::spawn(move || {
                // 等待旧解码器退出 (摄像头释放需要更多时间)
                std::thread::sleep(std::time::Duration::from_millis(1000));
                let mut camera = CameraDecoder::new(index, name, new_gen);
                camera.run();
            });
        }
        InputSource::Desktop => {
            println!("🖥️ 新输入源: 桌面捕获");

            thread::spawn(move || {
                // 等待旧解码器退出
                std::thread::sleep(std::time::Duration::from_millis(500));
                let mut desktop = DesktopDecoder::new(new_gen);
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

/// 获取可用的视频设备列表
pub fn get_video_devices() -> Vec<VideoDevice> {
    println!("🔍 正在扫描视频设备...");

    match ez_ffmpeg::device::get_input_video_devices() {
        Ok(devices) => {
            println!("✅ 找到 {} 个视频设备", devices.len());
            devices
                .into_iter()
                .enumerate()
                .map(|(index, name)| {
                    println!("   [{}] {}", index, name);
                    VideoDevice { name, index }
                })
                .collect()
        }
        Err(e) => {
            println!("⚠️  获取设备列表失败: {}", e);
            // 返回默认设备
            vec![VideoDevice {
                name: "默认摄像头".to_string(),
                index: 0,
            }]
        }
    }
}
