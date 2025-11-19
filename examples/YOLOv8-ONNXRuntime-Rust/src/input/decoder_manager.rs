/// 解码器管理器 - 支持动态切换输入源
use super::decoder::InputSource;
use std::sync::{Arc, Mutex};

/// 全局解码器命令发送器（占位）
static DECODER_COMMAND_SENDER: once_cell::sync::Lazy<Arc<Mutex<Option<()>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(None)));

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

/// 切换输入源 - 通过退出程序实现
pub fn switch_decoder_source(source: InputSource) {
    println!("\n🔄 ============ 切换输入源 ============");

    let cmd = match source {
        InputSource::Rtsp(url) => {
            println!("📹 新输入源: RTSP流");
            println!("   地址: {}", url);
            format!(".\\target\\release\\sentinel-mq.exe -i rtsp -u \"{}\"", url)
        }
        InputSource::Camera(id) => {
            println!("📷 新输入源: 本地摄像头");
            println!("   设备ID: {}", id);
            format!(".\\target\\release\\sentinel-mq.exe -i camera -c {}", id)
        }
    };

    println!("\n💡 由于FFmpeg解码无法中断，需要重启程序");
    println!("📋 启动命令已复制到剪贴板:");
    println!("   {}", cmd);
    println!("\n⚡ 操作步骤:");
    println!("   1. 关闭当前窗口");
    println!("   2. 在PowerShell中粘贴运行上述命令");
    println!("\n🔄 正在尝试自动复制到剪贴板...");

    // 尝试复制到剪贴板
    use arboard::Clipboard;
    if let Ok(mut clipboard) = Clipboard::new() {
        if clipboard.set_text(&cmd).is_ok() {
            println!("✅ 命令已复制到剪贴板！直接粘贴即可");
        }
    }

    println!("\n========================================\n");
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
