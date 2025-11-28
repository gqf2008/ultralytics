//! WGC 捕获测试
//!
//! 用法: cargo run --bin wgc_test -- <窗口名称关键字>

use std::thread;
use std::time::Duration;

use xcap::Window;
use yolov8_sentinel_lib::wgc_capture;

fn main() {
    // 检查 WGC 是否可用
    let supported = wgc_capture::is_wgc_supported();
    println!("WGC 支持: {}", supported);

    if !supported {
        eprintln!("❌ 当前系统不支持 WGC (需要 Windows 10 1903+)");
        return;
    }

    // 获取命令行参数
    let args: Vec<String> = std::env::args().collect();
    let keyword = args.get(1).map(|s| s.as_str()).unwrap_or("Chrome");

    println!("\n🔍 搜索包含 '{}' 的窗口...", keyword);

    // 查找窗口
    let windows = Window::all().unwrap_or_default();
    let target = windows.iter().find(|w| {
        let title = w.title().unwrap_or_default();
        !title.is_empty() && title.to_lowercase().contains(&keyword.to_lowercase())
    });

    let Some(window) = target else {
        eprintln!("❌ 未找到匹配的窗口");
        println!("\n可用窗口:");
        for w in &windows {
            let title = w.title().unwrap_or_default();
            if !title.is_empty() {
                println!("  - {} (HWND: {})", title, w.id().unwrap_or(0));
            }
        }
        return;
    };

    let hwnd = window.id().unwrap_or(0) as isize;
    let title = window.title().unwrap_or_default();
    println!("✅ 找到窗口: {} (HWND: {})", title, hwnd);

    // 创建 WGC 捕获器
    println!("\n📸 创建 WGC 捕获器...");
    let capture = match wgc_capture::WgcCapture::new(hwnd) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ 创建 WGC 捕获器失败: {:?}", e);
            return;
        }
    };

    // 启动捕获
    if let Err(e) = capture.start() {
        eprintln!("❌ 启动捕获失败: {:?}", e);
        return;
    }

    println!("\n⏳ 等待帧数据 (5 秒)...");
    println!("💡 提示: 尝试用其他窗口遮挡目标窗口，观察是否仍能捕获");

    // 等待并获取帧
    for i in 0..50 {
        thread::sleep(Duration::from_millis(100));

        // 使用轮询模式获取帧
        if let Some(frame) = capture.try_get_frame() {
            println!(
                "📷 帧 #{}: {}x{}, 数据大小: {} bytes, 时间戳: {}",
                capture.get_frame_id(),
                frame.width,
                frame.height,
                frame.data.len(),
                frame.timestamp
            );

            // 第一帧保存为文件验证
            if i == 10 {
                let path = "wgc_test_frame.rgba";
                if std::fs::write(path, &frame.data).is_ok() {
                    println!("💾 首帧已保存到 {} (RGBA 格式)", path);
                }
            }
        } else {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!("\n\n✅ 测试完成");
    capture.stop();
}
