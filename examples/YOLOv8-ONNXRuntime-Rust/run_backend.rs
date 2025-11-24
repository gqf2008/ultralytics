// 独立运行的后端服务(不依赖 Tauri 窗口)
// 使用方法: cargo run --bin backend --no-default-features

use std::thread;
use std::time::Duration;

fn main() {
    println!("🚀 YOLOv8 后端服务启动");
    println!("📡 请在浏览器中打开: http://localhost:5173");
    println!("");

    // 启动 RTSP 解码器
    let rtsp_url = "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1&subtype=0";
    println!("🎬 启动 RTSP 解码: {}", rtsp_url);

    let mut decoder = yolov8_rs::input::Decoder::new(
        rtsp_url.to_string(),
        0,
        yolov8_rs::input::decoder::DecoderPreference::Software,
    );

    println!("✅ 解码器已创建,开始接收帧...");
    println!("💡 提示: 前端需要单独运行 'npm run dev'");
    println!("");

    decoder.run();

    println!("⏹️ 解码器已停止");
}
