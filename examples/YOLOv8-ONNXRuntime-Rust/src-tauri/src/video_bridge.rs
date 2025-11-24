/// 视频解码桥接模块 - 连接 yolov8-rs 和 Tauri
use std::sync::Arc;
use std::thread;
use tauri::{AppHandle, Emitter, Manager};
use yolov8_rs::detection::types::DecodedFrame;
use yolov8_rs::input::decoder::{Decoder, DecoderPreference};
use yolov8_rs::xbus;

/// 启动 RTSP 视频流解码线程
pub fn start_rtsp_decoder(app: AppHandle, rtsp_url: String, generation: usize) {
    thread::Builder::new()
        .name(format!("rtsp-decoder-{}", generation))
        .spawn(move || {
            println!("🎬 启动 RTSP 解码线程 Gen:{}", generation);

            // 订阅解码帧 - 必须在解码器运行前创建,且持有订阅直到线程结束
            let app_clone = app.clone();
            let subscription = xbus::subscribe::<DecodedFrame, _>(move |frame| {
                println!("📦 收到解码帧: {}x{} ({} bytes)", frame.width, frame.height, frame.rgba_data.len());
                
                // 更新 Tauri 状态
                if let Err(e) = update_frame_state(
                    &app_clone,
                    frame.rgba_data.clone(),
                    frame.width,
                    frame.height,
                ) {
                    eprintln!("❌ 更新帧状态失败: {}", e);
                } else {
                    println!("✅ 帧已更新到 Tauri State");
                }
            });

            println!("✅ xbus 订阅已创建");

            // 启动 FFmpeg 解码器 (阻塞式运行)
            let mut decoder = Decoder::new(rtsp_url, generation, DecoderPreference::Software);
            decoder.run();

            // 保持订阅存活直到解码器退出
            drop(subscription);
            println!("✅ RTSP 解码线程退出");
        })
        .expect("Failed to spawn decoder thread");
}

/// 更新帧状态并通知前端
fn update_frame_state(
    app: &AppHandle,
    rgba: Arc<Vec<u8>>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let state = app.state::<crate::VideoFrameState>();

    // 获取 Arc 内部数据的引用 (零拷贝)
    let buffer = Arc::try_unwrap(rgba).unwrap_or_else(|arc| (*arc).clone());
    
    println!("📝 写入帧数据到 State: {}x{}, {} bytes", width, height, buffer.len());

    *state.buffer.write() = buffer;
    *state.width.write() = width;
    *state.height.write() = height;
    *state.updated.write() = true;

    // 发送事件到前端
    println!("📤 发送 frame-ready 事件到前端");
    app.emit(
        "frame-ready",
        crate::FrameData {
            width,
            height,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        },
    )
    .map_err(|e| format!("事件发送失败: {}", e))?;

    Ok(())
}
