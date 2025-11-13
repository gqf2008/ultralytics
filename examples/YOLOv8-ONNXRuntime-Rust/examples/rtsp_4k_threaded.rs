/// RTSP Real-time Detection - ggez GPU Accelerated + 3-Thread Architecture
/// Supports dynamic video resolution (720p/1080p/2K/4K/...)
///
/// 模块化版本 - 主程序入口
use crossbeam_channel::bounded;
use ggez::conf::{WindowMode, WindowSetup};
use ggez::event;
use ggez::graphics::FontData;
use ggez::{ContextBuilder, GameResult};
use yolov8_rs::rtsp::*;

fn main() -> GameResult {
    println!(
        "✅ 三线程架构 | ggez GPU加速 | 窗口 {}x{} | 动态分辨率 | 推理{}x{} | 自适应解码",
        WINDOW_WIDTH, WINDOW_HEIGHT, INF_SIZE, INF_SIZE
    );

    // Channel 1: Decode -> Render (动态分辨率 frames)
    let (tx_decode, rx_decode) = bounded::<DecodedFrame>(5);

    // Channel 2: Render -> Inference (320x320 resized frames)
    let (tx_to_inference, rx_resized) = bounded::<ResizedFrame>(1);

    // Channel 3: Inference -> Render (detection results)
    let (tx_result, rx_result) = bounded::<InferredFrame>(1);

    // 启动解码线程 (自适应选择最佳解码器)
    let tx_decode_clone = tx_decode.clone();
    std::thread::spawn(move || {
        println!("🎬 开始连接 RTSP...");

        let filter = DecodeFilter::new(tx_decode_clone);

        let rtsp_url = "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1&subtype=0";

        // 自适应解码器选择
        adaptive_decode(rtsp_url, filter);
    });

    // Start inference thread
    std::thread::spawn(move || {
        inference_thread(
            rx_resized,
            tx_result,
            "models/yolov8n.onnx".to_string(),
            "models/yolov8n-pose.onnx".to_string(),
            INF_SIZE,
        );
    });

    // ggez main thread
    let (mut ctx, event_loop) = ContextBuilder::new("yolo_4k", "ultralytics")
        .window_setup(WindowSetup::default().title("YOLO 4K Detection (ggez GPU Accelerated)"))
        .window_mode(WindowMode::default().dimensions(WINDOW_WIDTH, WINDOW_HEIGHT))
        .build()?;

    // 加载中文字体 (微软雅黑)
    let font_data = std::fs::read("assets/font/msyh.ttc")?;
    let font = FontData::from_vec(font_data)?;
    ctx.gfx.add_font("MicrosoftYaHei", font);
    println!("✅ 中文字体加载成功: 微软雅黑");

    let app = YoloApp::new(
        &mut ctx,
        rx_decode,
        rx_result,
        tx_to_inference,
        INF_SIZE,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "models/yolov8n.onnx".to_string(),
        "models/yolov8n-pose.onnx".to_string(),
    );

    event::run(ctx, event_loop, app)
}
