use clap::Parser;
/// RTSP Real-time Detection - ggez GPU Accelerated + 3-Thread Architecture
/// Supports dynamic video resolution (720p/1080p/2K/4K/...)
///
/// 主程序入口 - 直接运行: cargo run --bin yolov8-rtsp --release
use crossbeam_channel::bounded;
use ggez::conf::{WindowMode, WindowSetup};
use ggez::event;
use ggez::graphics::FontData;
use ggez::{ContextBuilder, GameResult};
use yolov8_rs::rtsp::*;

/// RTSP实时检测程序
#[derive(Parser, Debug)]
#[command(author, version, about = "YOLOv8 RTSP实时检测", long_about = None)]
struct Args {
    /// RTSP流地址
    #[arg(
        short,
        long,
        default_value = "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1&subtype=0"
    )]
    rtsp_url: String,

    /// 检测模型 (n/s/m/l/x/fastest/fastest-xl)
    #[arg(short, long, default_value = "m")]
    model: String,

    /// 是否启用姿态估计 (YOLO-Fastest不支持)
    #[arg(short, long, default_value_t = true)]
    pose: bool,

    /// 是否使用INT8量化模型 (更快,精度略降)
    #[arg(long)]
    int8: bool,

    /// 追踪算法: deepsort 或 bytetrack
    #[arg(long, default_value = "deepsort")]
    tracker: String,
}

fn main() -> GameResult {
    let args = Args::parse();

    // 解析追踪器类型
    let tracker_type = match args.tracker.to_lowercase().as_str() {
        "bytetrack" => TrackerType::ByteTrack,
        _ => TrackerType::DeepSort,
    };

    // 构建模型路径 (支持YOLO-Fastest)
    let (detect_model, pose_model, is_fastest) = if args.model.starts_with("fastest") {
        // YOLO-Fastest系列
        let fastest_variant = if args.model == "fastest-xl" {
            "yolo-fastest-xl"
        } else if args.model == "fastestv2" || args.model == "fastest" {
            "yolo-fastestv2-opt" // FastestV2优化模型
        } else {
            "yolo-fastest-1.1"
        };
        let detect = format!("models/{}.onnx", fastest_variant);
        let pose = String::new(); // YOLO-Fastest不支持姿态
        (detect, pose, true)
    } else {
        // YOLOv8系列
        let suffix = if args.int8 { "_int8" } else { "" };
        let detect = format!("models/yolov8{}{}.onnx", args.model, suffix);
        let pose = format!("models/yolov8{}-pose{}.onnx", args.model, suffix);
        (detect, pose, false)
    };

    println!(
        "✅ 三线程架构 | ggez GPU加速 | 窗口 {}x{} | 动态分辨率 | 推理{}x{}",
        WINDOW_WIDTH, WINDOW_HEIGHT, INF_SIZE, INF_SIZE
    );
    println!("📦 检测模型: {}", detect_model);

    if is_fastest {
        println!("⚡ YOLO-Fastest: 超轻量级模型 (0.35M params)");
        println!("⚠️  YOLO-Fastest不支持姿态估计");
    } else {
        if args.pose {
            println!("🦴 姿态模型: {}", pose_model);
        }
        if args.int8 {
            println!("⚡ INT8量化: 已启用 (3-4倍加速)");
        }
    }

    println!("🎯 追踪算法: {:?}", tracker_type);
    println!("📹 RTSP地址: {}", args.rtsp_url);
    println!();

    // Channel 1: Decode -> Render (动态分辨率 frames)
    // 增大缓冲区: 30 → 60 帧,避免关键帧丢失
    let (tx_decode, rx_decode) = bounded::<DecodedFrame>(60);

    // Channel 2: Render -> Inference (320x320 resized frames)
    // 增大缓冲区: 2 → 60 帧,避免阻塞渲染线程(FastestV2每帧推理)
    let (tx_to_inference, rx_resized) = bounded::<ResizedFrame>(60);

    // Channel 3: Inference -> Render (detection results)
    // 增大缓冲区: 2 → 60 帧,避免阻塞推理线程
    let (tx_result, rx_result) = bounded::<InferredFrame>(60);

    // 启动解码线程 (自适应选择最佳解码器)
    let rtsp_url = args.rtsp_url.clone();
    let tx_decode_clone = tx_decode.clone();
    std::thread::spawn(move || {
        println!("🎬 开始连接 RTSP...");

        let filter = DecodeFilter::new();

        // 自适应解码器选择
        adaptive_decode(&rtsp_url, filter);
    });

    // Start inference thread
    let detect_model_clone = detect_model.clone();
    let pose_model_clone = pose_model.clone();
    std::thread::spawn(move || {
        inference_thread(
            rx_resized,
            tx_result,
            detect_model_clone,
            pose_model_clone,
            INF_SIZE,
        );
    });

    // ggez main thread
    let (mut ctx, event_loop) = ContextBuilder::new("yolo_2k", "ultralytics")
        .window_setup(
            WindowSetup::default()
                .title("YOLO 2K Detection")
                .vsync(false), // 关闭vsync,不限制帧率
        )
        .window_mode(
            WindowMode::default()
                .dimensions(WINDOW_WIDTH, WINDOW_HEIGHT)
                .resizable(true),
        )
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
        detect_model,
        pose_model,
        tracker_type,
    );

    event::run(ctx, event_loop, app)
}
