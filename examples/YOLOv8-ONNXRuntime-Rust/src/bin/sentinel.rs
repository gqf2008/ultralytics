/// 数字卫兵 (Digital Sentinel)
/// 
/// 智能视频监控系统
///
/// 系统架构:
/// 1. 采集线程: 视频解码与预处理 (独立工作线程)
/// 2. 检测线程: 目标检测与追踪 (独立工作线程)
/// 3. 主线程:   渲染显示 (ggez事件循环)

use clap::Parser;
use ggez::conf::{WindowMode, WindowSetup};
use ggez::event;
use ggez::graphics::FontData;
use ggez::{ContextBuilder, GameResult};
use yolov8_rs::rtsp::{self, WINDOW_WIDTH, WINDOW_HEIGHT, INF_SIZE};
use yolov8_rs::systems::{acquisition, detection};

mod renderer;
use renderer::Renderer;

/// 数字卫兵参数
#[derive(Parser, Debug)]
#[command(author, version, about = "数字卫兵 - 智能视频监控系统", long_about = None)]
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

    /// 是否启用姿态估计
    #[arg(short, long, default_value_t = true)]
    pose: bool,

    /// 追踪算法: deepsort 或 bytetrack
    #[arg(long, default_value = "deepsort")]
    tracker: String,
}

fn main() -> GameResult {
    let args = Args::parse();

    // 构建模型路径
    let fastest_variant = if args.model == "fastest" || args.model == "fastestv2" {
        "yolo-fastestv2-opt"
    } else {
        "yolo-fastest-1.1"
    };

    let detect_model = if args.model == "fastest" || args.model.starts_with("fastest") {
        format!("models/{}.onnx", fastest_variant)
    } else {
        format!("models/yolov8{}-det.onnx", args.model)
    };

    let pose_model = if args.pose && !args.model.starts_with("fastest") {
        format!("models/yolov8{}-pose.onnx", args.model)
    } else {
        String::new()
    };

    println!("🚀 数字卫兵系统启动");
    println!("📦 检测模型: {}", detect_model);
    if !pose_model.is_empty() {
        println!("🦴 姿态模型: {}", pose_model);
    }
    println!("📹 RTSP地址: {}", args.rtsp_url);
    println!();

    // ========== 启动采集线程 ==========
    let rtsp_url = args.rtsp_url.clone();
    std::thread::spawn(move || {
        let mut acq = acquisition::Decoder::new(rtsp_url);
        acq.run();
    });

    // ========== 启动检测线程 ==========
    let detect_model_clone = detect_model.clone();
    let pose_model_clone = pose_model.clone();
    let tracker_type = match args.tracker.to_lowercase().as_str() {
        "bytetrack" => rtsp::TrackerType::ByteTrack,
        _ => rtsp::TrackerType::DeepSort,
    };
    
    std::thread::spawn(move || {
        let mut det = detection::Detector::new(
            detect_model_clone,
            pose_model_clone,
            tracker_type,
            INF_SIZE,
        );
        det.run();
    });

    // ========== 主线程: 数字卫兵渲染 ==========
    let (mut ctx, event_loop) = ContextBuilder::new("sentinel", "ultralytics")
        .window_setup(
            WindowSetup::default()
                .title("数字卫兵 - Digital Sentinel")
                .vsync(false),
        )
        .window_mode(
            WindowMode::default()
                .dimensions(WINDOW_WIDTH, WINDOW_HEIGHT)
                .resizable(true),
        )
        .build()?;

    // 加载中文字体
    let font_data = std::fs::read("assets/font/msyh.ttc")?;
    let font = FontData::from_vec(font_data)?;
    ctx.gfx.add_font("MicrosoftYaHei", font);
    println!("✅ 中文字体加载成功: 微软雅黑");

    let renderer = Renderer::new(&mut ctx)?;

    println!("✅ 系统就绪,开始监控...\n");

    event::run(ctx, event_loop, renderer)
}
