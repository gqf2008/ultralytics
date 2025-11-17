/// 数字卫兵 (Digital Sentinel)
///
/// 智能视频监控系统
///
/// 系统架构:
/// 1. 采集线程: 视频解码与预处理 (独立工作线程)
/// 2. 检测线程: 目标检测与追踪 (独立工作线程)
/// 3. 主线程:   渲染显示 (ggez事件循环)

// 使用 mimalloc 替代系统默认分配器 (性能提升 10-30%)
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use ggez::conf::{WindowMode, WindowSetup};
use ggez::event;
use ggez::graphics::FontData;
use ggez::{ContextBuilder, GameResult};
use yolov8_rs::detection::INF_SIZE;
use yolov8_rs::renderer::{Renderer, WINDOW_HEIGHT, WINDOW_WIDTH};
use yolov8_rs::{detection, input};

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

    /// 检测模型 (n/s/m/l/x/fastest/fastest-xl/n-int8/m-int8/v5n/v5s/v5m/nanodet/nanodet-m/nanodet-plus)
    #[arg(short, long, default_value = "fastestv2")]
    model: String,
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
    } else if args.model.starts_with("nanodet") {
        // NanoDet 模型 (例如: nanodet-m -> nanodet-m.onnx, nanodet-plus -> nanodet-plus-m_320.onnx)
        if args.model == "nanodet" || args.model == "nanodet-m" {
            "models/nanodet-m.onnx".to_string()
        } else if args.model == "nanodet-plus" {
            "models/nanodet-plus-m_320.onnx".to_string()
        } else if args.model == "nanodet-plus-416" {
            "models/nanodet-plus-m_416.onnx".to_string()
        } else if args.model == "nanodet-plus-1.5x" {
            "models/nanodet-plus-m-1.5x_320.onnx".to_string()
        } else if args.model == "nanodet-plus-1.5x-416" {
            "models/nanodet-plus-m-1.5x_416.onnx".to_string()
        } else {
            format!("models/{}.onnx", args.model)
        }
    } else if args.model.starts_with("v5") {
        // YOLOv5 模型 (例如: v5n -> yolov5n.onnx)
        let variant = args.model.trim_start_matches("v5");
        format!("models/yolov5{}.onnx", variant)
    } else if args.model.ends_with("-int8") {
        // INT8量化模型 (例如: n-int8 -> yolov8n_int8.onnx)
        let base = args.model.trim_end_matches("-int8");
        format!("models/yolov8{}_int8.onnx", base)
    } else {
        // YOLOv8 标准模型 (例如: yolov8n -> yolov8n.onnx, 或直接 n -> yolov8n.onnx)
        if args.model.starts_with("yolov8") {
            format!("models/{}.onnx", args.model)
        } else {
            format!("models/yolov8{}.onnx", args.model)
        }
    };

    println!("🚀 数字卫兵系统启动");
    println!("📦 检测模型: {}", detect_model);
    println!("📹 RTSP地址: {}", args.rtsp_url);
    println!();

    // ========== 启动输入线程 ==========
    let rtsp_url = args.rtsp_url.clone();
    std::thread::spawn(move || {
        let mut decoder = input::Decoder::new(rtsp_url);
        decoder.run();
    });

    // ========== 启动检测线程 ==========
    let detect_model_clone = detect_model.clone();

    std::thread::spawn(move || {
        let mut det = detection::Detector::new(detect_model_clone, INF_SIZE);
        det.run();
    });

    // ========== 主线程: 数字卫兵渲染 ==========
    let (mut ctx, event_loop) = ContextBuilder::new("sentinel", "ultralytics")
        .window_setup(
            WindowSetup::default()
                .title("数字卫兵 - Digital Sentinel")
                .vsync(true),
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

    // 提取干净的模型名称 (去掉路径和扩展名)
    let detect_model_name = detect_model.replace("models/", "").replace(".onnx", "");

    let renderer = Renderer::new(detect_model_name, String::new(), String::new())?;

    println!("✅ 系统就绪,开始监控...\n");

    event::run(ctx, event_loop, renderer)
}
