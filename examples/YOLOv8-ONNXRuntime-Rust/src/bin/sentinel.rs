#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
/// 数字卫兵 (Digital Sentinel)
///
/// 智能视频监控系统
///
/// 系统架构:
/// 1. 采集线程: 视频解码与预处理 (独立工作线程)
/// 2. 检测线程: 目标检测与追踪 (独立工作线程)
/// 3. 主线程:   渲染显示 (macroquad事件循环)
//
// 使用 mimalloc 替代系统默认分配器 (性能提升 10-30%)
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use egui_macroquad::egui;
use macroquad::prelude::*;
use yolov8_rs::detection::INF_SIZE;
use yolov8_rs::renderer::Renderer;

/// 数字卫兵参数
#[derive(Parser, Debug)]
#[command(author, version, about = "数字卫兵 - 智能视频监控系统", long_about = None)]
struct Args {
    /// 检测模型 (n/s/m/l/x/v10n/v10s/v10m/v11n/v11s/v11m/fastest/fastest-xl/n-int8/m-int8/v5n/v5s/v5m/nanodet/nanodet-m/nanodet-plus/yolox_s/yolox_m/yolox_l)
    #[arg(short, long, default_value = "n")]
    model: String,

    /// 跟踪算法 (deepsort/bytetrack/none)
    #[arg(short = 't', long, default_value = "none")]
    tracker: String,

    /// 启用姿态估计 (需要pose模型支持)
    #[arg(short = 'p', long, default_value_t = false)]
    pose: bool,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "数字卫兵 - Digital Sentinel".to_owned(),
        window_width: 1280,
        window_height: 720,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Args::parse();
    // 加载中文字体
    let font_data = match std::fs::read("assets/font/msyh.ttc") {
        Ok(data) => {
            println!("✅ 中文字体加载成功: 微软雅黑");
            Some(data)
        }
        Err(e) => {
            eprintln!("⚠️  中文字体加载失败: {}, 将使用默认字体", e);
            None
        }
    };

    // 设置 egui 中文字体
    if let Some(font_bytes) = font_data {
        egui_macroquad::cfg(|ctx| {
            let mut fonts = egui::FontDefinitions::default();
            fonts.font_data.insert(
                "msyh".to_owned(),
                std::sync::Arc::new(egui::FontData::from_owned(font_bytes)),
            );

            // 将中文字体设置为优先字体
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "msyh".to_owned());

            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push("msyh".to_owned());

            ctx.set_fonts(fonts);
            ctx.set_pixels_per_point(ctx.zoom_factor());
        });
    }

    // 构建模型路径
    let fastest_variant = if args.model == "fastest" || args.model == "fastestv2" {
        "yolo-fastestv2-opt"
    } else {
        "yolo-fastest-1.1"
    };

    let detect_model = if args.model.starts_with("yolox") {
        format!("models/{}.onnx", args.model)
    } else if args.model.starts_with("v10") {
        let variant = args.model.trim_start_matches("v10");
        format!("models/yolov10{}.onnx", variant)
    } else if args.model.starts_with("v11") {
        let variant = args.model.trim_start_matches("v11");
        format!("models/yolov11{}.onnx", variant)
    } else if args.model == "fastest" || args.model.starts_with("fastest") {
        format!("models/{}.onnx", fastest_variant)
    } else if args.model.starts_with("nanodet") {
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
        let variant = args.model.trim_start_matches("v5");
        format!("models/yolov5{}.onnx", variant)
    } else if args.model.ends_with("-int8") {
        let base = args.model.trim_end_matches("-int8");
        format!("models/yolov8{}_int8.onnx", base)
    } else {
        if args.model.starts_with("yolov8") {
            format!("models/{}.onnx", args.model)
        } else {
            format!("models/yolov8{}.onnx", args.model)
        }
    };

    println!("🚀 数字卫兵系统启动");
    println!("📦 默认检测模型: {}", detect_model);
    println!("🎯 默认跟踪算法: {}", args.tracker);
    println!(
        "🧍 默认姿态估计: {}",
        if args.pose { "启用" } else { "禁用" }
    );
    println!("\n💡 请在UI中配置输入源,检测模块将在启动视频流时自动启动");
    println!();

    // 创建配置更新通道
    let (config_tx, config_rx) = crossbeam_channel::bounded(5);

    // 不再自动启动解码器和检测器,等待用户在UI中配置
    // 解码器和检测器将通过 switch_decoder_source() 函数启动

    // 不在这里启动检测线程,而是在首次启动解码器时启动
    // 这样可以避免不必要的资源占用

    // 提取干净的模型名称
    let detect_model_name = detect_model.replace("models/", "").replace(".onnx", "");

    let mut renderer = Renderer::new(detect_model_name, String::new(), args.tracker.clone());
    renderer.set_config_sender(config_tx.clone());

    // 保存检测器启动参数,供后续使用
    renderer.set_detector_params(
        detect_model.clone(),
        INF_SIZE,
        args.tracker.clone(),
        args.pose,
    );

    println!("✅ 系统就绪,等待配置输入源...\n");

    // 主循环
    loop {
        renderer.update();
        renderer.handle_input();
        renderer.draw();
        renderer.draw_egui();

        next_frame().await;
    }
}
