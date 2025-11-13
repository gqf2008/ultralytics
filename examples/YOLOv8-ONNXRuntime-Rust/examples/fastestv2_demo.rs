// YOLO-FastestV2 示例程序
// 演示如何使用FastestV2模型进行目标检测

use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array, IxDyn};
use std::path::PathBuf;

use yolov8_rs::*;

#[derive(Parser)]
struct Args {
    /// ONNX模型路径
    #[arg(long)]
    model: PathBuf,

    /// 输入图片路径
    #[arg(long)]
    source: PathBuf,

    /// 使用CUDA
    #[arg(long)]
    cuda: bool,

    /// 使用TensorRT
    #[arg(long)]
    trt: bool,

    /// 设备ID
    #[arg(long, default_value_t = 0)]
    device_id: i32,

    /// 置信度阈值
    #[arg(long, default_value_t = 0.3)]
    conf: f32,

    /// IoU阈值
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// 启用性能分析
    #[arg(long)]
    profile: bool,
}

fn preprocess_image(img: &DynamicImage, width: u32, height: u32) -> Array<f32, IxDyn> {
    // 调整图片大小
    let (w0, h0) = img.dimensions();
    let ratio = (width as f32 / w0 as f32).min(height as f32 / h0 as f32);
    let w_new = (w0 as f32 * ratio).round() as u32;
    let h_new = (h0 as f32 * ratio).round() as u32;

    let resized = img.resize_exact(w_new, h_new, image::imageops::FilterType::Triangle);

    // 创建输入tensor (1, 3, 352, 352)
    let mut input = Array::ones((1, 3, height as usize, width as usize)).into_dyn();
    input.fill(144.0 / 255.0); // 灰色填充

    // 填充图像数据
    for (x, y, rgb) in resized.pixels() {
        let x = x as usize;
        let y = y as usize;
        let [r, g, b, _] = rgb.0;
        input[[0, 0, y, x]] = r as f32 / 255.0;
        input[[0, 1, y, x]] = g as f32 / 255.0;
        input[[0, 2, y, x]] = b as f32 / 255.0;
    }

    input
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 加载图片
    println!("Loading image: {:?}", args.source);
    let img = image::open(&args.source)?;
    println!("Image size: {}x{}", img.width(), img.height());

    // 配置执行提供者
    let ep = if args.trt {
        OrtEP::Trt(args.device_id)
    } else if args.cuda {
        OrtEP::CUDA(args.device_id)
    } else {
        OrtEP::CPU
    };

    // 构建ONNX Runtime后端
    let batch = Batch {
        opt: 1,
        min: 1,
        max: 1,
    };

    let ort_config = OrtConfig {
        ep,
        batch,
        f: args.model.to_string_lossy().to_string(),
        task: Some(YOLOTask::Detect), // FastestV2是检测任务
        trt_fp16: false,
        image_size: (Some(352), Some(352)),
    };

    println!("Building ORT engine...");
    let mut engine = OrtBackend::build(ort_config)?;
    println!(
        "Engine built: {:?}, input shape: [{}, {}, {}]",
        engine.ep(),
        engine.batch(),
        engine.height(),
        engine.width()
    );

    // 配置FastestV2后处理器
    let config = FastestV2Config {
        conf_threshold: args.conf,
        iou_threshold: args.iou,
        ..Default::default()
    };
    let postprocessor = FastestV2Postprocessor::new(config, 352, 352);

    // 预处理
    let t_pre = std::time::Instant::now();
    let input = preprocess_image(&img, 352, 352);
    if args.profile {
        println!("[Preprocess]: {:?}", t_pre.elapsed());
    }

    // 推理
    let t_infer = std::time::Instant::now();
    let outputs = engine.run(input, args.profile)?;
    if args.profile {
        println!("[Inference]: {:?}", t_infer.elapsed());
    }

    // 检查输出
    println!("\nModel outputs:");
    for (i, output) in outputs.iter().enumerate() {
        println!("  Output {}: shape {:?}", i, output.shape());
    }

    // 后处理
    let t_post = std::time::Instant::now();
    let results = postprocessor.postprocess(outputs, &[img.clone()])?;
    if args.profile {
        println!("[Postprocess]: {:?}", t_post.elapsed());
    }

    // 打印结果
    println!("\nDetection results:");
    if let Some(bboxes) = results[0].bboxes() {
        println!("  Found {} objects:", bboxes.len());
        for (i, bbox) in bboxes.iter().enumerate() {
            println!(
                "    [{:2}] class={:2}, conf={:.3}, bbox=[{:.1}, {:.1}, {:.1}, {:.1}]",
                i,
                bbox.id(),
                bbox.confidence(),
                bbox.xmin(),
                bbox.ymin(),
                bbox.width(),
                bbox.height()
            );
        }
    } else {
        println!("  No objects detected");
    }

    Ok(())
}
