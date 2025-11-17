// 调试FastestV2输出格式
use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, IxDyn};
use std::path::PathBuf;

use yolov8_rs::*;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    source: PathBuf,
}

fn preprocess_image(img: &DynamicImage, width: u32, height: u32) -> Array<f32, IxDyn> {
    let (w0, h0) = img.dimensions();
    let ratio = (width as f32 / w0 as f32).min(height as f32 / h0 as f32);
    let w_new = (w0 as f32 * ratio).round() as u32;
    let h_new = (h0 as f32 * ratio).round() as u32;

    let resized = img.resize_exact(w_new, h_new, image::imageops::FilterType::Triangle);
    let mut input = Array::ones((1, 3, height as usize, width as usize)).into_dyn();
    input.fill(144.0 / 255.0);

    for (x, y, rgb) in resized.pixels() {
        let [r, g, b, _] = rgb.0;
        input[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
        input[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
        input[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
    }
    input
}

fn main() -> Result<()> {
    let args = Args::parse();
    let img = image::open(&args.source)?;

    let ep = OrtEP::CPU;
    let ort_config = OrtConfig {
        ep,
        batch: Batch {
            opt: 1,
            min: 1,
            max: 1,
        },
        f: args.model.to_string_lossy().to_string(),
        task: Some(YOLOTask::Detect),
        trt_fp16: false,
        image_size: (Some(352), Some(352)),
    };

    let mut engine = OrtBackend::build(ort_config)?;
    let input = preprocess_image(&img, 352, 352);
    let outputs = engine.run(input, false)?;

    // 分析输出格式
    println!("\n=== Output Analysis ===");
    for (i, output) in outputs.iter().enumerate() {
        let shape = output.shape();
        println!("\nOutput {}: shape {:?}", i, shape);

        // 取第一个位置的值 [0, 0, 0, :]
        if shape.len() == 4 {
            let h = shape[1];
            let w = shape[2];
            let c = shape[3];

            // 中心位置
            let h_center = h / 2;
            let w_center = w / 2;

            println!("  Center position [{}, {}]:", h_center, w_center);
            println!("    First 10 channels:");
            for ch in 0..10.min(c) {
                let val = output[[0, h_center, w_center, ch]];
                println!("      ch[{:2}] = {:.6}", ch, val);
            }

            println!("    Channels 12-14 (obj confidence):");
            for ch in 12..15 {
                let val = output[[0, h_center, w_center, ch]];
                println!("      ch[{:2}] = {:.6}", ch, val);
            }

            println!("    Channels 15-24 (first 10 classes):");
            for ch in 15..25 {
                let val = output[[0, h_center, w_center, ch]];
                let class_id = ch - 15;
                println!("      ch[{:2}] = {:.6} (class {})", ch, val, class_id);
            }

            println!("    Last 10 channels (classes 70-79):");
            for ch in (c - 10).max(0)..c {
                let val = output[[0, h_center, w_center, ch]];
                let class_id = ch - 15;
                println!("      ch[{:2}] = {:.6} (class {})", ch, val, class_id);
            }

            // 统计值分布
            let mut max_val = f32::MIN;
            let mut min_val = f32::MAX;
            let mut sum = 0.0f32;
            let mut count = 0;

            for h_idx in 0..h {
                for w_idx in 0..w {
                    for c_idx in 0..c {
                        let val = output[[0, h_idx, w_idx, c_idx]];
                        max_val = max_val.max(val);
                        min_val = min_val.min(val);
                        sum += val;
                        count += 1;
                    }
                }
            }

            println!("\n  Value statistics:");
            println!("    Min: {:.6}", min_val);
            println!("    Max: {:.6}", max_val);
            println!("    Mean: {:.6}", sum / count as f32);

            // 找出最大置信度位置
            println!("\n  Finding max confidence detections:");
            for h_idx in 0..h {
                for w_idx in 0..w {
                    for b in 0..3 {
                        let obj_conf = output[[0, h_idx, w_idx, 12 + b]];
                        if obj_conf > 0.15 {
                            let class_scores = &output.slice(s![0, h_idx, w_idx, 15..95]);
                            let (class_id, &class_score) = class_scores
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                .unwrap();
                            let confidence = obj_conf * class_score;
                            if confidence > 0.05 {
                                println!(
                                    "    pos[{},{},anchor{}]: obj={:.3}, class{}={:.3}, conf={:.3}",
                                    h_idx, w_idx, b, obj_conf, class_id, class_score, confidence
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
