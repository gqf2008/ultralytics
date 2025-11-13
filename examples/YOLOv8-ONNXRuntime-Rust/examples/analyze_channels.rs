use anyhow::Result;
use image::io::Reader as ImageReader;
use ndarray::{s, Array, IxDyn};
use std::path::PathBuf;
use yolov8_rs::ort_backend::{Batch, OrtBackend, OrtConfig, OrtEP, YOLOTask};

fn main() -> Result<()> {
    // 加载图片
    let img_path = r"d:\my\Documents\GitHub\Yolo-FastestV2\img\000139.jpg";
    let img = image::open(img_path)?;

    // 构建后端
    let ort_config = OrtConfig {
        ep: OrtEP::CPU,
        batch: Batch {
            opt: 1,
            min: 1,
            max: 1,
        },
        f: "models/yolo-fastestv2-opt.onnx".to_string(),
        task: Some(YOLOTask::Detect),
        trt_fp16: false,
        image_size: (Some(352), Some(352)),
    };
    let mut backend = OrtBackend::build(ort_config)?;

    // 预处理
    fn preprocess(img: &image::DynamicImage) -> Array<f32, IxDyn> {
        use image::{GenericImageView, ImageBuffer, Rgb};
        let img = img.resize_exact(352, 352, image::imageops::FilterType::Triangle);
        let mut input = Array::zeros(IxDyn(&[1, 3, 352, 352]));
        for (x, y, pixel) in img.pixels() {
            let [r, g, b, _] = pixel.0;
            input[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
            input[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
            input[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
        }
        input
    }

    let input = preprocess(&img);

    // 推理
    let outputs = backend.run(input, false)?;

    // 分析Output 0的中心点[11,11]所有95个通道
    println!("\n=== Analyzing Output 0: [1,22,22,95] Center[11,11] ===");
    let out0 = &outputs[0];
    for ch in 0..95 {
        let val = out0[[0, 11, 11, ch]];
        println!("ch[{:2}] = {:.6}", ch, val);
    }

    Ok(())
}
