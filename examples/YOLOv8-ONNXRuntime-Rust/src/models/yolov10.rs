// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// YOLOv10 模型实现 (NMS-Free端到端检测)
// 特性: 无需NMS后处理, 直接输出最终检测框

use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use ndarray::{s, Array, IxDyn};

use crate::{Batch, Bbox, DetectionResult, OrtBackend, OrtConfig, OrtEP, YOLOTask};

/// YOLOv10 模型结构
pub struct YOLOv10 {
    engine: OrtBackend,
    nc: u32,
    height: u32,
    width: u32,
    batch: u32,
    conf: f32,
    iou: f32,
    names: Vec<String>,
    color_palette: Vec<(u8, u8, u8)>,
    profile: bool,
}

impl YOLOv10 {
    /// 从配置创建 YOLOv10 模型
    pub fn new(config: crate::Args) -> Result<Self> {
        // execution provider - 优先使用 Auto 自动选择最佳 EP
        let ep = if config.trt {
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::CUDA(config.device_id)
        } else {
            OrtEP::Auto // 自动选择: CUDA > DirectML > CPU
        };

        // batch
        let batch = Batch {
            opt: config.batch,
            min: config.batch_min,
            max: config.batch_max,
        };

        // build ort engine
        let ort_args = OrtConfig {
            ep,
            batch,
            f: config.model,
            task: Some(YOLOTask::Detect), // YOLOv10 only supports detection
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        // get batch, height, width, nc
        let (batch, height, width) = (engine.batch(), engine.height(), engine.width());
        let nc = engine.nc().or(config.nc).unwrap_or_else(|| {
            panic!("Failed to get num_classes, make it explicit with `--nc`");
        });

        // class names
        let names = engine.names().unwrap_or(vec!["Unknown".to_string()]);

        // color palette (与YOLOv8保持一致)
        let bright_colors = vec![
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (255, 0, 128),
            (128, 255, 0),
            (0, 128, 255),
            (128, 0, 255),
            (255, 128, 128),
        ];
        let color_palette: Vec<(u8, u8, u8)> = (0..nc)
            .map(|i| bright_colors[i as usize % bright_colors.len()])
            .collect();

        Ok(Self {
            engine,
            nc,
            height,
            width,
            batch,
            conf: config.conf,
            iou: config.iou,
            names,
            color_palette,
            profile: config.profile,
        })
    }
}

impl crate::models::Model for YOLOv10 {
    /// 预处理: 图像缩放与归一化 (与YOLOv8相同)
    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Vec<Array<f32, IxDyn>>> {
        let mut ys =
            Array::ones((xs.len(), 3, self.height as usize, self.width as usize)).into_dyn();
        ys.fill(144.0 / 255.0); // YOLOv8填充值

        for (idx, x) in xs.iter().enumerate() {
            let img = x.resize_exact(
                self.width,
                self.height,
                image::imageops::FilterType::Triangle,
            );
            let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_raw(self.width, self.height, img.to_rgb8().into_raw())
                    .expect("Failed to create image buffer");

            for (x, y, pixel) in img.enumerate_pixels() {
                let [r, g, b] = pixel.0;
                ys[[idx, 0, y as usize, x as usize]] = r as f32 / 255.0;
                ys[[idx, 1, y as usize, x as usize]] = g as f32 / 255.0;
                ys[[idx, 2, y as usize, x as usize]] = b as f32 / 255.0;
            }
        }

        Ok(vec![ys])
    }

    /// 推理: 调用ONNX Runtime
    fn run(&mut self, xs: Vec<Array<f32, IxDyn>>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        self.profile = profile;
        let all_results: Vec<Vec<_>> = xs
            .into_iter()
            .map(|x| self.engine.run(x, profile))
            .collect::<Result<Vec<_>>>()?;
        Ok(all_results.into_iter().flatten().collect())
    }

    /// 后处理: YOLOv10端到端输出 (无需NMS)
    ///
    /// YOLOv10输出格式: [batch, num_boxes, 6]
    /// 其中 6 = [x1, y1, x2, y2, confidence, class_id]
    ///
    /// 关键区别:
    /// - YOLOv8: 输出 [batch, num_boxes, 4+num_classes], 需要NMS
    /// - YOLOv10: 输出 [batch, num_boxes, 6], 已经过模型内部NMS
    fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        if self.profile {
            println!("\n[YOLOv10 后处理 - NMS-Free]");
        }

        let mut ys: Vec<DetectionResult> = Vec::new();
        let preds = &xs[0]; // [batch, num_boxes, 6]

        for (idx, x0) in xs0.iter().enumerate() {
            let (width_original, height_original) = x0.dimensions();
            let ratio = (self.width as f32 / width_original as f32)
                .min(self.height as f32 / height_original as f32);

            let mut bboxes_vec: Vec<Bbox> = Vec::new();

            // 遍历所有检测框
            for i in 0..preds.shape()[1] {
                let pred = preds.slice(s![idx, i, ..]);

                // YOLOv10输出: [x1, y1, x2, y2, confidence, class_id]
                let confidence = pred[4];

                // 置信度过滤
                if confidence < self.conf {
                    continue;
                }

                let class_id = pred[5] as usize;
                if class_id >= self.nc as usize {
                    continue;
                }

                // 坐标已经是 x1,y1,x2,y2 格式
                let x1 = pred[0] / ratio;
                let y1 = pred[1] / ratio;
                let x2 = pred[2] / ratio;
                let y2 = pred[3] / ratio;

                let width = x2 - x1;
                let height = y2 - y1;

                // 边界检查
                if width <= 0.0 || height <= 0.0 {
                    continue;
                }

                // 构建检测框
                let bbox = Bbox::new(
                    x1.max(0.0),
                    y1.max(0.0),
                    width.min(width_original as f32 - x1),
                    height.min(height_original as f32 - y1),
                    class_id,
                    confidence,
                );

                bboxes_vec.push(bbox);
            }

            if self.profile && !bboxes_vec.is_empty() {
                println!("  检测到 {} 个目标 (NMS-Free直接输出)", bboxes_vec.len());
            }

            let data = DetectionResult {
                probs: None,
                bboxes: if bboxes_vec.is_empty() {
                    None
                } else {
                    Some(bboxes_vec)
                },
                keypoints: None,
                masks: None,
            };

            ys.push(data);
        }

        Ok(ys)
    }

    fn engine_mut(&mut self) -> &mut OrtBackend {
        &mut self.engine
    }

    fn summary(&self) {
        println!("\n模型摘要:");
        println!("┌─────────────────────────────────────────┐");
        println!("│ Model: YOLOv10 (NMS-Free)               │");
        println!("│ Task: Object Detection                  │");
        println!("├─────────────────────────────────────────┤");
        println!(
            "│ Input: [{}, 3, {}, {}]           │",
            self.batch, self.height, self.width
        );
        println!("│ Classes: {}                              │", self.nc);
        println!("│ Confidence: {}                         │", self.conf);
        println!("│ NMS: Not Required (End-to-End)         │");
        println!("└─────────────────────────────────────────┘\n");
    }

    fn supports_task(&self, task: YOLOTask) -> bool {
        matches!(task, YOLOTask::Detect)
    }

    fn set_conf(&mut self, val: f32) {
        self.conf = val;
    }

    fn conf(&self) -> f32 {
        self.conf
    }

    fn set_iou(&mut self, val: f32) {
        self.iou = val;
    }

    fn iou(&self) -> f32 {
        self.iou
    }
}
