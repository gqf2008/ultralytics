//! YOLOX 模型实现
//! YOLOX is an anchor-free YOLO with high performance
//!
//! 核心特点:
//! - Anchor-Free: 无需预设锚框
//! - Decoupled Head: 解耦检测头
//! - SimOTA: 先进的标签分配策略

use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Axis, IxDyn};

use crate::{
    non_max_suppression, Batch, Bbox, DetectionResult, OrtBackend, OrtConfig, OrtEP, Point2,
    YOLOTask,
};

/// YOLOX 模型结构
pub struct YOLOX {
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

impl YOLOX {
    /// 从配置创建 YOLOX 模型
    pub fn new(config: crate::Args) -> Result<Self> {
        // execution provider
        let ep = if config.trt {
            OrtEP::Trt(config.device_id)
        } else if config.cuda {
            OrtEP::CUDA(config.device_id)
        } else {
            OrtEP::CPU
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
            task: Some(YOLOTask::Detect), // YOLOX only supports detection
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        // get batch, height, width
        let (batch, height, width) = (engine.batch(), engine.height(), engine.width());

        // YOLOX uses COCO classes by default
        let nc = engine.nc().or(config.nc).unwrap_or(80);

        // class names
        let names = engine.names().unwrap_or_else(|| {
            // COCO class names (80 classes)
            vec![
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect()
        });

        // color palette
        let bright_colors = vec![
            (255, 0, 0),     // 红色
            (0, 255, 0),     // 绿色
            (0, 0, 255),     // 蓝色
            (255, 255, 0),   // 黄色
            (255, 0, 255),   // 品红
            (0, 255, 255),   // 青色
            (255, 128, 0),   // 橙色
            (255, 0, 128),   // 粉红
            (128, 255, 0),   // 黄绿
            (0, 128, 255),   // 天蓝
            (255, 255, 255), // 白色
            (128, 0, 255),   // 紫色
        ];

        let color_palette: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(i, _)| bright_colors[i % bright_colors.len()])
            .collect();

        Ok(Self {
            engine,
            names,
            conf: config.conf,
            iou: config.iou,
            color_palette,
            profile: config.profile,
            nc,
            height,
            width,
            batch,
        })
    }

    fn scale_wh(&self, w0: f32, h0: f32, w1: f32, h1: f32) -> (f32, f32, f32) {
        let r = (w1 / w0).min(h1 / h0);
        (r, (w0 * r).round(), (h0 * r).round())
    }

    /// 获取类别名称列表
    pub fn names(&self) -> &Vec<String> {
        &self.names
    }

    /// 获取调色板
    pub fn color_palette(&self) -> &Vec<(u8, u8, u8)> {
        &self.color_palette
    }
}

impl crate::models::Model for YOLOX {
    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Vec<Array<f32, IxDyn>>> {
        let mut ys =
            Array::ones((xs.len(), 3, self.height as usize, self.width as usize)).into_dyn();
        ys.fill(114.0 / 255.0); // YOLOX uses 114 as padding value

        for (idx, x) in xs.iter().enumerate() {
            let (w0, h0) = (x.width() as f32, x.height() as f32);
            let (_, w_new, h_new) = self.scale_wh(w0, h0, self.width as f32, self.height as f32);

            let img = x.resize_exact(
                w_new as u32,
                h_new as u32,
                image::imageops::FilterType::Triangle,
            );

            for (x, y, rgb) in img.to_rgb8().enumerate_pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b] = rgb.0;
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }

        Ok(vec![ys])
    }

    fn run(&mut self, xs: Vec<Array<f32, IxDyn>>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        if xs.is_empty() {
            return Ok(vec![]);
        }
        self.engine.run(xs[0].clone(), profile || self.profile)
    }

    fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        if xs.is_empty() {
            return Ok(vec![]);
        }

        // YOLOX output: [batch, num_boxes, 85] or [batch, num_boxes, nc+5]
        // Format: [x_center, y_center, width, height, objectness, class_probs...]
        const CXYWHC_OFFSET: usize = 5;
        let preds = &xs[0];

        let mut ys = Vec::new();
        for (idx, x0) in xs0.iter().enumerate() {
            // original image width & height
            let width_original = x0.width() as f32;
            let height_original = x0.height() as f32;

            // ratios
            let ratio =
                (self.width as f32 / width_original).min(self.height as f32 / height_original);

            // save each result
            let mut data: Vec<Vec<f32>> = Vec::new();
            let preds_single = preds.slice(ndarray::s![idx, .., ..]);

            // iterate over predictions
            for pred in preds_single.axis_iter(Axis(0)) {
                let bbox = pred.slice(ndarray::s![0..4]);
                let clss = pred.slice(ndarray::s![CXYWHC_OFFSET..]);
                let obj_conf = pred[4];

                // find class with max confidence
                let (class_id, &class_conf) = clss
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                // confidence = objectness * class_confidence
                let confidence = obj_conf * class_conf;

                // filter by confidence threshold
                if confidence < self.conf {
                    continue;
                }

                // YOLOX outputs: [cx, cy, w, h]
                let cx = bbox[0];
                let cy = bbox[1];
                let w = bbox[2];
                let h = bbox[3];

                // convert to [x1, y1, x2, y2] and scale to original image
                let x1 = (cx - w / 2.0) / ratio;
                let y1 = (cy - h / 2.0) / ratio;
                let x2 = (cx + w / 2.0) / ratio;
                let y2 = (cy + h / 2.0) / ratio;

                // clamp to image boundaries
                let x1 = x1.max(0.0).min(width_original);
                let y1 = y1.max(0.0).min(height_original);
                let x2 = x2.max(0.0).min(width_original);
                let y2 = y2.max(0.0).min(height_original);

                data.push(vec![x1, y1, x2, y2, confidence, class_id as f32]);
            }

            // nms
            let mut bboxes: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();
            for item in data.iter() {
                let x1 = item[0];
                let y1 = item[1];
                let x2 = item[2];
                let y2 = item[3];
                bboxes.push((
                    Bbox::new(
                        x1,
                        y1,
                        x2 - x1, // width
                        y2 - y1, // height
                        item[5] as usize,
                        item[4],
                    ),
                    None,
                    None,
                ));
            }
            non_max_suppression(&mut bboxes, self.iou);

            // extract bboxes only
            let final_bboxes: Vec<Bbox> = bboxes.into_iter().map(|(bbox, _, _)| bbox).collect();

            // save result
            ys.push(DetectionResult {
                bboxes: Some(final_bboxes),
                keypoints: None,
                probs: None,
                masks: None,
            });
        }

        Ok(ys)
    }

    fn engine_mut(&mut self) -> &mut OrtBackend {
        &mut self.engine
    }

    fn summary(&self) {
        println!(
            "\n📋 YOLOX Model Summary\n\
             Task: {:?}\n\
             Batch: {}\n\
             Width: {}\n\
             Height: {}\n\
             Classes: {}\n\
             Conf Threshold: {}\n\
             IoU Threshold: {}\n",
            YOLOTask::Detect,
            self.batch,
            self.width,
            self.height,
            self.nc,
            self.conf,
            self.iou
        );
    }

    fn supports_task(&self, task: YOLOTask) -> bool {
        matches!(task, YOLOTask::Detect)
    }
}
