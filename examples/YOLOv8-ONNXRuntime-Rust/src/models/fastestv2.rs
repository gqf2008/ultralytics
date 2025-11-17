// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// YOLO-FastestV2 后处理模块
// 基于官方NCNN实现: https://github.com/dog-qiuqiu/Yolo-FastestV2
//
// 注意: FastestV2 当前仅实现后处理器，通过 detection::PostprocessorFactory 统一管理
//       完整的模型加载、预处理由 detector.rs 中的 OrtBackend 处理
//       如需完整 Model trait 实现，可参考 yolov8.rs

use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, IxDyn};

use crate::{non_max_suppression, Bbox, DetectionResult, Point2};

/// YOLO-FastestV2 配置
pub struct FastestV2Config {
    pub num_classes: usize,
    pub num_anchors: usize,
    pub strides: Vec<usize>,
    pub anchors: Vec<f32>,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for FastestV2Config {
    fn default() -> Self {
        Self {
            num_classes: 80, // COCO
            num_anchors: 3,
            strides: vec![16, 32], // 352/22=16, 352/11=32
            // COCO anchors from modelzoo (w, h pairs)
            anchors: vec![
                12.64, 19.39, // anchor 0
                37.88, 51.48, // anchor 1
                55.71, 138.31, // anchor 2
                126.91, 78.23, // anchor 3
                131.57, 214.55, // anchor 4
                279.92, 258.87, // anchor 5
            ],
            conf_threshold: 0.15, // FastestV2输出置信度较低,建议0.1-0.2
            iou_threshold: 0.45,
        }
    }
}

/// YOLO-FastestV2 后处理器
pub struct FastestV2Postprocessor {
    config: FastestV2Config,
    input_width: usize,
    input_height: usize,
}

impl FastestV2Postprocessor {
    pub fn new(config: FastestV2Config, input_width: usize, input_height: usize) -> Self {
        Self {
            config,
            input_width,
            input_height,
        }
    }

    /// 解码单个特征图
    ///
    /// FastestV2输出格式 [1, h, w, 95]:
    /// - 前12通道: 3个anchor的bbox坐标 (4×3): [a0_xy wh, a1_xywh, a2_xywh]
    /// - 3通道: 3个anchor的obj置信度
    /// - 80通道: 类别分数(softmax后,所有anchor共享)
    fn decode_feature_map(
        &self,
        output: &Array<f32, IxDyn>,
        stride: usize,
        anchor_offset: usize,
        scale_w: f32,
        scale_h: f32,
    ) -> Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> {
        let mut results = Vec::new();

        let shape = output.shape();
        let height = shape[1];
        let width = shape[2];

        // 遍历空间位置
        for h in 0..height {
            for w in 0..width {
                // 提取当前位置的所有通道值
                let pred = output.slice(s![0, h, w, ..]);

                // 遍历3个anchor
                for b in 0..self.config.num_anchors {
                    let anchor_idx = anchor_offset + b;

                    // 提取bbox坐标 (已sigmoid)
                    let tx = pred[b * 4 + 0];
                    let ty = pred[b * 4 + 1];
                    let tw = pred[b * 4 + 2];
                    let th = pred[b * 4 + 3];

                    // FastestV2通道布局: [bbox×12] + [obj×3] + [classes×80]
                    // 提取obj置信度 (12-14通道, 已sigmoid)
                    let obj_conf = pred[12 + b];

                    // 提取类别分数 (15-94通道, 已softmax)
                    let class_scores = pred.slice(s![15..95]);

                    // 找到最大类别分数和ID
                    let (class_id, &class_score) = class_scores
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();

                    // 综合置信度 = obj * class_score
                    let confidence = obj_conf * class_score;

                    // 置信度过滤
                    if confidence < self.config.conf_threshold {
                        continue;
                    }

                    // 解码bbox (FastestV2使用与YOLOv5相同的解码)
                    let anchor_w = self.config.anchors[anchor_idx * 2];
                    let anchor_h = self.config.anchors[anchor_idx * 2 + 1];

                    let bcx = ((tx * 2.0 - 0.5) + w as f32) * stride as f32;
                    let bcy = ((ty * 2.0 - 0.5) + h as f32) * stride as f32;
                    let bw = (tw * 2.0).powi(2) * anchor_w;
                    let bh = (th * 2.0).powi(2) * anchor_h;

                    // 转换为原图坐标
                    let x1 = (bcx - bw * 0.5) * scale_w;
                    let y1 = (bcy - bh * 0.5) * scale_h;
                    let x2 = (bcx + bw * 0.5) * scale_w;
                    let y2 = (bcy + bh * 0.5) * scale_h;

                    let bbox = Bbox::new(
                        x1.max(0.0),
                        y1.max(0.0),
                        (x2 - x1).max(0.0),
                        (y2 - y1).max(0.0),
                        class_id,
                        confidence,
                    );

                    results.push((bbox, None, None));
                }
            }
        }

        results
    }

    /// 后处理主函数
    ///
    /// # 参数
    /// - `outputs`: 模型输出 [output1, output2]
    ///   - output1: [1, 22, 22, 95] (stride 16)
    ///   - output2: [1, 11, 11, 95] (stride 32)
    /// - `original_images`: 原始输入图像
    pub fn postprocess(
        &self,
        outputs: Vec<Array<f32, IxDyn>>,
        original_images: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        let mut results = Vec::new();

        // 对每张图片处理
        for (_idx, img) in original_images.iter().enumerate() {
            let width_original = img.width() as f32;
            let height_original = img.height() as f32;

            // 计算缩放比例
            let scale_w = width_original / self.input_width as f32;
            let scale_h = height_original / self.input_height as f32;

            let mut all_detections: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();

            // 处理第一个输出 (22x22, stride=16, anchors 0-2)
            if outputs.len() > 0 {
                let output1 = &outputs[0];
                let mut dets =
                    self.decode_feature_map(output1, self.config.strides[0], 0, scale_w, scale_h);
                all_detections.append(&mut dets);
            }

            // 处理第二个输出 (11x11, stride=32, anchors 3-5)
            if outputs.len() > 1 {
                let output2 = &outputs[1];
                let mut dets =
                    self.decode_feature_map(output2, self.config.strides[1], 3, scale_w, scale_h);
                all_detections.append(&mut dets);
            }

            // NMS
            non_max_suppression(&mut all_detections, self.config.iou_threshold);

            // 提取bbox
            let bboxes: Vec<Bbox> = all_detections
                .into_iter()
                .map(|(bbox, _, _)| bbox)
                .collect();

            let result = DetectionResult::new(
                None,
                if !bboxes.is_empty() {
                    Some(bboxes)
                } else {
                    None
                },
                None,
                None,
            );

            results.push(result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FastestV2Config::default();
        assert_eq!(config.num_classes, 80);
        assert_eq!(config.num_anchors, 3);
        assert_eq!(config.anchors.len(), 12); // 6 anchors * 2 (w,h)
    }
}

// ========================================
// 完整 FastestV2 模型实现 (实现 Model trait)
// ========================================

use crate::{Batch, OrtBackend, OrtConfig, OrtEP};

/// YOLO-FastestV2 完整模型
pub struct FastestV2 {
    engine: OrtBackend,
    postprocessor: FastestV2Postprocessor,
    width: u32,
    height: u32,
}

impl FastestV2 {
    /// 从配置创建 FastestV2 模型
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
            task: Some(crate::YOLOTask::Detect), // FastestV2 only supports detection
            trt_fp16: config.fp16,
            image_size: (config.height, config.width),
        };
        let engine = OrtBackend::build(ort_args)?;

        let width = engine.width();
        let height = engine.height();

        // FastestV2 后处理器配置
        let postprocessor_config = FastestV2Config {
            num_classes: config.nc.unwrap_or(80) as usize,
            num_anchors: 3,
            strides: vec![16, 32],
            anchors: vec![
                12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92,
                258.87,
            ],
            conf_threshold: config.conf,
            iou_threshold: config.iou,
        };

        let postprocessor =
            FastestV2Postprocessor::new(postprocessor_config, width as usize, height as usize);

        Ok(Self {
            engine,
            postprocessor,
            width,
            height,
        })
    }
}

// 实现 Model trait
impl super::Model for FastestV2 {
    fn preprocess(&mut self, images: &[DynamicImage]) -> Result<Vec<Array<f32, IxDyn>>> {
        // 复用 YOLOv8 的预处理逻辑 (letterbox + normalize)
        let mut ys =
            Array::ones((images.len(), 3, self.height as usize, self.width as usize)).into_dyn();
        ys.fill(144.0 / 255.0);

        for (idx, img) in images.iter().enumerate() {
            let (w0, h0) = img.dimensions();
            let w0 = w0 as f32;
            let h0 = h0 as f32;
            let r = (self.width as f32 / w0).min(self.height as f32 / h0);
            let w_new = (w0 * r).round() as u32;
            let h_new = (h0 * r).round() as u32;

            let resized = img.resize_exact(w_new, h_new, image::imageops::FilterType::Triangle);

            for (x, y, rgb) in resized.pixels() {
                let x = x as usize;
                let y = y as usize;
                let [r, g, b, _] = rgb.0;
                ys[[idx, 0, y, x]] = (r as f32) / 255.0;
                ys[[idx, 1, y, x]] = (g as f32) / 255.0;
                ys[[idx, 2, y, x]] = (b as f32) / 255.0;
            }
        }

        Ok(vec![ys])
    }

    fn run(&mut self, xs: Vec<Array<f32, IxDyn>>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>> {
        self.engine.run(xs[0].clone(), profile)
    }

    fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>> {
        self.postprocessor.postprocess(xs, xs0)
    }

    fn engine_mut(&mut self) -> &mut OrtBackend {
        &mut self.engine
    }

    fn summary(&self) {
        println!("\n[YOLO-FastestV2 模型信息]");
        println!("  任务类型: Detection");
        println!("  输入尺寸: {}x{}", self.width, self.height);
        println!("  类别数量: {}", self.postprocessor.config.num_classes);
        println!("  Anchor数: {}", self.postprocessor.config.num_anchors);
        println!("  置信度阈值: {}", self.postprocessor.config.conf_threshold);
        println!("  IOU阈值: {}", self.postprocessor.config.iou_threshold);
    }

    fn supports_task(&self, task: crate::YOLOTask) -> bool {
        // FastestV2 仅支持目标检测
        matches!(task, crate::YOLOTask::Detect)
    }
}
