// NanoDet 后处理模块
// 基于官方实现: https://github.com/RangiLyu/nanodet
// NanoDet是FCOS-style anchor-free单阶段目标检测器

use anyhow::Result;
use image::DynamicImage;
use ndarray::{s, Array, IxDyn};

use crate::{non_max_suppression, Bbox, Point2, YOLOResult};

/// NanoDet 配置
pub struct NanoDetConfig {
    pub num_classes: usize,
    pub strides: Vec<usize>,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub reg_max: usize, // Distribution Focal Loss参数,默认7
}

impl Default for NanoDetConfig {
    fn default() -> Self {
        Self {
            num_classes: 80, // COCO
            strides: vec![8, 16, 32], // NanoDet-Plus三个特征层
            conf_threshold: 0.35, // NanoDet推荐0.35-0.4
            iou_threshold: 0.6,   // NanoDet推荐0.5-0.6
            reg_max: 7,           // DFL参数
        }
    }
}

/// NanoDet 后处理器
/// 
/// NanoDet输出格式 (anchor-free):
/// - cls_pred_stride8: [batch, num_classes, h, w] - 分类分数
/// - dis_pred_stride8: [batch, 32, h, w] - 边界框分布 (4边×8个bin)
/// - 对应stride 16, 32的输出同理
pub struct NanoDetPostprocessor {
    config: NanoDetConfig,
    input_width: usize,
    input_height: usize,
}

impl NanoDetPostprocessor {
    pub fn new(config: NanoDetConfig, input_width: usize, input_height: usize) -> Self {
        Self {
            config,
            input_width,
            input_height,
        }
    }

    /// Distribution Focal Loss (DFL) 解码
    /// 
    /// 将分布预测转换为实际距离值
    /// dis: [reg_max+1] - 单边的分布概率
    fn dfl_decode(&self, dis: &[f32]) -> f32 {
        let mut distance = 0.0;
        for (i, &prob) in dis.iter().enumerate() {
            distance += i as f32 * prob;
        }
        distance
    }

    /// Softmax激活
    fn softmax(x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = x.iter().map(|&v| (v - max_val).exp()).sum();
        x.iter().map(|&v| (v - max_val).exp() / exp_sum).collect()
    }

    /// 解码单个特征图
    /// 
    /// # 参数
    /// - cls_pred: [1, num_classes, h, w] - 分类预测
    /// - dis_pred: [1, 4*(reg_max+1), h, w] - 距离预测分布
    /// - stride: 下采样步长
    fn decode_feature_map(
        &self,
        cls_pred: &Array<f32, IxDyn>,
        dis_pred: &Array<f32, IxDyn>,
        stride: usize,
        scale_w: f32,
        scale_h: f32,
    ) -> Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> {
        let mut results = Vec::new();

        let cls_shape = cls_pred.shape();
        let height = cls_shape[2];
        let width = cls_shape[3];

        let reg_max_plus_1 = self.config.reg_max + 1;

        // 遍历空间位置
        for h in 0..height {
            for w in 0..width {
                // 提取分类分数 [num_classes]
                let cls_scores = cls_pred.slice(s![0, .., h, w]);
                
                // Sigmoid激活
                let cls_scores: Vec<f32> = cls_scores
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-x).exp()))
                    .collect();

                // 找到最大类别
                let (class_id, &confidence) = cls_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                // 置信度过滤
                if confidence < self.config.conf_threshold {
                    continue;
                }

                // 提取距离预测 [4*(reg_max+1)]
                let dis_preds = dis_pred.slice(s![0, .., h, w]);

                // 计算中心点坐标
                let cx = (w as f32 + 0.5) * stride as f32;
                let cy = (h as f32 + 0.5) * stride as f32;

                // 解码4条边的距离 (left, top, right, bottom)
                let mut distances = Vec::with_capacity(4);
                for i in 0..4 {
                    let start = i * reg_max_plus_1;
                    let end = start + reg_max_plus_1;
                    let dis_slice: Vec<f32> = dis_preds
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .collect();
                    
                    // Softmax + DFL解码
                    let dis_softmax = Self::softmax(&dis_slice);
                    let distance = self.dfl_decode(&dis_softmax);
                    distances.push(distance * stride as f32);
                }

                // distances: [left, top, right, bottom]
                let x1 = (cx - distances[0]) * scale_w;
                let y1 = (cy - distances[1]) * scale_h;
                let x2 = (cx + distances[2]) * scale_w;
                let y2 = (cy + distances[3]) * scale_h;

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

        results
    }

    /// 后处理主函数
    /// 
    /// # 参数
    /// - `outputs`: 模型输出 [cls_8, dis_8, cls_16, dis_16, cls_32, dis_32]
    ///   NanoDet-Plus-m 输出6个tensor,每个stride对应(cls_pred, dis_pred)
    /// - `original_images`: 原始输入图像
    pub fn postprocess(
        &self,
        outputs: Vec<Array<f32, IxDyn>>,
        original_images: &[DynamicImage],
    ) -> Result<Vec<YOLOResult>> {
        let mut results = Vec::new();

        // 对每张图片处理
        for img in original_images.iter() {
            let width_original = img.width() as f32;
            let height_original = img.height() as f32;

            // 计算缩放比例
            let scale_w = width_original / self.input_width as f32;
            let scale_h = height_original / self.input_height as f32;

            let mut all_detections: Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)> = Vec::new();

            // NanoDet输出: [cls_8, dis_8, cls_16, dis_16, cls_32, dis_32]
            let num_strides = self.config.strides.len();
            
            for i in 0..num_strides {
                let cls_idx = i * 2;
                let dis_idx = i * 2 + 1;

                if outputs.len() <= dis_idx {
                    break;
                }

                let cls_pred = &outputs[cls_idx];
                let dis_pred = &outputs[dis_idx];
                let stride = self.config.strides[i];

                let mut dets = self.decode_feature_map(
                    cls_pred,
                    dis_pred,
                    stride,
                    scale_w,
                    scale_h,
                );
                all_detections.append(&mut dets);
            }

            // NMS
            non_max_suppression(&mut all_detections, self.config.iou_threshold);

            // 提取bbox
            let bboxes: Vec<Bbox> = all_detections
                .into_iter()
                .map(|(bbox, _, _)| bbox)
                .collect();

            let result = YOLOResult::new(
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
        let config = NanoDetConfig::default();
        assert_eq!(config.num_classes, 80);
        assert_eq!(config.strides.len(), 3);
        assert_eq!(config.reg_max, 7);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = NanoDetPostprocessor::softmax(&x);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dfl_decode() {
        let config = NanoDetConfig::default();
        let processor = NanoDetPostprocessor::new(config, 320, 320);
        
        // 均匀分布应该返回中间值
        let dis = vec![0.125; 8];
        let distance = processor.dfl_decode(&dis);
        assert!((distance - 3.5).abs() < 0.1);
    }
}
