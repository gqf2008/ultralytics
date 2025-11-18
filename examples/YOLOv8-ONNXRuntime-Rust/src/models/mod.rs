/// 模型统一接口与实现
///
/// # 架构说明
///
/// ## 完整模型实现 (Full Model Implementation)
/// - **YOLOv8**: 包含完整的 struct + impl Model trait
///   - 模型加载 (new)
///   - 预处理 (preprocess)  
///   - 推理 (run)
///   - 后处理 (postprocess)
///   - 文件: `yolov8.rs`
///
/// ## 后处理器模式 (Postprocessor Pattern)  
/// - **FastestV2/NanoDet**: 仅实现后处理器
///   - 通过 `detection::PostprocessorFactory` 统一管理
///   - 模型加载/预处理由 `detector.rs` 中的 `OrtBackend` 处理
///   - 适用于轻量级模型或特定场景
///   - 文件: `fastestv2.rs`, `nanodet.rs`
///
/// ## Model Trait
/// 统一的模型接口，定义标准流程: preprocess → run → postprocess
///
/// ## 使用示例
/// ```rust
/// use yolov8_rs::models::{Model, YOLOv8};
/// use yolov8_rs::Args;
///
/// // 方式1: 使用完整模型 (推荐)
/// let mut model = YOLOv8::new(args)?;
/// let results = model.run(&images)?;
///
/// // 方式2: 使用 Model trait (灵活)
/// let results = model.forward(&images)?;
/// ```
use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, IxDyn};

use crate::{DetectionResult, OrtBackend, YOLOTask};

/// 模型类型枚举（用于自动识别模型）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// YOLOv8 标准模型
    YOLOv8,
    /// YOLOv5 模型
    YOLOv5,
    /// YOLOv10 端到端模型 (NMS-Free)
    YOLOv10,
    /// YOLOv11 改进模型 (C3k2 + SPPF)
    YOLOv11,
    /// YOLOX 无锚点模型
    YOLOX,
    /// YOLO-FastestV2 超轻量级模型
    FastestV2,
    /// NanoDet 系列模型
    NanoDet,
}

impl ModelType {
    /// 从模型路径推断模型类型
    pub fn from_path(path: &str) -> Self {
        if path.contains("yolov10") || path.contains("v10") {
            ModelType::YOLOv10
        } else if path.contains("yolov11") || path.contains("v11") {
            ModelType::YOLOv11
        } else if path.contains("yolox") {
            ModelType::YOLOX
        } else if path.contains("fastestv2") {
            ModelType::FastestV2
        } else if path.contains("nanodet") {
            ModelType::NanoDet
        } else if path.contains("v5") {
            ModelType::YOLOv5
        } else {
            ModelType::YOLOv8
        }
    }

    /// 获取模型推荐的置信度阈值
    pub fn default_conf_threshold(&self) -> f32 {
        match self {
            ModelType::YOLOv10 => 0.20, // v10端到端模型已过滤
            ModelType::YOLOv11 => 0.15, // v11与v8相同
            ModelType::YOLOX => 0.25,
            ModelType::FastestV2 => 0.10,
            ModelType::NanoDet => 0.35,
            ModelType::YOLOv5 => 0.25,
            ModelType::YOLOv8 => 0.15,
        }
    }

    /// 获取模型推荐的IOU阈值
    pub fn default_iou_threshold(&self) -> f32 {
        match self {
            ModelType::NanoDet => 0.6,
            _ => 0.45,
        }
    }
}

/// 统一的深度学习模型接口
///
/// 所有模型(YOLOv8, YOLOv5, FastestV2, NanoDet等)都应实现此 trait
///
/// ## 核心流程
/// ```text
/// 原始图片 → preprocess → ndarray张量
///          ↓
///     推理引擎 run
///          ↓
///     原始输出 → postprocess → 检测结果
/// ```
pub trait Model {
    /// 预处理: 图片 → ndarray 张量
    ///
    /// # Arguments
    /// * `images` - 输入图片数组(支持批量处理)
    ///
    /// # Returns
    /// * `Vec<Array<f32, IxDyn>>` - NCHW格式的张量,准备送入推理引擎
    fn preprocess(&mut self, images: &[DynamicImage]) -> Result<Vec<Array<f32, IxDyn>>>;

    /// 推理: 执行模型前向传播
    ///
    /// # Arguments
    /// * `xs` - 预处理后的张量
    /// * `profile` - 是否启用性能分析
    ///
    /// # Returns
    /// * `Vec<Array<f32, IxDyn>>` - 模型原始输出(未解码)
    fn run(&mut self, xs: Vec<Array<f32, IxDyn>>, profile: bool) -> Result<Vec<Array<f32, IxDyn>>>;

    /// 后处理: 原始输出 → 检测结果
    ///
    /// # Arguments
    /// * `xs` - 模型原始输出
    /// * `xs0` - 原始图片(用于坐标还原)
    ///
    /// # Returns
    /// * `Vec<DetectionResult>` - 解码后的检测结果(bbox、keypoint、mask等)
    fn postprocess(
        &self,
        xs: Vec<Array<f32, IxDyn>>,
        xs0: &[DynamicImage],
    ) -> Result<Vec<DetectionResult>>;

    /// 完整的推理流程: preprocess → run → postprocess
    ///
    /// 默认实现调用上面三个方法,子类可以重写以优化性能
    fn forward(&mut self, images: &[DynamicImage]) -> Result<Vec<DetectionResult>> {
        let xs = self.preprocess(images)?;
        let ys = self.run(xs, false)?;
        self.postprocess(ys, images)
    }

    /// 获取底层推理引擎的可变引用
    ///
    /// 用于直接调用 OrtBackend::run (绕过 Model::run 的封装)
    fn engine_mut(&mut self) -> &mut OrtBackend;

    /// 打印模型信息
    fn summary(&self);

    /// 检查模型是否支持指定任务
    ///
    /// # Arguments
    /// * `task` - 要检查的任务类型(Detect/Pose/Segment/Classify)
    ///
    /// # Returns
    /// * `bool` - true表示支持该任务, false表示不支持
    ///
    /// # Examples
    /// ```
    /// if model.supports_task(YOLOTask::Pose) {
    ///     // 执行姿态估计
    /// }
    /// ```
    fn supports_task(&self, task: YOLOTask) -> bool;
}

// 各模型的具体实现
pub mod fastestv2;
pub mod nanodet;
pub mod yolov10; // YOLOv10 端到端模型 (NMS-Free)
pub mod yolov11; // YOLOv11 改进模型
pub mod yolov8; // YOLOv8 完整模型 + 实现 Model trait
pub mod yolox; // YOLOX 无锚点模型

// Re-exports
pub use fastestv2::{FastestV2, FastestV2Config, FastestV2Postprocessor};
pub use nanodet::{NanoDet, NanoDetConfig, NanoDetPostprocessor};
pub use yolov10::YOLOv10;
pub use yolov11::YOLOv11;
pub use yolov8::{YOLOv8, YOLOv8Config, YOLOv8Postprocessor};
pub use yolox::YOLOX;
