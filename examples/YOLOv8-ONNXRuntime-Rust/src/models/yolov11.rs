// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// YOLOv11 模型实现 (改进的C3k2和SPPF模块)
// 特性: 比YOLOv8精度更高,速度相当
// 
// 注: YOLOv11与YOLOv8的ONNX接口完全兼容,
// 差异仅在网络结构内部(C3k2, SPPF改进),
// 因此直接复用YOLOv8的实现

use anyhow::Result;
use image::DynamicImage;

use crate::YOLOTask;

/// YOLOv11 模型结构 (内部委托给YOLOv8)
pub struct YOLOv11 {
    inner: crate::models::YOLOv8,
}

impl YOLOv11 {
    /// 从配置创建 YOLOv11 模型 (委托给YOLOv8)
    pub fn new(config: crate::Args) -> Result<Self> {
        let inner = crate::models::YOLOv8::new(config)?;
        Ok(Self { inner })
    }
}

impl crate::models::Model for YOLOv11 {
    /// 预处理: 委托给YOLOv8
    fn preprocess(&mut self, xs: &[DynamicImage]) -> Result<Vec<ndarray::Array<f32, ndarray::IxDyn>>> {
        let vec_xs = xs.to_vec();
        Ok(vec![self.inner.preprocess(&vec_xs)?])
    }

    /// 推理: 委托给YOLOv8
    fn run(&mut self, xs: Vec<ndarray::Array<f32, ndarray::IxDyn>>, profile: bool) -> Result<Vec<ndarray::Array<f32, ndarray::IxDyn>>> {
        Ok(xs.into_iter()
            .map(|x| self.inner.engine_mut().run(x, profile))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }

    /// 后处理: 委托给YOLOv8
    fn postprocess(&self, xs: Vec<ndarray::Array<f32, ndarray::IxDyn>>, xs0: &[DynamicImage]) -> Result<Vec<crate::DetectionResult>> {
        self.inner.postprocess(xs, xs0)
    }

    fn engine_mut(&mut self) -> &mut crate::OrtBackend {
        self.inner.engine_mut()
    }

    fn summary(&self) {
        println!("\n模型摘要:");
        println!("┌─────────────────────────────────────────┐");
        println!("│ Model: YOLOv11 (Improved Architecture)  │");
        println!("│ Backend: YOLOv8 (ONNX Compatible)       │");
        println!("└─────────────────────────────────────────┘");
        self.inner.summary();
    }

    fn supports_task(&self, task: YOLOTask) -> bool {
        self.inner.supports_task(task)
    }
}
