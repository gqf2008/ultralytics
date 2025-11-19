#![allow(clippy::type_complexity)]
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
pub mod config; // 模型配置参数
pub mod detection; // 智能检测系统
pub mod input; // 视频输入系统
pub mod models; // 模型接口与具体实现
pub mod ui_config; // UI配置面板

pub mod ort_backend;
// pub mod renderer; // ggez 版本的 renderer (旧版)
// macroquad 版本的 renderer 在 bin/sentinel_macroquad.rs 中直接引用
pub mod xbus;

pub use crate::config::Args;
pub use crate::models::{
    FastestV2Config, FastestV2Postprocessor, Model, NanoDetConfig, NanoDetPostprocessor, YOLOv8,
};
pub use crate::ort_backend::{Batch, OrtBackend, OrtConfig, OrtEP, YOLOTask};

pub fn non_max_suppression(
    xs: &mut Vec<(Bbox, Option<Vec<Point2>>, Option<Vec<f32>>)>,
    iou_threshold: f32,
) {
    xs.sort_by(|b1, b2| b2.0.confidence().partial_cmp(&b1.0.confidence()).unwrap());

    let mut current_index = 0;
    for index in 0..xs.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            let iou = xs[prev_index].0.iou(&xs[index].0);
            if iou > iou_threshold {
                drop = true;
                break;
            }
        }
        if !drop {
            xs.swap(current_index, index);
            current_index += 1;
        }
    }
    xs.truncate(current_index);
}

pub fn gen_time_string(delimiter: &str) -> String {
    let offset = chrono::FixedOffset::east_opt(8 * 60 * 60).unwrap(); // Beijing
    let t_now = chrono::Utc::now().with_timezone(&offset);
    let fmt = format!(
        "%Y{}%m{}%d{}%H{}%M{}%S{}%f",
        delimiter, delimiter, delimiter, delimiter, delimiter, delimiter
    );
    t_now.format(&fmt).to_string()
}

pub const SKELETON: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

use ndarray::{Array, Axis, IxDyn};

#[derive(Clone, PartialEq, Default)]
pub struct DetectionResult {
    // YOLO tasks results of an image
    pub probs: Option<Embedding>,
    pub bboxes: Option<Vec<Bbox>>,
    pub keypoints: Option<Vec<Vec<Point2>>>,
    pub masks: Option<Vec<Vec<u8>>>,
}

impl std::fmt::Debug for DetectionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOResult")
            .field(
                "Probs(top5)",
                &format_args!("{:?}", self.probs().map(|probs| probs.topk(5))),
            )
            .field("Bboxes", &self.bboxes)
            .field("Keypoints", &self.keypoints)
            .field(
                "Masks",
                &format_args!("{:?}", self.masks().map(|masks| masks.len())),
            )
            .finish()
    }
}

impl DetectionResult {
    pub fn new(
        probs: Option<Embedding>,
        bboxes: Option<Vec<Bbox>>,
        keypoints: Option<Vec<Vec<Point2>>>,
        masks: Option<Vec<Vec<u8>>>,
    ) -> Self {
        Self {
            probs,
            bboxes,
            keypoints,
            masks,
        }
    }

    pub fn probs(&self) -> Option<&Embedding> {
        self.probs.as_ref()
    }

    pub fn keypoints(&self) -> Option<&Vec<Vec<Point2>>> {
        self.keypoints.as_ref()
    }

    pub fn masks(&self) -> Option<&Vec<Vec<u8>>> {
        self.masks.as_ref()
    }

    pub fn bboxes(&self) -> Option<&Vec<Bbox>> {
        self.bboxes.as_ref()
    }

    pub fn bboxes_mut(&mut self) -> Option<&mut Vec<Bbox>> {
        self.bboxes.as_mut()
    }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Point2 {
    // A point2d with x, y, conf
    x: f32,
    y: f32,
    confidence: f32,
}

impl Point2 {
    pub fn new_with_conf(x: f32, y: f32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }

    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            ..Default::default()
        }
    }

    pub fn x(&self) -> f32 {
        self.x
    }

    pub fn y(&self) -> f32 {
        self.y
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Embedding {
    // An float32 n-dims tensor
    data: Array<f32, IxDyn>,
}

impl Embedding {
    pub fn new(data: Array<f32, IxDyn>) -> Self {
        Self { data }
    }

    pub fn data(&self) -> &Array<f32, IxDyn> {
        &self.data
    }

    pub fn topk(&self, k: usize) -> Vec<(usize, f32)> {
        let mut probs = self
            .data
            .iter()
            .enumerate()
            .map(|(a, b)| (a, *b))
            .collect::<Vec<_>>();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut topk = Vec::new();
        for &(id, confidence) in probs.iter().take(k) {
            topk.push((id, confidence));
        }
        topk
    }

    pub fn norm(&self) -> Array<f32, IxDyn> {
        let std_ = self.data.mapv(|x| x * x).sum_axis(Axis(0)).mapv(f32::sqrt);
        self.data.clone() / std_
    }

    pub fn top1(&self) -> (usize, f32) {
        self.topk(1)[0]
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bbox {
    // a bounding box around an object
    xmin: f32,
    ymin: f32,
    width: f32,
    height: f32,
    id: usize,
    confidence: f32,
}

impl Bbox {
    pub fn new_from_xywh(xmin: f32, ymin: f32, width: f32, height: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            ..Default::default()
        }
    }

    pub fn new(xmin: f32, ymin: f32, width: f32, height: f32, id: usize, confidence: f32) -> Self {
        Self {
            xmin,
            ymin,
            width,
            height,
            id,
            confidence,
        }
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.height
    }

    pub fn xmin(&self) -> f32 {
        self.xmin
    }

    pub fn ymin(&self) -> f32 {
        self.ymin
    }

    pub fn xmax(&self) -> f32 {
        self.xmin + self.width
    }

    pub fn ymax(&self) -> f32 {
        self.ymin + self.height
    }

    pub fn tl(&self) -> Point2 {
        Point2::new(self.xmin, self.ymin)
    }

    pub fn br(&self) -> Point2 {
        Point2::new(self.xmax(), self.ymax())
    }

    pub fn cxcy(&self) -> Point2 {
        Point2::new(self.xmin + self.width / 2., self.ymin + self.height / 2.)
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    pub fn intersection_area(&self, another: &Bbox) -> f32 {
        let l = self.xmin.max(another.xmin);
        let r = (self.xmin + self.width).min(another.xmin + another.width);
        let t = self.ymin.max(another.ymin);
        let b = (self.ymin + self.height).min(another.ymin + another.height);
        (r - l + 1.).max(0.) * (b - t + 1.).max(0.)
    }

    pub fn union(&self, another: &Bbox) -> f32 {
        self.area() + another.area() - self.intersection_area(another)
    }

    pub fn iou(&self, another: &Bbox) -> f32 {
        self.intersection_area(another) / self.union(another)
    }
}
