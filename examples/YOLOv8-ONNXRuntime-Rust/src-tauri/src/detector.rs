//! 轻量级 YOLO 检测器 - 直接使用 ort (ONNX Runtime)
//! 不依赖主项目，避免 FFmpeg 链接问题
//!
//! 数据流:
//! 前端 WebCodecs (GPU硬解码) → Canvas Resize 640x640 → RGBA Uint8Array
//!     → Tauri IPC → ort ONNX Runtime (CUDA/CPU) → 检测结果 JSON → 前端绘制

use ndarray::{Array, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

/// 检测结果 (发送到前端)
#[derive(Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub boxes: Vec<DetectionBox>,
    pub inference_time_ms: f64,
    pub detect_fps: f64,
}

/// 检测框
#[derive(Clone, Serialize, Deserialize)]
pub struct DetectionBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
    pub class_name: String,
    pub track_id: Option<u32>,
}

/// COCO 类别名称
const COCO_CLASSES: &[&str] = &[
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
];

/// 检测器状态
pub struct DetectorState {
    session: RwLock<Option<Session>>,
    model_name: RwLock<String>,
    conf_threshold: RwLock<f32>,
    iou_threshold: RwLock<f32>,
    // 模型输入尺寸
    input_width: RwLock<u32>,
    input_height: RwLock<u32>,
    // 统计
    frame_count: RwLock<u64>,
    last_time: RwLock<Instant>,
    current_fps: RwLock<f64>,
}

impl Default for DetectorState {
    fn default() -> Self {
        Self {
            session: RwLock::new(None),
            model_name: RwLock::new(String::new()),
            conf_threshold: RwLock::new(0.25),
            iou_threshold: RwLock::new(0.45),
            input_width: RwLock::new(640),  // 默认值
            input_height: RwLock::new(640), // 默认值
            frame_count: RwLock::new(0),
            last_time: RwLock::new(Instant::now()),
            current_fps: RwLock::new(0.0),
        }
    }
}

impl DetectorState {
    /// 加载 ONNX 模型，返回模型需要的输入尺寸
    pub fn load_model(&self, model_path: &str) -> Result<(u32, u32), String> {
        // 检查文件是否存在
        if !Path::new(model_path).exists() {
            return Err(format!("模型文件不存在: {}", model_path));
        }

        println!("🔄 正在加载模型: {}", model_path);

        // 创建 ONNX Runtime Session
        let session = Session::builder()
            .map_err(|e| format!("创建 Session Builder 失败: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("设置优化级别失败: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("加载模型失败: {}", e))?;

        // 固定使用 640x640 尺寸
        let width = 640u32;
        let height = 640u32;

        if let Some(input) = session.inputs.first() {
            println!("📐 模型输入: {} - {:?}", input.name, input.input_type);
        }

        println!("📐 使用固定输入尺寸: {}x{}", width, height);

        println!(
            "✅ 模型加载成功: {} (输入尺寸: {}x{})",
            model_path, width, height
        );

        // 保存尺寸
        *self.input_width.write() = width;
        *self.input_height.write() = height;

        *self.session.write() = Some(session);
        *self.model_name.write() = model_path.to_string();

        Ok((width, height))
    }

    /// 获取模型需要的输入尺寸
    pub fn get_input_size(&self) -> (u32, u32) {
        (*self.input_width.read(), *self.input_height.read())
    }

    /// 停止检测器
    pub fn stop(&self) {
        *self.session.write() = None;
        println!("🛑 检测器已停止");
    }

    /// 初始化追踪器 (预留接口)
    pub fn init_tracker(&self, tracker_type: &str) {
        // TODO: 实现追踪器 (ByteTrack/DeepSORT)
        println!("📦 追踪器配置: {} (暂未实现)", tracker_type);
    }

    /// 检测帧 (接收前端已 resize 的 640x640 RGBA 数据)
    pub fn detect(
        &self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<DetectionResult, String> {
        let start = Instant::now();

        // 使用 write() 获取可变引用，因为 Session::run 需要 &mut self
        let mut session_guard = self.session.write();
        let session = session_guard.as_mut().ok_or("模型未加载")?;

        // 1. RGBA → RGB 并归一化到 [0, 1]
        let pixel_count = (width * height) as usize;
        let mut input_data = vec![0f32; pixel_count * 3];

        // 并行处理: RGBA → CHW 格式 (Channel, Height, Width)
        // YOLOv8 期望的输入格式: [1, 3, 640, 640]
        input_data
            .par_chunks_exact_mut(pixel_count)
            .enumerate()
            .for_each(|(c, channel)| {
                for i in 0..pixel_count {
                    let rgba_idx = i * 4 + c; // R=0, G=1, B=2
                    channel[i] = rgba_data[rgba_idx] as f32 / 255.0;
                }
            });

        // 2. 创建输入张量 [1, 3, H, W]
        let input_array =
            Array::from_shape_vec(IxDyn(&[1, 3, height as usize, width as usize]), input_data)
                .map_err(|e| format!("创建输入张量失败: {}", e))?;

        // 3. 创建 ort 输入值
        let input_value = ort::value::Value::from_array(input_array)
            .map_err(|e| format!("创建输入值失败: {}", e))?;

        // 4. 推理
        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| format!("推理失败: {}", e))?;

        // 5. 解析输出 - YOLOv8 输出格式: [1, 84, 8400] 或 [1, 8400, 84]
        let output = outputs
            .iter()
            .next()
            .map(|(_name, value)| value)
            .ok_or("找不到输出")?;

        // try_extract_tensor 返回 (Shape, &[f32])
        let (out_shape, out_slice) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("提取输出失败: {}", e))?;

        // Shape 实现了 Deref<Target = [i64]>，可以直接迭代
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();

        // 构建 ndarray
        let output_array = Array::from_shape_vec(IxDyn(&dims), out_slice.to_vec())
            .map_err(|e| format!("构建输出数组失败: {}", e))?;

        let output_view = output_array.view();
        let shape = output_view.shape();

        // 5. 后处理 - 提取检测框
        let conf_threshold = *self.conf_threshold.read();
        let iou_threshold = *self.iou_threshold.read();

        let mut boxes = self.parse_yolov8_output(&output_view, shape, conf_threshold)?;

        // 6. NMS
        self.non_max_suppression(&mut boxes, iou_threshold);

        // 7. 更新统计
        let inference_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut count = self.frame_count.write();
        *count += 1;

        let mut last_time = self.last_time.write();
        let now = Instant::now();
        if now.duration_since(*last_time).as_secs() >= 1 {
            let mut fps = self.current_fps.write();
            *fps = *count as f64 / now.duration_since(*last_time).as_secs_f64();
            *count = 0;
            *last_time = now;
        }

        Ok(DetectionResult {
            boxes,
            inference_time_ms: inference_ms,
            detect_fps: *self.current_fps.read(),
        })
    }

    /// 解析 YOLOv8 输出
    fn parse_yolov8_output(
        &self,
        output: &ndarray::ArrayView<f32, IxDyn>,
        shape: &[usize],
        conf_threshold: f32,
    ) -> Result<Vec<DetectionBox>, String> {
        let mut boxes = Vec::new();

        // YOLOv8 输出: [1, 84, 8400] - 需要转置为 [1, 8400, 84]
        // 84 = 4 (xywh) + 80 (classes)

        if shape.len() != 3 {
            return Err(format!("不支持的输出维度: {:?}", shape));
        }

        let (_batch, dim1, dim2) = (shape[0], shape[1], shape[2]);

        // 判断输出格式
        let (num_boxes, num_attrs) = if dim1 == 84 || dim1 == 85 {
            // [1, 84, 8400] 格式 - 需要转置
            (dim2, dim1)
        } else {
            // [1, 8400, 84] 格式
            (dim1, dim2)
        };

        let _num_classes = num_attrs - 4;

        for i in 0..num_boxes {
            // 获取当前框的数据
            let (cx, cy, w, h, class_scores) = if dim1 == 84 || dim1 == 85 {
                // 转置格式: output[0, attr, box_idx]
                let cx = output[[0, 0, i]];
                let cy = output[[0, 1, i]];
                let w = output[[0, 2, i]];
                let h = output[[0, 3, i]];
                let scores: Vec<f32> = (4..num_attrs).map(|j| output[[0, j, i]]).collect();
                (cx, cy, w, h, scores)
            } else {
                // 标准格式: output[0, box_idx, attr]
                let cx = output[[0, i, 0]];
                let cy = output[[0, i, 1]];
                let w = output[[0, i, 2]];
                let h = output[[0, i, 3]];
                let scores: Vec<f32> = (4..num_attrs).map(|j| output[[0, i, j]]).collect();
                (cx, cy, w, h, scores)
            };

            // 找到最大类别置信度
            let (class_id, &confidence) = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if confidence < conf_threshold {
                continue;
            }

            // xywh → xyxy
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            let class_name = COCO_CLASSES.get(class_id).unwrap_or(&"unknown").to_string();

            boxes.push(DetectionBox {
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id: class_id as u32,
                class_name,
                track_id: None,
            });
        }

        Ok(boxes)
    }

    /// 非极大值抑制 (NMS)
    fn non_max_suppression(&self, boxes: &mut Vec<DetectionBox>, iou_threshold: f32) {
        // 按置信度排序
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = vec![true; boxes.len()];

        for i in 0..boxes.len() {
            if !keep[i] {
                continue;
            }

            for j in (i + 1)..boxes.len() {
                if !keep[j] {
                    continue;
                }

                let iou = self.calculate_iou(&boxes[i], &boxes[j]);
                if iou > iou_threshold {
                    keep[j] = false;
                }
            }
        }

        let mut idx = 0;
        boxes.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }

    /// 计算 IoU
    fn calculate_iou(&self, a: &DetectionBox, b: &DetectionBox) -> f32 {
        let x1 = a.x1.max(b.x1);
        let y1 = a.y1.max(b.y1);
        let x2 = a.x2.min(b.x2);
        let y2 = a.y2.min(b.y2);

        let inter_w = (x2 - x1).max(0.0);
        let inter_h = (y2 - y1).max(0.0);
        let inter_area = inter_w * inter_h;

        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        let union_area = area_a + area_b - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }
}
