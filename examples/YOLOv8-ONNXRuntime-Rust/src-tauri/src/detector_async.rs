//! 异步检测器 - 使用 channel 避免阻塞主线程
//!
//! 数据流:
//! 前端发送帧 (Raw Request) → mpsc Channel → 后台检测线程 → Tauri Channel → 前端

use ndarray::{Array, IxDyn};
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
};
use ort::session::{builder::GraphOptimizationLevel, Session};
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Instant;
use tauri::ipc::Channel;

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

/// 帧数据 (发送给检测线程)
struct FrameData {
    rgba_data: Vec<u8>,
    width: u32,
    height: u32,
}

/// 异步检测器状态
pub struct AsyncDetectorState {
    // 发送帧到检测线程
    sender: RwLock<Option<Sender<FrameData>>>,
    // 模型输入尺寸
    input_width: RwLock<u32>,
    input_height: RwLock<u32>,
    // 是否正在运行
    is_running: RwLock<bool>,
    // 最新检测结果 (供前端轮询)
    last_result: RwLock<Option<DetectionResult>>,
    // 结果 Channel (发送到前端)
    result_channel: RwLock<Option<Channel<DetectionResult>>>,
}

impl Default for AsyncDetectorState {
    fn default() -> Self {
        Self {
            sender: RwLock::new(None),
            input_width: RwLock::new(640),
            input_height: RwLock::new(640),
            is_running: RwLock::new(false),
            last_result: RwLock::new(None),
            result_channel: RwLock::new(None),
        }
    }
}

impl AsyncDetectorState {
    /// 启动检测器 (后台线程)
    pub fn start(
        &self,
        model_path: &str,
        result_channel: Channel<DetectionResult>,
    ) -> Result<(u32, u32), String> {
        if *self.is_running.read() {
            return Err("检测器已在运行".to_string());
        }

        // 保存结果 Channel
        *self.result_channel.write() = Some(result_channel.clone());

        // 检查模型文件
        if !Path::new(model_path).exists() {
            return Err(format!("模型文件不存在: {}", model_path));
        }

        println!("🔄 正在加载模型: {}", model_path);

        // 选择最佳执行提供程序: CUDA > DirectML > CPU
        let (ep_name, providers) = Self::select_best_execution_provider();
        println!("🎯 使用执行提供程序: {}", ep_name);

        // 加载模型 - 启用多线程并行推理
        let session = Session::builder()
            .map_err(|e| format!("创建 Session Builder 失败: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("设置优化级别失败: {}", e))?
            .with_intra_threads(4) // 算子内并行 (4线程)
            .map_err(|e| format!("设置 intra_threads 失败: {}", e))?
            .with_inter_threads(2) // 算子间并行 (2线程)
            .map_err(|e| format!("设置 inter_threads 失败: {}", e))?
            .with_execution_providers(providers)
            .map_err(|e| format!("设置 {} EP 失败: {}", ep_name, e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("加载模型失败: {}", e))?;

        // 使用 640x640 提高检测精度 (Raw Request 优化后 IPC 开销可接受)
        let width = 640u32;
        let height = 640u32;
        *self.input_width.write() = width;
        *self.input_height.write() = height;

        println!("✅ 模型加载成功，使用 {}x{} 输入尺寸", width, height);

        // 创建 channel (只保留最新帧，丢弃旧帧)
        let (tx, rx): (Sender<FrameData>, Receiver<FrameData>) = mpsc::channel();
        *self.sender.write() = Some(tx);
        *self.is_running.write() = true;

        // 启动检测线程
        let conf_threshold = 0.25f32;
        let iou_threshold = 0.45f32;

        thread::spawn(move || {
            Self::detection_loop(
                session,
                rx,
                result_channel,
                conf_threshold,
                iou_threshold,
                width,
                height,
            );
        });

        Ok((width, height))
    }

    /// 检测线程主循环
    fn detection_loop(
        mut session: Session,
        rx: Receiver<FrameData>,
        result_channel: Channel<DetectionResult>,
        conf_threshold: f32,
        iou_threshold: f32,
        input_width: u32,
        input_height: u32,
    ) {
        println!("🚀 检测线程已启动");

        let mut frame_count = 0u64;
        let mut last_time = Instant::now();
        let mut current_fps = 0.0f64;
        let mut received_frames = 0u64;

        loop {
            // 接收帧 (阻塞等待)
            let frame = match rx.recv() {
                Ok(f) => f,
                Err(_) => {
                    println!("📭 Channel 已关闭，检测线程退出");
                    break;
                }
            };

            received_frames += 1;
            if received_frames <= 3 || received_frames % 10 == 0 {
                println!(
                    "📥 收到帧 #{}, 尺寸: {}x{}, 数据: {} bytes",
                    received_frames,
                    frame.width,
                    frame.height,
                    frame.rgba_data.len()
                );
            }

            // 丢弃队列中的旧帧，只处理最新的
            let mut latest_frame = frame;
            while let Ok(newer_frame) = rx.try_recv() {
                latest_frame = newer_frame;
            }

            // 验证帧尺寸
            if latest_frame.width != input_width || latest_frame.height != input_height {
                eprintln!(
                    "⚠️ 帧尺寸不匹配: 收到 {}x{}, 期望 {}x{}",
                    latest_frame.width, latest_frame.height, input_width, input_height
                );
                continue;
            }

            let start = Instant::now();

            // 执行检测 (前端已 resize 到正确尺寸)
            match Self::detect_frame_with_resize(
                &mut session,
                &latest_frame.rgba_data,
                latest_frame.width,
                latest_frame.height,
                input_width,
                input_height,
                conf_threshold,
                iou_threshold,
            ) {
                Ok(boxes) => {
                    let inference_time = start.elapsed().as_secs_f64() * 1000.0;

                    // 更新 FPS
                    frame_count += 1;
                    let now = Instant::now();
                    if now.duration_since(last_time).as_secs_f64() >= 1.0 {
                        current_fps = frame_count as f64;
                        frame_count = 0;
                        last_time = now;
                        // 打印检测统计
                        println!(
                            "📊 检测FPS: {:.1}, 推理: {:.1}ms, 检测到: {} 个目标",
                            current_fps,
                            inference_time,
                            boxes.len()
                        );
                    }

                    // 调试：前几帧总是打印
                    if received_frames <= 10 {
                        println!(
                            "🎯 帧 #{} 检测结果: {} 个目标, 耗时: {:.1}ms",
                            received_frames,
                            boxes.len(),
                            inference_time
                        );
                    }

                    let result = DetectionResult {
                        boxes,
                        inference_time_ms: inference_time,
                        detect_fps: current_fps,
                    };

                    // 通过 Channel 发送结果到前端 (比 emit 更高效)
                    match result_channel.send(result) {
                        Ok(_) => {
                            if received_frames <= 5 {
                                println!("✅ 检测结果已发送到前端");
                            }
                        }
                        Err(e) => {
                            println!("❌ 发送检测结果失败: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ 检测失败: {}", e);
                }
            }
        }

        println!("🛑 检测线程已退出");
    }

    /// 检测单帧 (优化版 - 前端已 resize 到正确尺寸，避免额外拷贝)
    fn detect_frame_with_resize(
        session: &mut Session,
        rgba_data: &[u8],
        src_width: u32,
        src_height: u32,
        model_width: u32,
        model_height: u32,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<DetectionBox>, String> {
        let src_pixel_count = (src_width * src_height) as usize;
        let expected_size = src_pixel_count * 4;

        if rgba_data.len() != expected_size {
            return Err(format!(
                "数据尺寸不匹配: 期望 {} bytes ({}x{}x4), 实际 {} bytes",
                expected_size,
                src_width,
                src_height,
                rgba_data.len()
            ));
        }

        // 前端已发送正确尺寸，直接使用（避免 resize 和额外拷贝）
        let (final_width, final_height) = (src_width, src_height);
        let pixel_count = (final_width * final_height) as usize;

        // 预分配输出缓冲区
        let mut input_data = vec![0f32; pixel_count * 3];

        // 优化: RGBA → CHW 格式转换 (并行 + 避免边界检查)
        // 分成 R、G、B 三个通道并行处理
        let (r_channel, rest) = input_data.split_at_mut(pixel_count);
        let (g_channel, b_channel) = rest.split_at_mut(pixel_count);

        // 并行处理三个通道
        rayon::scope(|s| {
            s.spawn(|_| {
                for i in 0..pixel_count {
                    r_channel[i] = rgba_data[i * 4] as f32 / 255.0;
                }
            });
            s.spawn(|_| {
                for i in 0..pixel_count {
                    g_channel[i] = rgba_data[i * 4 + 1] as f32 / 255.0;
                }
            });
            s.spawn(|_| {
                for i in 0..pixel_count {
                    b_channel[i] = rgba_data[i * 4 + 2] as f32 / 255.0;
                }
            });
        });

        // 创建输入张量
        let input_array = Array::from_shape_vec(
            IxDyn(&[1, 3, final_height as usize, final_width as usize]),
            input_data,
        )
        .map_err(|e| format!("创建输入张量失败: {}", e))?;

        // 创建 ort 输入值
        let input_value = ort::value::Value::from_array(input_array)
            .map_err(|e| format!("创建输入值失败: {}", e))?;

        // 推理
        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(|e| format!("推理失败: {}", e))?;

        // 解析输出
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

        // 构建 ndarray (避免 to_vec 拷贝，直接使用 slice)
        let output_array = Array::from_shape_vec(IxDyn(&dims), out_slice.to_vec())
            .map_err(|e| format!("构建输出数组失败: {}", e))?;

        let output_data = output_array.view();

        // YOLOv8 输出: [1, 84, 8400] -> 转置为 [8400, 84]
        let shape = output_data.shape();
        if shape.len() != 3 {
            return Err(format!("输出形状不正确: {:?}", shape));
        }

        let num_classes = shape[1] - 4;
        let num_detections = shape[2];

        let mut detections = Vec::new();

        for i in 0..num_detections {
            let x_center = output_data[[0, 0, i]];
            let y_center = output_data[[0, 1, i]];
            let w = output_data[[0, 2, i]];
            let h = output_data[[0, 3, i]];

            let mut max_score = 0f32;
            let mut max_class = 0usize;

            for c in 0..num_classes {
                let score = output_data[[0, 4 + c, i]];
                if score > max_score {
                    max_score = score;
                    max_class = c;
                }
            }

            if max_score >= conf_threshold {
                let x1 = (x_center - w / 2.0) / final_width as f32;
                let y1 = (y_center - h / 2.0) / final_height as f32;
                let x2 = (x_center + w / 2.0) / final_width as f32;
                let y2 = (y_center + h / 2.0) / final_height as f32;

                let class_name = COCO_CLASSES
                    .get(max_class)
                    .unwrap_or(&"unknown")
                    .to_string();

                detections.push(DetectionBox {
                    x1: x1.clamp(0.0, 1.0),
                    y1: y1.clamp(0.0, 1.0),
                    x2: x2.clamp(0.0, 1.0),
                    y2: y2.clamp(0.0, 1.0),
                    confidence: max_score,
                    class_id: max_class as u32,
                    class_name,
                    track_id: None,
                });
            }
        }

        // NMS
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        let mut keep = vec![true; detections.len()];

        for i in 0..detections.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..detections.len() {
                if !keep[j] {
                    continue;
                }
                if detections[i].class_id == detections[j].class_id {
                    let iou = Self::calculate_iou(&detections[i], &detections[j]);
                    if iou > iou_threshold {
                        keep[j] = false;
                    }
                }
            }
        }

        let final_detections: Vec<DetectionBox> = detections
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, d)| d)
            .collect();

        Ok(final_detections)
    }

    /// 计算 IoU
    fn calculate_iou(a: &DetectionBox, b: &DetectionBox) -> f32 {
        let inter_x1 = a.x1.max(b.x1);
        let inter_y1 = a.y1.max(b.y1);
        let inter_x2 = a.x2.min(b.x2);
        let inter_y2 = a.y2.min(b.y2);

        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
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

    /// 发送帧到检测线程 (非阻塞)
    pub fn send_frame(&self, rgba_data: Vec<u8>, width: u32, height: u32) -> Result<(), String> {
        // 调试：统计发送的帧
        static SEND_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = SEND_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 5 || count % 50 == 0 {
            println!(
                "📤 send_frame #{}: {}x{}, {} bytes",
                count,
                width,
                height,
                rgba_data.len()
            );
        }

        let sender = self.sender.read();
        if let Some(tx) = sender.as_ref() {
            // 非阻塞发送，如果队列满了就丢弃
            match tx.send(FrameData {
                rgba_data,
                width,
                height,
            }) {
                Ok(_) => {
                    if count < 5 {
                        println!("✅ 帧已发送到检测线程");
                    }
                }
                Err(e) => {
                    println!("❌ 发送帧失败: {}", e);
                }
            }
            Ok(())
        } else {
            println!("⚠️ send_frame: 检测器未启动 (sender 为空)");
            Err("检测器未启动".to_string())
        }
    }

    /// 停止检测器
    pub fn stop(&self) {
        *self.sender.write() = None; // 关闭 channel，线程会自动退出
        *self.is_running.write() = false;
        println!("🛑 检测器已停止");
    }

    /// 获取输入尺寸
    pub fn get_input_size(&self) -> (u32, u32) {
        (*self.input_width.read(), *self.input_height.read())
    }

    /// 是否正在运行
    pub fn is_running(&self) -> bool {
        *self.is_running.read()
    }

    /// 从模型获取输入尺寸
    fn get_model_input_size(session: &Session) -> Result<(u32, u32), String> {
        use ort::value::ValueType;

        if session.inputs.is_empty() {
            return Err("模型没有输入".to_string());
        }

        let input = &session.inputs[0];
        if let ValueType::Tensor { shape, .. } = &input.input_type {
            // YOLO 输入格式: [batch, channels, height, width]
            if shape.len() >= 4 {
                let height = shape[2];
                let width = shape[3];

                if height > 0 && width > 0 {
                    println!("📐 从模型读取输入尺寸: {}x{}", width, height);
                    return Ok((width as u32, height as u32));
                }
            }
        }

        // 默认 640x640
        println!("⚠️ 无法从模型读取尺寸，使用默认 640x640");
        Ok((640, 640))
    }

    /// 选择最佳执行提供程序: DirectML > CUDA > CPU
    /// (DirectML 在 Windows 上更可靠，不需要额外安装 CUDA SDK)
    fn select_best_execution_provider() -> (
        &'static str,
        Vec<ort::execution_providers::ExecutionProviderDispatch>,
    ) {
        use ort::execution_providers::ExecutionProviderDispatch;

        // 1. 优先尝试 DirectML (Windows GPU 通用加速 - AMD/Intel/NVIDIA 都支持，无需额外安装)
        let dml = DirectMLExecutionProvider::default();
        if dml.is_available().unwrap_or(false) {
            println!("✅ DirectML 可用 (Windows GPU 加速)");
            return ("DirectML", vec![ExecutionProviderDispatch::from(dml)]);
        }
        println!("⚠️ DirectML 不可用");

        // 2. 尝试 CUDA (需要 NVIDIA GPU + CUDA SDK + cuDNN)
        let cuda = CUDAExecutionProvider::default();
        if cuda.is_available().unwrap_or(false) {
            println!("✅ CUDA 可用");
            return ("CUDA", vec![ExecutionProviderDispatch::from(cuda)]);
        }
        println!("⚠️ CUDA 不可用");

        // 3. 回退到 CPU
        println!("ℹ️ 使用 CPU 执行");
        (
            "CPU",
            vec![ExecutionProviderDispatch::from(
                CPUExecutionProvider::default(),
            )],
        )
    }
}
