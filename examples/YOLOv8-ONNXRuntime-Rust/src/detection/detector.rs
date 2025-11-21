//! 检测器 (Detector)
//! 职责: 订阅DecodedFrame → YOLO检测 → 发送DetectionResult消息

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use fast_image_resize as fr;
use image::{DynamicImage, ImageBuffer, RgbImage, Rgba};

use super::types::DecodedFrame;
use super::{ByteTracker, PersonTracker};
use crate::detection::types::{self, ControlMessage};
use crate::models::{FastestV2, Model, ModelType, NanoDet, YOLOv10, YOLOv11, YOLOv8, YOLOX};
use crate::{xbus, Args, YOLOTask};

#[cfg(feature = "gpu")]
use crate::utils::affine_transform::{AffineMatrix, BorderMode, InterpolationMethod};
#[cfg(feature = "gpu")]
use crate::utils::affine_transform_wgpu::WgpuAffineTransform;

/// 检测结果 (检测模块 → 渲染模块)
#[derive(Clone, Debug)]
pub struct DetectionResult {
    pub bboxes: Vec<types::BBox>,
    pub keypoints: Vec<types::PoseKeypoints>,
    pub inference_fps: f64,
    pub inference_ms: f64,
    pub tracker_fps: f64,               // 追踪器FPS
    pub tracker_ms: f64,                // 追踪器耗时
    pub resized_image: Option<Vec<u8>>, // Resize后的RGB图像数据 (用于右下角显示)
    pub resized_size: u32,              // Resize后的图像尺寸
    pub reid_features: Vec<Vec<f32>>,   // 每个bbox对应的ReID特征向量
}

/// 跟踪器类型
enum TrackerType {
    DeepSort(PersonTracker),
    ByteTrack(ByteTracker),
    None,
}

pub struct Detector {
    detect_model_path: String,
    inf_size: u32,
    tracker: TrackerType,
    pose_enabled: bool,
    detection_enabled: bool,
    config_rx: Option<Receiver<ControlMessage>>,

    // Resize优化: 预计算的映射表
    resize_x_map: Vec<usize>,
    resize_y_map: Vec<usize>,
    src_width: usize,
    src_height: usize,

    // GPU加速支持
    #[cfg(feature = "gpu")]
    gpu_transform: Option<WgpuAffineTransform>,

    // 统计
    count: u64,
    last: Instant,
    current_fps: f64,

    // 跟踪统计
    tracker_count: u64,
    tracker_last: Instant,
    tracker_current_fps: f64,
}
impl Detector {
    pub fn new(
        detect_model: String,
        inf_size: u32,
        tracker_name: String,
        pose_enabled: bool,
    ) -> Self {
        // 根据跟踪器名称初始化
        let tracker = match tracker_name.to_lowercase().as_str() {
            "deepsort" => {
                println!("🎯 跟踪器: DeepSort (级联匹配 + 外观特征)");
                TrackerType::DeepSort(PersonTracker::new())
            }
            "bytetrack" => {
                println!("🎯 跟踪器: ByteTrack (高低分分开处理)");
                TrackerType::ByteTrack(ByteTracker::new())
            }
            _ => {
                println!("🎯 跟踪器: 禁用");
                TrackerType::None
            }
        };

        Self {
            detect_model_path: detect_model,
            inf_size,
            tracker,
            pose_enabled,
            detection_enabled: true,
            config_rx: None,
            // 初始化为空映射表,首帧时更新
            resize_x_map: Vec::new(),
            resize_y_map: Vec::new(),
            src_width: 0,
            src_height: 0,
            // 尝试初始化GPU加速
            #[cfg(feature = "gpu")]
            gpu_transform: WgpuAffineTransform::new().ok(),
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            tracker_count: 0,
            tracker_last: Instant::now(),
            tracker_current_fps: 0.0,
        }
    }

    /// CPU并行resize (RGBA → RGB + 缩放)
    fn cpu_resize_rgba_to_rgb(
        src_buffer: &[u8],
        src_w: usize,
        src_h: usize,
        dst_size: usize,
        x_map: &mut Vec<usize>,
        y_map: &mut Vec<usize>,
        cached_w: &mut usize,
        cached_h: &mut usize,
    ) -> Vec<u8> {
        use rayon::prelude::*;

        // 仅在分辨率变化时重新计算映射表
        if *cached_w != src_w || *cached_h != src_h {
            let scale_x = src_w as f32 / dst_size as f32;
            let scale_y = src_h as f32 / dst_size as f32;

            *x_map = (0..dst_size)
                .map(|x| ((x as f32 * scale_x) as usize).min(src_w - 1))
                .collect();
            *y_map = (0..dst_size)
                .map(|y| ((y as f32 * scale_y) as usize).min(src_h - 1))
                .collect();
            *cached_w = src_w;
            *cached_h = src_h;
            eprintln!(
                "📐 CPU Resize映射表已更新: {}x{} → {}",
                src_w, src_h, dst_size
            );
        }

        // 预分配输出
        let mut rgb_data = vec![0u8; dst_size * dst_size * 3];

        // 并行处理每一行 - 极致优化版本
        rgb_data
            .par_chunks_exact_mut(dst_size * 3)
            .enumerate()
            .for_each(|(y, row_chunk)| {
                let src_y = y_map[y];
                let src_row_base = src_y * src_w * 4;

                // 手动展开循环 + 避免边界检查
                let mut out_idx = 0;
                for &src_x in x_map.iter() {
                    let src_idx = src_row_base + src_x * 4;
                    unsafe {
                        // 使用unsafe避免边界检查 (映射表已保证安全)
                        *row_chunk.get_unchecked_mut(out_idx) = *src_buffer.get_unchecked(src_idx);
                        *row_chunk.get_unchecked_mut(out_idx + 1) =
                            *src_buffer.get_unchecked(src_idx + 1);
                        *row_chunk.get_unchecked_mut(out_idx + 2) =
                            *src_buffer.get_unchecked(src_idx + 2);
                    }
                    out_idx += 3;
                }
            });

        rgb_data
    }

    pub fn set_config_receiver(&mut self, rx: Receiver<ControlMessage>) {
        self.config_rx = Some(rx);
    }

    fn load_model(&self, model_path: &str) -> Option<Arc<Mutex<Box<dyn Model>>>> {
        // 识别模型类型
        let model_type = ModelType::from_path(model_path);

        // 加载检测模型
        let detect_args = Args {
            model: model_path.to_string(),
            width: Some(self.inf_size),
            height: Some(self.inf_size),
            conf: model_type.default_conf_threshold(),
            iou: model_type.default_iou_threshold(),
            source: String::new(),
            device_id: 0,
            trt: false,
            cuda: false,
            batch: 1,
            batch_min: 1,
            batch_max: 1,
            fp16: false,
            task: Some(YOLOTask::Detect),
            nc: None,
            nk: None,
            nm: None,
            kconf: 0.55,
            profile: false,
        };

        match model_type {
            ModelType::YOLOv8 | ModelType::YOLOv5 => match YOLOv8::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLOv8/v5 检测模型加载成功: {}", model_path);
                    Some(Arc::new(Mutex::new(Box::new(m))))
                }
                Err(e) => {
                    eprintln!("❌ YOLOv8/v5 模型加载失败: {}", e);
                    None
                }
            },
            ModelType::FastestV2 => match FastestV2::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLO-FastestV2 检测模型加载成功");
                    Some(Arc::new(Mutex::new(Box::new(m))))
                }
                Err(e) => {
                    eprintln!("❌ FastestV2 模型加载失败: {}", e);
                    None
                }
            },
            ModelType::NanoDet => match NanoDet::new(detect_args) {
                Ok(m) => {
                    println!("✅ NanoDet 检测模型加载成功");
                    Some(Arc::new(Mutex::new(Box::new(m))))
                }
                Err(e) => {
                    eprintln!("❌ NanoDet 模型加载失败: {}", e);
                    None
                }
            },
            ModelType::YOLOv10 => match YOLOv10::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLOv10 检测模型加载成功");
                    Some(Arc::new(Mutex::new(Box::new(m))))
                }
                Err(e) => {
                    eprintln!("❌ YOLOv10 模型加载失败: {}", e);
                    None
                }
            },
            ModelType::YOLOv11 => match YOLOv11::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLOv11 检测模型加载成功");
                    Some(Arc::new(Mutex::new(Box::new(m))))
                }
                Err(e) => {
                    eprintln!("❌ YOLOv11 模型加载失败: {}", e);
                    None
                }
            },
            ModelType::YOLOX => match YOLOX::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLOX 检测模型加载成功");
                    Some(Arc::new(Mutex::new(Box::new(m))))
                }
                Err(e) => {
                    eprintln!("❌ YOLOX 模型加载失败: {}", e);
                    None
                }
            },
        }
    }

    pub fn run(&mut self) {
        println!("🔍 检测模块启动");

        // 延迟加载模型 - 等待第一帧数据时才加载
        let mut detect_model: Option<Arc<Mutex<Box<dyn Model>>>> = None;
        let mut model_loaded = false;

        // 订阅解码帧 - 仅将任务放入队列
        let inf_size = self.inf_size;
        // 进一步减小队列长度以降低内存占用 (5 -> 2)
        // 牺牲少量延迟稳定性换取更低的内存占用
        let (tx, rx): (Sender<DecodedFrame>, Receiver<DecodedFrame>) =
            crossbeam_channel::bounded(2);

        let _sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            // 轻量级操作：仅将帧放入工作队列
            if let Err(_) = tx.try_send(frame.clone()) {
                //eprintln!("❌ 目标检测队列发送失败: {}", e);
            }
        });

        println!("✅ 检测模块已订阅DecodedFrame,等待视频流启动...");

        // 工作线程: 异步处理检测任务
        loop {
            // 检查配置更新
            if let Some(rx) = &self.config_rx {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        ControlMessage::UpdateParams {
                            conf_threshold,
                            iou_threshold,
                        } => {
                            if let Some(ref model) = detect_model {
                                let mut m = model.lock().unwrap();
                                m.set_conf(conf_threshold);
                                m.set_iou(iou_threshold);
                            }
                        }
                        ControlMessage::SwitchModel(model_path) => {
                            println!("🔄 正在切换模型: {}", model_path);
                            if let Some(new_model) = self.load_model(&model_path) {
                                detect_model = Some(new_model);
                                self.detect_model_path = model_path.clone();
                                model_loaded = true;

                                // 重新检查姿态估计支持
                                let m = detect_model.as_ref().unwrap().lock().unwrap();
                                if self.pose_enabled && !m.supports_task(YOLOTask::Pose) {
                                    println!("⚠️ 新模型不支持姿态估计,已自动禁用");
                                    self.pose_enabled = false;
                                }
                            }
                        }
                        ControlMessage::SwitchTracker(tracker_name) => {
                            println!("🔄 正在切换跟踪器: {}", tracker_name);
                            self.tracker = match tracker_name.to_lowercase().as_str() {
                                "deepsort" => TrackerType::DeepSort(PersonTracker::new()),
                                "bytetrack" => TrackerType::ByteTrack(ByteTracker::new()),
                                _ => TrackerType::None,
                            };
                        }
                        ControlMessage::TogglePose(enabled) => {
                            self.pose_enabled = enabled;
                            if enabled {
                                if let Some(ref model) = detect_model {
                                    let m = model.lock().unwrap();
                                    if !m.supports_task(YOLOTask::Pose) {
                                        println!("⚠️ 当前模型不支持姿态估计,无法启用");
                                        self.pose_enabled = false;
                                    } else {
                                        println!("✅ 姿态估计已启用");
                                    }
                                }
                            } else {
                                println!("🚫 姿态估计已禁用");
                            }
                        }
                        ControlMessage::ToggleDetection(enabled) => {
                            self.detection_enabled = enabled;
                            if enabled {
                                println!("✅ 目标检测已启用");
                            } else {
                                println!("🚫 目标检测已禁用");
                            }
                        }
                    }
                }
            }

            match rx.recv() {
                Ok(frame) => {
                    // 延迟加载: 收到第一帧时才加载模型
                    if !model_loaded {
                        println!("📥 收到第一帧数据,开始加载模型: {}", self.detect_model_path);
                        match self.load_model(&self.detect_model_path) {
                            Some(model) => {
                                // 检查姿态估计支持
                                {
                                    let m = model.lock().unwrap();
                                    if self.pose_enabled && !m.supports_task(YOLOTask::Pose) {
                                        println!("⚠️ 姿态估计: 已请求但模型不支持,将禁用");
                                        self.pose_enabled = false;
                                    } else if self.pose_enabled {
                                        println!("✅ 姿态估计: 已启用");
                                    }
                                }
                                detect_model = Some(model);
                                model_loaded = true;
                                println!("✅ 模型加载完成,开始处理视频流");
                            }
                            None => {
                                eprintln!("❌ 模型加载失败,跳过此帧");
                                continue;
                            }
                        }
                    }

                    if self.detection_enabled {
                        if let Some(ref model) = detect_model {
                            self.process_frame(frame, model, inf_size);
                        }
                    } else {
                        // 如果检测被禁用，仍然需要发送空结果以维持FPS统计和画面更新
                        // 或者直接跳过处理，取决于架构设计。
                        // 这里我们选择发送一个空的检测结果，以便渲染线程知道没有检测到物体
                        // 但为了节省资源，我们不进行任何图像处理
                        xbus::post(DetectionResult {
                            bboxes: Vec::new(),
                            keypoints: Vec::new(),
                            inference_fps: 0.0,
                            inference_ms: 0.0,
                            tracker_fps: 0.0,
                            tracker_ms: 0.0,
                            resized_image: None,
                            resized_size: inf_size,
                            reid_features: Vec::new(),
                        });
                    }
                }
                Err(e) => {
                    eprintln!("❌ 目标检测队列接收失败: {}", e);
                    break;
                }
            }

            // TODO: 监听SystemControl消息,支持优雅退出
        }
    }

    /// 处理单帧检测 (在工作线程中执行)
    fn process_frame(
        &mut self,
        frame: DecodedFrame,
        detect_model: &Arc<Mutex<Box<dyn Model>>>,
        inf_size: u32,
    ) {
        let start_total = Instant::now();

        // 2. Resize: 动态分辨率 → 640x640 (CPU并行优化)
        let t2 = Instant::now();

        let src_w = frame.width as usize;
        let src_h = frame.height as usize;
        let dst_size = inf_size as usize;
        let src_buffer = &frame.rgba_data;

        // 纯CPU优化 (避免GPU数据传输开销)
        let rgb_data = Self::cpu_resize_rgba_to_rgb(
            src_buffer,
            src_w,
            src_h,
            dst_size,
            &mut self.resize_x_map,
            &mut self.resize_y_map,
            &mut self.src_width,
            &mut self.src_height,
        );

        let resize_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 3. RGB → DynamicImage (零拷贝)
        let rgb_img = match RgbImage::from_raw(inf_size, inf_size, rgb_data) {
            Some(img) => img,
            None => {
                eprintln!("❌ RGB图像转换失败");
                return;
            }
        };
        let img = DynamicImage::ImageRgb8(rgb_img);

        // 5. YOLO检测 (统一处理所有模型类型)
        let t5_preprocess = Instant::now();

        // 方式1: 细粒度控制 - 分步调用以便计时
        // 方式2: 简化版 - model.forward(&images) (内部自动调用三步)
        let images = vec![img]; // 只创建一次Vec,避免重复clone
        let mut model = detect_model.lock().unwrap();
        let xs = model.preprocess(&images).unwrap_or_default();
        let preprocess_time = t5_preprocess.elapsed().as_secs_f64() * 1000.0;

        let t5_inference = Instant::now();
        let ys = model.run(xs, false).unwrap_or_default();
        let inference_time = t5_inference.elapsed().as_secs_f64() * 1000.0;

        let t5_postprocess = Instant::now();
        let detect_results = model.postprocess(ys, &images).unwrap_or_default();
        let postprocess_time = t5_postprocess.elapsed().as_secs_f64() * 1000.0;
        drop(model);

        let (_preprocess_ms, inference_ms, _postprocess_ms) =
            (preprocess_time, inference_time, postprocess_time);

        // 6. 提取检测框并缩放到原始分辨率
        let scale_x = frame.width as f32 / inf_size as f32;
        let scale_y = frame.height as f32 / inf_size as f32;

        let mut bboxes = Vec::new();
        let mut all_detections_count = 0; // 调试: 统计所有类别的检测数
        let mut person_detections_count = 0; // 调试: 统计人的检测数

        // COCO类别: 0=person, 39=bottle, 41=cup, 56=chair, 62=tv, 63=laptop, 73=book, 76=scissors
        const DETECT_CLASSES: &[usize] = &[0]; // 只检测人,如需检测其他类别可添加: &[0, 39, 41, 56, 62, 63, 73, 76]

        for result in &detect_results {
            if let Some(boxes) = result.bboxes() {
                all_detections_count += boxes.len();
                for bbox in boxes {
                    // 检测指定类别
                    if DETECT_CLASSES.contains(&bbox.id()) {
                        if bbox.id() == 0 {
                            person_detections_count += 1;
                        }
                        if bbox.confidence() >= 0.01 {
                            bboxes.push(types::BBox {
                                x1: bbox.xmin() * scale_x,
                                y1: bbox.ymin() * scale_y,
                                x2: bbox.xmax() * scale_x,
                                y2: bbox.ymax() * scale_y,
                                confidence: bbox.confidence(),
                                class_id: bbox.id() as u32,
                            });
                        } else if self.count % 30 == 0 && bbox.id() == 0 {
                            eprintln!("⚠️ 极低置信度人检测被过滤: conf={:.3}", bbox.confidence());
                        }
                    }
                }
            }
        }

        // 调试日志 - 统计各类别分布
        if self.count % 30 == 0 && all_detections_count > 0 {
            use std::collections::HashMap;
            let mut class_counts: HashMap<usize, usize> = HashMap::new();
            for result in &detect_results {
                if let Some(boxes) = result.bboxes() {
                    for bbox in boxes {
                        *class_counts.entry(bbox.id()).or_insert(0) += 1;
                    }
                }
            }
            let mut sorted: Vec<_> = class_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let top3: Vec<String> = sorted
                .iter()
                .take(3)
                .map(|(k, v)| format!("c{}:{}", k, v))
                .collect();
            eprintln!(
                "🔍 原始检测: 总{}个 (top3: {}) | 人{}个 | 通过阈值{}个",
                all_detections_count,
                top3.join(" "),
                person_detections_count,
                bboxes.len()
            );
        }

        // 7. 姿态估计
        let mut keypoints = Vec::new();
        if self.pose_enabled {
            for result in &detect_results {
                if let Some(kpts) = result.keypoints() {
                    for kpt in kpts {
                        // 转换关键点数据: Vec<Point2> -> Vec<(f32, f32, f32)>
                        let points: Vec<(f32, f32, f32)> =
                            kpt.iter().map(|p| (p.x(), p.y(), p.confidence())).collect();
                        keypoints.push(types::PoseKeypoints { points });
                    }
                }
            }
        }

        // 8. 跟踪器更新
        let tracker_start = Instant::now();
        let (tracked_bboxes, reid_features) = match &mut self.tracker {
            TrackerType::DeepSort(tracker) => {
                // 传入原始图像数据以启用ReID特征提取
                // 注意: 这里需要传入原始图像数据,我们直接使用Arc切片
                let frame_data = Some((frame.rgba_data.as_slice(), frame.width, frame.height));
                let tracked = tracker.update(&bboxes, &keypoints, frame_data);

                // 将跟踪结果转换为BBox格式(保持原有结构)
                let bboxes: Vec<types::BBox> = tracked
                    .iter()
                    .map(|t| types::BBox {
                        x1: t.bbox.x1,
                        y1: t.bbox.y1,
                        x2: t.bbox.x2,
                        y2: t.bbox.y2,
                        confidence: t.bbox.confidence,
                        class_id: t.id, // 使用跟踪ID替换class_id
                    })
                    .collect();

                // 获取ReID特征
                let reid_feats = tracker.get_reid_features();
                (bboxes, reid_feats)
            }
            TrackerType::ByteTrack(tracker) => {
                let tracked = tracker.update(&bboxes);
                let bboxes = tracked
                    .iter()
                    .map(|t| types::BBox {
                        x1: t.bbox.x1,
                        y1: t.bbox.y1,
                        x2: t.bbox.x2,
                        y2: t.bbox.y2,
                        confidence: t.bbox.confidence,
                        class_id: t.id,
                    })
                    .collect();
                (bboxes, Vec::new())
            }
            TrackerType::None => (bboxes.clone(), Vec::new()), // 不使用跟踪器,直接返回检测结果
        };
        let tracker_ms = tracker_start.elapsed().as_secs_f64() * 1000.0;

        // 更新跟踪器统计
        if !matches!(self.tracker, TrackerType::None) {
            self.tracker_count += 1;
            let now_tracker = Instant::now();
            if now_tracker.duration_since(self.tracker_last).as_secs() >= 1 {
                self.tracker_current_fps = self.tracker_count as f64
                    / now_tracker.duration_since(self.tracker_last).as_secs_f64();
                self.tracker_count = 0;
                self.tracker_last = now_tracker;
            }
        }

        // 使用跟踪后的结果替换原始检测框
        let bboxes = tracked_bboxes;

        // 9. 更新统计
        self.count += 1;
        let now = Instant::now();
        if now.duration_since(self.last).as_secs() >= 1 {
            self.current_fps = self.count as f64 / now.duration_since(self.last).as_secs_f64();
            self.count = 0;
            self.last = now;
        }

        // 计算总耗时 (移除未使用的tracker_ms变量)
        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        // 性能监控日志 (每60帧打印一次简洁信息)
        if self.count % 60 == 0 {
            if matches!(self.tracker, TrackerType::None) {
                eprintln!(
                    "🎯 检测: {}人 | {:.1}ms/帧 | {:.1}fps (Resize:{:.1}ms | 推理:{:.1}ms)",
                    bboxes.len(),
                    total_ms,
                    self.current_fps,
                    resize_ms,
                    inference_ms
                );
            } else {
                eprintln!(
                    "🎯 检测+跟踪: {}人 | {:.1}ms/帧 | {:.1}fps (Resize:{:.1}ms | 推理:{:.1}ms | 跟踪:{:.1}ms)",
                    bboxes.len(),
                    total_ms,
                    self.current_fps,
                    resize_ms,
                    inference_ms,
                    tracker_ms
                );
            }
        }

        // 10. 发送检测结果到XBus
        // 移除 resized_image 以节省内存 (每帧 640x640x4 = 1.6MB)
        xbus::post(DetectionResult {
            bboxes,
            keypoints,
            inference_fps: self.current_fps,
            inference_ms,
            tracker_fps: self.tracker_current_fps,
            tracker_ms,
            resized_image: None, // 不再传输预览图像,节省内存
            resized_size: inf_size,
            reid_features,
        });
    }
}
