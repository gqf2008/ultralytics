//! 检测器 (Detector)
//! 职责: 订阅DecodedFrame → YOLO检测 → 发送DetectionResult消息

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use fast_image_resize as fr;
use image::{DynamicImage, ImageBuffer, RgbImage, Rgba};

use super::types::DecodedFrame;
use super::{ByteTracker, PersonTracker};
use crate::detection::types;
use crate::models::{FastestV2, Model, ModelType, NanoDet, YOLOv8, YOLOX};
use crate::{xbus, Args, YOLOTask};

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
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            tracker_count: 0,
            tracker_last: Instant::now(),
            tracker_current_fps: 0.0,
        }
    }

    pub fn run(&mut self) {
        println!("🔍 检测模块启动");

        // 识别模型类型
        let model_type = ModelType::from_path(&self.detect_model_path);

        // 加载检测模型
        let detect_args = Args {
            model: self.detect_model_path.clone(),
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

        // 根据模型类型创建对应的模型实例
        let detect_model: Arc<Mutex<Box<dyn Model>>> = match model_type {
            ModelType::YOLOv8 | ModelType::YOLOv5 => match YOLOv8::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLOv8 检测模型加载成功");
                    // 检查姿态估计能力
                    if self.pose_enabled {
                        if m.supports_task(YOLOTask::Pose) {
                            println!("✅ 姿态估计: 已启用 (模型支持)");
                        } else {
                            println!("⚠️ 姿态估计: 已请求但模型不支持,将禁用");
                            self.pose_enabled = false;
                        }
                    }
                    Arc::new(Mutex::new(Box::new(m)))
                }
                Err(e) => {
                    eprintln!("❌ YOLOv8 模型加载失败: {}", e);
                    return;
                }
            },
            ModelType::FastestV2 => match FastestV2::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLO-FastestV2 检测模型加载成功");
                    Arc::new(Mutex::new(Box::new(m)))
                }
                Err(e) => {
                    eprintln!("❌ FastestV2 模型加载失败: {}", e);
                    return;
                }
            },
            ModelType::NanoDet => match NanoDet::new(detect_args) {
                Ok(m) => {
                    println!("✅ NanoDet 检测模型加载成功");
                    Arc::new(Mutex::new(Box::new(m)))
                }
                Err(e) => {
                    eprintln!("❌ NanoDet 模型加载失败: {}", e);
                    return;
                }
            },
            ModelType::YOLOX => match YOLOX::new(detect_args) {
                Ok(m) => {
                    println!("✅ YOLOX 检测模型加载成功");
                    Arc::new(Mutex::new(Box::new(m)))
                }
                Err(e) => {
                    eprintln!("❌ YOLOX 模型加载失败: {}", e);
                    return;
                }
            },
        };

        // 订阅解码帧 - 仅将任务放入队列
        let inf_size = self.inf_size;
        let (tx, rx): (Sender<DecodedFrame>, Receiver<DecodedFrame>) =
            crossbeam_channel::bounded(120);

        let _sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            // 轻量级操作：仅将帧放入工作队列
            if let Err(_) = tx.try_send(frame.clone()) {
                //eprintln!("❌ 目标检测队列发送失败: {}", e);
            }
        });

        println!("✅ 检测模块已订阅DecodedFrame,等待数据...");

        // 工作线程: 异步处理检测任务
        loop {
            match rx.recv() {
                Ok(frame) => {
                    self.process_frame(frame, &detect_model, inf_size);
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

        // 1. RGBA → RgbaImage
        let rgba_img = match ImageBuffer::<Rgba<u8>, _>::from_raw(
            frame.width,
            frame.height,
            frame.rgba_data,
        ) {
            Some(img) => img,
            None => {
                eprintln!("❌ RGBA图像转换失败");
                return;
            }
        };

        // 2. Resize: 动态分辨率 → 320x320 (使用 fast_image_resize 高性能库 + Nearest 插值)
        let t2 = Instant::now();

        // 创建源图像 (RGBA)
        let src_buffer = rgba_img.as_raw().clone();
        let src_image = fr::images::Image::from_vec_u8(
            frame.width,
            frame.height,
            src_buffer,
            fr::PixelType::U8x4,
        )
        .unwrap();

        // 创建目标图像 (RGBA)
        let mut dst_image = fr::images::Image::new(inf_size, inf_size, fr::PixelType::U8x4);

        // 执行超快速缩放 (Nearest 算法,比 Bilinear 快 5-10 倍,牺牲少量质量换取极致速度)
        let mut resizer = fr::Resizer::new();
        resizer
            .resize(
                &src_image,
                &mut dst_image,
                &fr::ResizeOptions::new().resize_alg(fr::ResizeAlg::Nearest), // 最快插值算法
            )
            .unwrap();

        let resize_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 3. RGBA → RGB (优化版: 预分配 + 直接循环)
        let dst_pixels = dst_image.buffer();
        let mut rgb_data = Vec::with_capacity((inf_size * inf_size * 3) as usize);
        for chunk in dst_pixels.chunks_exact(4) {
            rgb_data.push(chunk[0]); // R
            rgb_data.push(chunk[1]); // G
            rgb_data.push(chunk[2]); // B
                                     // 跳过 Alpha 通道
        }

        // 保存一份用于右下角显示 (转换为RGBA格式,ggez需要)
        let mut resized_rgba = Vec::with_capacity((inf_size * inf_size * 4) as usize);
        for chunk in dst_pixels.chunks_exact(4) {
            resized_rgba.push(chunk[0]); // R
            resized_rgba.push(chunk[1]); // G
            resized_rgba.push(chunk[2]); // B
            resized_rgba.push(255); // A (不透明)
        }

        // 4. RGB → DynamicImage
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
                        if bbox.confidence() >= 0.05 {
                            bboxes.push(types::BBox {
                                x1: bbox.xmin() * scale_x,
                                y1: bbox.ymin() * scale_y,
                                x2: bbox.xmax() * scale_x,
                                y2: bbox.ymax() * scale_y,
                                confidence: bbox.confidence(),
                                class_id: bbox.id() as u32,
                            });
                        } else if self.count % 30 == 0 && bbox.id() == 0 {
                            eprintln!("⚠️ 低置信度人检测: conf={:.3}", bbox.confidence());
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
        let tracked_bboxes = match &mut self.tracker {
            TrackerType::DeepSort(tracker) => {
                let tracked = tracker.update(&bboxes, &keypoints, None);
                // 将跟踪结果转换为BBox格式(保持原有结构)
                tracked
                    .iter()
                    .map(|t| types::BBox {
                        x1: t.bbox.x1,
                        y1: t.bbox.y1,
                        x2: t.bbox.x2,
                        y2: t.bbox.y2,
                        confidence: t.bbox.confidence,
                        class_id: t.id, // 使用跟踪ID替换class_id
                    })
                    .collect()
            }
            TrackerType::ByteTrack(tracker) => {
                let tracked = tracker.update(&bboxes);
                tracked
                    .iter()
                    .map(|t| types::BBox {
                        x1: t.bbox.x1,
                        y1: t.bbox.y1,
                        x2: t.bbox.x2,
                        y2: t.bbox.y2,
                        confidence: t.bbox.confidence,
                        class_id: t.id,
                    })
                    .collect()
            }
            TrackerType::None => bboxes.clone(), // 不使用跟踪器,直接返回检测结果
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
        xbus::post(DetectionResult {
            bboxes,
            keypoints,
            inference_fps: self.current_fps,
            inference_ms: total_ms,
            tracker_fps: self.tracker_current_fps,
            tracker_ms,
            resized_image: Some(resized_rgba),
            resized_size: inf_size,
        });
    }
}
