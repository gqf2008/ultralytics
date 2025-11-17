/// 检测器 (Detector)
/// 职责: 订阅DecodedFrame → YOLO检测 → 追踪 → 发送DetectionResult消息
use crate::fastestv2::{FastestV2Config, FastestV2Postprocessor};
use crate::nanodet::{NanoDetConfig, NanoDetPostprocessor};
use crate::rtsp::DecodedFrame;
use crate::rtsp::{tracker::PersonTracker, types, TrackerType};
use crate::xbus;
use crate::{Args as YoloArgs, YOLOTask, YOLOv8};
use crossbeam_channel::{self, Receiver, Sender};
use fast_image_resize as fr;
use image::{DynamicImage, ImageBuffer, RgbImage, Rgba};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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

pub struct Detector {
    detect_model_path: String,
    pose_model_path: String,
    tracker_type: TrackerType,
    inf_size: u32,

    // 统计
    count: u64,
    last: Instant,
    current_fps: f64,

    // 追踪器性能统计
    tracker_count: u64,
    tracker_last: Instant,
    tracker_fps: f64,

    // 追踪器 (已禁用用于性能测试)
    #[allow(dead_code)]
    tracker: Option<PersonTracker>,
}

impl Detector {
    pub fn new(
        detect_model: String,
        pose_model: String,
        tracker_type: TrackerType,
        inf_size: u32,
    ) -> Self {
        Self {
            detect_model_path: detect_model,
            pose_model_path: pose_model,
            tracker_type,
            inf_size,
            count: 0,
            last: Instant::now(),
            current_fps: 0.0,
            tracker_count: 0,
            tracker_last: Instant::now(),
            tracker_fps: 0.0,
            tracker: None, // 已禁用用于性能测试
        }
    }

    pub fn run(&mut self) {
        println!("🔍 检测模块启动");

        let is_fastestv2 = self.detect_model_path.contains("fastestv2");
        let is_nanodet = self.detect_model_path.contains("nanodet");

        // 加载检测模型
        let detect_args = YoloArgs {
            model: self.detect_model_path.clone(),
            width: Some(self.inf_size),
            height: Some(self.inf_size),
            conf: if is_fastestv2 {
                0.10
            } else if is_nanodet {
                0.35 // NanoDet推荐0.35
            } else {
                0.15
            },
            iou: if is_nanodet { 0.6 } else { 0.45 },
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

        let detect_model = match YOLOv8::new(detect_args) {
            Ok(m) => {
                println!("✅ 检测模型加载成功");
                Arc::new(Mutex::new(m))
            }
            Err(e) => {
                eprintln!("❌ 检测模型加载失败: {}", e);
                return;
            }
        };

        // FastestV2专用后处理
        let fastestv2_postprocessor = if is_fastestv2 {
            let config = FastestV2Config {
                conf_threshold: 0.05, // FastestV2输出置信度较低,使用0.05阈值
                iou_threshold: 0.45,
                num_classes: 80,          // COCO 类别数
                num_anchors: 3,           // 每个尺度3个anchor
                strides: vec![8, 16, 32], // YOLOv8默认stride
                anchors: vec![
                    12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 79.57, 257.11, 140.63, 149.70,
                    279.92, 258.87,
                ],
            };
            Some(FastestV2Postprocessor::new(
                config,
                self.inf_size as usize,
                self.inf_size as usize,
            ))
        } else {
            None
        };

        // NanoDet专用后处理
        let nanodet_postprocessor = if is_nanodet {
            let config = NanoDetConfig {
                num_classes: 80,
                strides: vec![8, 16, 32], // NanoDet-Plus三层特征
                conf_threshold: 0.35,     // NanoDet推荐0.35
                iou_threshold: 0.6,
                reg_max: 7, // DFL参数
            };
            Some(NanoDetPostprocessor::new(
                config,
                self.inf_size as usize,
                self.inf_size as usize,
            ))
        } else {
            None
        };

        // 加载姿态模型 (可选)
        let pose_model = if !self.pose_model_path.is_empty() {
            let pose_args = YoloArgs {
                model: self.pose_model_path.clone(),
                width: Some(self.inf_size),
                height: Some(self.inf_size),
                conf: 0.5,
                iou: 0.45,
                kconf: 0.55,
                source: String::new(),
                device_id: 0,
                trt: false,
                cuda: false,
                batch: 1,
                batch_min: 1,
                batch_max: 1,
                fp16: false,
                task: Some(YOLOTask::Pose),
                nc: None,
                nk: Some(17),
                nm: None,
                profile: false,
            };

            match YOLOv8::new(pose_args) {
                Ok(m) => {
                    println!("✅ 姿态模型加载成功");
                    Some(Arc::new(Mutex::new(m)))
                }
                Err(e) => {
                    println!("⚠️  姿态模型加载失败: {}", e);
                    None
                }
            }
        } else {
            None
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

        // 初始化追踪器
        self.tracker = Some(PersonTracker::new());
        println!(
            "✅ 追踪器初始化成功 ({})",
            match self.tracker_type {
                TrackerType::DeepSort => "DeepSort",
                TrackerType::ByteTrack => "ByteTrack",
            }
        );

        // 工作线程:异步处理检测任务
        loop {
            match rx.recv() {
                Ok(frame) => {
                    self.process_frame(
                        frame,
                        &detect_model,
                        &pose_model,
                        &fastestv2_postprocessor,
                        &nanodet_postprocessor,
                        inf_size,
                    );
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
        detect_model: &Arc<Mutex<YOLOv8>>,
        _pose_model: &Option<Arc<Mutex<YOLOv8>>>,
        fastestv2_postprocessor: &Option<FastestV2Postprocessor>,
        nanodet_postprocessor: &Option<NanoDetPostprocessor>,
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

        // 5. YOLO检测 (只保留inference_ms用于日志)
        let t5_preprocess = Instant::now();
        let (detect_results, _preprocess_ms, inference_ms, _postprocess_ms) =
            if let Some(ref pp) = fastestv2_postprocessor {
                // FastestV2专用后处理
                let mut model = detect_model.lock().unwrap();
                let xs = model.preprocess(&vec![img.clone()]).unwrap_or_default();
                let preprocess_time = t5_preprocess.elapsed().as_secs_f64() * 1000.0;

                let t5_inference = Instant::now();
                let ys = model.engine_mut().run(xs, false).unwrap_or_default();
                let inference_time = t5_inference.elapsed().as_secs_f64() * 1000.0;
                drop(model);

                let t5_postprocess = Instant::now();
                let results = pp.postprocess(ys, &vec![img.clone()]).unwrap_or_default();
                let postprocess_time = t5_postprocess.elapsed().as_secs_f64() * 1000.0;

                (results, preprocess_time, inference_time, postprocess_time)
            } else if let Some(ref pp) = nanodet_postprocessor {
                // NanoDet专用后处理
                let mut model = detect_model.lock().unwrap();
                let xs = model.preprocess(&vec![img.clone()]).unwrap_or_default();
                let preprocess_time = t5_preprocess.elapsed().as_secs_f64() * 1000.0;

                let t5_inference = Instant::now();
                let ys = model.engine_mut().run(xs, false).unwrap_or_default();
                let inference_time = t5_inference.elapsed().as_secs_f64() * 1000.0;
                drop(model);

                let t5_postprocess = Instant::now();
                let results = pp.postprocess(ys, &vec![img.clone()]).unwrap_or_default();
                let postprocess_time = t5_postprocess.elapsed().as_secs_f64() * 1000.0;

                (results, preprocess_time, inference_time, postprocess_time)
            } else {
                let mut model = detect_model.lock().unwrap();
                let t5_run = Instant::now();
                let results = model.run(&vec![img.clone()]).unwrap_or_default();
                let run_time = t5_run.elapsed().as_secs_f64() * 1000.0;
                drop(model);
                (results, 0.0, run_time, 0.0)
            };

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

        // 7. 姿态估计 (可选,性能优先 - 跳帧策略) - 已禁用用于性能测试
        let keypoints = Vec::new();
        // if let Some(pose_model) = pose_model { ... } // 已禁用

        // 8. 追踪器更新 (使用检测结果和姿态关键点) - 已禁用用于性能测试
        let tracker_start = Instant::now();
        // if let Some(ref mut tracker) = self.tracker { ... } // 已禁用
        let tracker_ms = tracker_start.elapsed().as_secs_f64() * 1000.0;

        // 9. 更新统计
        self.count += 1;
        self.tracker_count += 1;
        let now = Instant::now();
        if now.duration_since(self.last).as_secs() >= 1 {
            self.current_fps = self.count as f64 / now.duration_since(self.last).as_secs_f64();
            self.count = 0;
            self.last = now;
        }
        if now.duration_since(self.tracker_last).as_secs() >= 1 {
            self.tracker_fps =
                self.tracker_count as f64 / now.duration_since(self.tracker_last).as_secs_f64();
            self.tracker_count = 0;
            self.tracker_last = now;
        }

        // 计算总耗时 (移除未使用的tracker_ms变量)
        let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        // 性能监控日志 (每60帧打印一次简洁信息)
        if self.count % 60 == 0 {
            eprintln!(
                "🎯 检测: {}人 | {:.1}ms/帧 | {:.1}fps (Resize:{:.1}ms | 推理:{:.1}ms)",
                bboxes.len(),
                total_ms,
                self.current_fps,
                resize_ms,
                inference_ms
            );
        }

        // 10. 发送检测结果到XBus
        xbus::post(DetectionResult {
            bboxes,
            keypoints,
            inference_fps: self.current_fps,
            inference_ms: total_ms,
            tracker_fps: self.tracker_fps,
            tracker_ms,
            resized_image: Some(resized_rgba),
            resized_size: inf_size,
        });
    }
}
