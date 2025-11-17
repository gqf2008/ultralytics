/// 检测器 (Detector)
/// 职责: 订阅DecodedFrame → YOLO检测 → 追踪 → 发送DetectionResult消息
use crate::fastestv2::{FastestV2Config, FastestV2Postprocessor};
use crate::rtsp::DecodedFrame;
use crate::rtsp::{tracker::PersonTracker, types, TrackerType};
use crate::xbus;
use crate::{Args as YoloArgs, YOLOTask, YOLOv8};
use crossbeam_channel::{self, Receiver, Sender};
use image::{imageops, DynamicImage, ImageBuffer, RgbImage, Rgba};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// 检测结果 (检测模块 → 渲染模块)
#[derive(Clone, Debug)]
pub struct DetectionResult {
    pub bboxes: Vec<types::BBox>,
    pub keypoints: Vec<types::PoseKeypoints>,
    pub inference_fps: f64,
    pub inference_ms: f64,
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

    // 姿态估计优化: 跳帧策略
    pose_skip_counter: u32,
    pose_skip_interval: u32, // 每N帧做一次姿态估计

    // 追踪器
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
            pose_skip_counter: 0,
            pose_skip_interval: 5, // 每5帧做一次姿态估计,提升性能
            tracker: None,         // 延迟初始化,在run中创建
        }
    }

    pub fn run(&mut self) {
        println!("🔍 检测模块启动");

        let is_fastestv2 = self.detect_model_path.contains("fastestv2");

        // 加载检测模型
        let detect_args = YoloArgs {
            model: self.detect_model_path.clone(),
            width: Some(self.inf_size),
            height: Some(self.inf_size),
            conf: if is_fastestv2 { 0.10 } else { 0.15 },
            iou: 0.45,
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
        pose_model: &Option<Arc<Mutex<YOLOv8>>>,
        fastestv2_postprocessor: &Option<FastestV2Postprocessor>,
        inf_size: u32,
    ) {
        let start = Instant::now();

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
        // 2. Resize: 动态分辨率 → 320x320
        let resized_rgba = imageops::resize(
            &rgba_img,
            inf_size,
            inf_size,
            imageops::FilterType::Triangle,
        );

        // 3. RGBA → RGB
        let rgb_data: Vec<u8> = resized_rgba
            .pixels()
            .flat_map(|p| vec![p.0[0], p.0[1], p.0[2]])
            .collect();

        // 4. RGB → DynamicImage
        let rgb_img = match RgbImage::from_raw(inf_size, inf_size, rgb_data) {
            Some(img) => img,
            None => {
                eprintln!("❌ RGB图像转换失败");
                return;
            }
        };
        let img = DynamicImage::ImageRgb8(rgb_img);

        // 5. YOLO检测
        let detect_results = if let Some(ref pp) = fastestv2_postprocessor {
            // FastestV2专用后处理
            let mut model = detect_model.lock().unwrap();
            let xs = model.preprocess(&vec![img.clone()]).unwrap_or_default();
            let ys = model.engine_mut().run(xs, false).unwrap_or_default();
            drop(model);
            pp.postprocess(ys, &vec![img.clone()]).unwrap_or_default()
        } else {
            let mut model = detect_model.lock().unwrap();
            model.run(&vec![img.clone()]).unwrap_or_default()
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

        // 7. 姿态估计 (可选,性能优先 - 跳帧策略)
        let mut keypoints = Vec::new();
        if let Some(pose_model) = pose_model {
            // 跳帧优化: 每N帧才做一次姿态估计
            self.pose_skip_counter += 1;
            let should_run_pose = self.pose_skip_counter >= self.pose_skip_interval;
            if should_run_pose {
                self.pose_skip_counter = 0;
            }

            // 限制姿态估计数量,避免性能下降 (只对第1个人做姿态估计)
            let max_pose_detections = 1;
            let bboxes_for_pose: Vec<_> = bboxes.iter().take(max_pose_detections).collect();

            if should_run_pose && !bboxes_for_pose.is_empty() {
                if let Ok(mut model) = pose_model.lock() {
                    // 对每个人体边界框运行姿态估计 (bbox已缩放到原始分辨率)
                    for bbox in bboxes_for_pose {
                        // 裁剪边界框区域 (带padding)
                        let padding = 20.0;
                        let x1 = (bbox.x1 - padding).max(0.0) as u32;
                        let y1 = (bbox.y1 - padding).max(0.0) as u32;
                        let x2 = (bbox.x2 + padding).min(frame.width as f32) as u32;
                        let y2 = (bbox.y2 + padding).min(frame.height as f32) as u32;

                        let crop_w = x2.saturating_sub(x1);
                        let crop_h = y2.saturating_sub(y1);

                        // 验证裁剪区域有效性
                        if crop_w < 10 || crop_h < 10 {
                            continue;
                        }

                        // 创建裁剪区域的子图像
                        let cropped_img =
                            imageops::crop_imm(&rgba_img, x1, y1, crop_w, crop_h).to_image();
                        let dynamic_img = DynamicImage::ImageRgba8(cropped_img);

                        // 运行姿态估计
                        if let Ok(pose_results) = model.run(&vec![dynamic_img]) {
                            for result in &pose_results {
                                if let Some(kpts_batch) = result.keypoints() {
                                    for kpts_person in kpts_batch {
                                        let mut points = Vec::new();
                                        for kp in kpts_person.iter() {
                                            // 转换坐标到原图
                                            points.push((
                                                kp.x() + x1 as f32,
                                                kp.y() + y1 as f32,
                                                kp.confidence(),
                                            ));
                                        }
                                        if !points.is_empty() {
                                            keypoints.push(types::PoseKeypoints { points });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 8. 追踪器更新 (使用检测结果和姿态关键点)
        if let Some(ref mut tracker) = self.tracker {
            // 准备RGBA帧数据用于特征提取
            let frame_rgba = Some((rgba_img.as_raw().as_slice(), frame.width, frame.height));

            // 更新追踪器,获取带ID的追踪对象
            let tracked_persons = tracker.update(&bboxes, &keypoints, frame_rgba);

            // 用追踪结果替换原始检测框 (现在bbox有稳定ID了)
            let original_keypoints = keypoints.clone(); // 保留原始关键点
            bboxes.clear();
            keypoints.clear();

            for (idx, person) in tracked_persons.iter().enumerate() {
                // 显示所有轨迹 (包括未确认的,调试用)
                // TODO: 恢复为 if person.confirmed 只显示稳定轨迹
                if true || person.confirmed {
                    bboxes.push(types::BBox {
                        x1: person.bbox.x1,
                        y1: person.bbox.y1,
                        x2: person.bbox.x2,
                        y2: person.bbox.y2,
                        confidence: person.bbox.confidence,
                        class_id: person.id, // 使用追踪ID
                    });

                    // 如果有对应的姿态关键点,也添加进去
                    if idx < original_keypoints.len() {
                        keypoints.push(original_keypoints[idx].clone());
                    }
                }
            }
        }

        // 9. 更新统计
        self.count += 1;
        let now = Instant::now();
        if now.duration_since(self.last).as_secs() >= 1 {
            self.current_fps = self.count as f64 / now.duration_since(self.last).as_secs_f64();
            self.count = 0;
            self.last = now;
        }

        let inference_ms = start.elapsed().as_secs_f64() * 1000.0;

        // 调试日志 (每秒打印一次)
        if self.count % 30 == 0 {
            eprintln!(
                "🎯 检测: {}人 | {}关键点组 | {:.1}ms | {:.1}fps",
                bboxes.len(),
                keypoints.len(),
                inference_ms,
                self.current_fps
            );
        }

        // 9. 发送检测结果到XBus
        xbus::post(DetectionResult {
            bboxes,
            keypoints,
            inference_fps: self.current_fps,
            inference_ms,
        });
    }
}
