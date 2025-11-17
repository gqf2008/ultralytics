use crate::fastestv2::{FastestV2Config, FastestV2Postprocessor};
use crate::pipeline::{DecodedFrame, DetectionResult};
/// 检测器 (Detector)
/// 职责: 订阅DecodedFrame → YOLO检测 → 发送DetectionResult消息
use crate::rtsp::{self, TrackerType};
use crate::xbus::{self, Subscription};
use crate::{Args as YoloArgs, YOLOTask, YOLOv8};
use image::{imageops, DynamicImage, ImageBuffer, Rgb, RgbImage, Rgba};
use std::time::Instant;

pub struct Detector {
    detect_model_path: String,
    pose_model_path: String,
    tracker_type: TrackerType,
    inf_size: u32,

    // 统计
    count: u64,
    last: Instant,
    current_fps: f64,
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
            conf: if is_fastestv2 { 0.20 } else { 0.25 },
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

        let mut detect_model = match YOLOv8::new(detect_args) {
            Ok(m) => {
                println!("✅ 检测模型加载成功");
                m
            }
            Err(e) => {
                eprintln!("❌ 检测模型加载失败: {}", e);
                return;
            }
        };

        // FastestV2专用后处理
        let fastestv2_postprocessor = if is_fastestv2 {
            let config = FastestV2Config {
                conf_threshold: 0.20,
                iou_threshold: 0.45,
                anchors: vec![
                    12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 79.57, 257.11, 140.63, 149.70,
                    279.92, 258.87,
                ],
            };
            Some(FastestV2Postprocessor::new(config))
        } else {
            None
        };

        // 加载姿态模型 (可选)
        let mut pose_model = if !self.pose_model_path.is_empty() {
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
                    Some(m)
                }
                Err(e) => {
                    println!("⚠️  姿态模型加载失败: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // 订阅解码帧
        let inf_size = self.inf_size;
        let _sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            let start = Instant::now();

            // 1. RGBA → RgbaImage
            let rgba_img = match ImageBuffer::<Rgba<u8>, _>::from_raw(
                frame.width,
                frame.height,
                frame.rgba_data.clone(),
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
                let xs = detect_model
                    .preprocess(&vec![img.clone()])
                    .unwrap_or_default();
                let ys = detect_model.engine_mut().run(xs, false).unwrap_or_default();
                pp.postprocess(ys, &vec![img.clone()]).unwrap_or_default()
            } else {
                detect_model.run(&vec![img.clone()]).unwrap_or_default()
            };

            // 6. 提取检测框
            let mut bboxes = Vec::new();
            for result in &detect_results {
                if let Some(boxes) = result.bboxes() {
                    for bbox in boxes {
                        // 只检测人 (class=0)
                        if bbox.id() == 0 && bbox.confidence() >= 0.20 {
                            bboxes.push(rtsp::BBox {
                                x1: bbox.xmin(),
                                y1: bbox.ymin(),
                                x2: bbox.xmax(),
                                y2: bbox.ymax(),
                                confidence: bbox.confidence(),
                                class_id: bbox.id(),
                            });
                        }
                    }
                }
            }

            // 7. 姿态估计 (TODO: 实现)
            let keypoints = Vec::new();

            // 8. 计算FPS
            let inference_ms = start.elapsed().as_secs_f64() * 1000.0;

            // 9. 发送检测结果
            xbus::post(DetectionResult {
                frame_id: frame.frame_id,
                bboxes,
                keypoints,
                inference_fps: 0.0, // 在这里统计
                inference_ms,
            });
        });

        println!("✅ 检测模块已订阅DecodedFrame,等待数据...");

        // 保持线程运行
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));

            // TODO: 监听SystemControl消息,支持优雅退出
        }
    }
}
