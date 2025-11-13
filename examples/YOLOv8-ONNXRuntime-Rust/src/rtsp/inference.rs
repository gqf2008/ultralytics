/// YOLOv8推理线程模块
/// YOLOv8 inference thread module
use super::types::{BBox, DecodedFrame, PoseKeypoints, RenderData};
use crate::fastestv2::{FastestV2Config, FastestV2Postprocessor}; // 导入FastestV2后处理
use crate::{Args as YoloArgs, YOLOv8};
use crossbeam_channel::{Receiver, Sender};
use image::{imageops, DynamicImage, ImageBuffer, RgbImage, Rgba};
use std::time::Instant;

/// 推理线程: 接收原始帧 → Resize+检测+姿态 → 返回渲染数据
pub fn inference_thread(
    rx_decoded: Receiver<DecodedFrame>,
    tx_render: Sender<RenderData>,
    detect_model: String,
    pose_model: String,
    inf_size: u32,
) {
    println!("✅ Inference thread started");

    // 判断是否为FastestV2模型
    let is_fastestv2 = detect_model.contains("fastestv2");

    // Detection model
    let detect_args = YoloArgs {
        model: detect_model,
        width: Some(inf_size),
        height: Some(inf_size),
        conf: if is_fastestv2 { 0.12 } else { 0.25 }, // FastestV2使用更低阈值
        iou: 0.45,
        source: String::new(),
        device_id: 0,
        trt: false,
        cuda: false,
        batch: 1,
        batch_min: 1,
        batch_max: 1,
        fp16: false,
        task: Some(crate::YOLOTask::Detect), // 明确指定为检测任务
        nc: None,
        nk: None,
        nm: None,
        kconf: 0.55,
        profile: false,
    };

    // Pose model
    let pose_args = YoloArgs {
        model: pose_model,
        width: Some(inf_size),
        height: Some(inf_size),
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
        task: None,
        nc: None,
        nk: None,
        nm: None,
        profile: false,
    };

    let mut detect_model = match YOLOv8::new(detect_args) {
        Ok(m) => {
            println!("✅ 检测模型加载成功");
            m
        }
        Err(e) => {
            eprintln!("❌ 检测模型加载失败: {:?}", e);
            return;
        }
    };

    // 为FastestV2创建专用后处理器
    let fastestv2_postprocessor = if is_fastestv2 {
        let mut config = FastestV2Config::default();
        config.conf_threshold = 0.20; // 提高阈值减少误检
        Some(FastestV2Postprocessor::new(
            config,
            inf_size as usize,
            inf_size as usize,
        ))
    } else {
        None
    };

    // FastestV2不支持姿态估计,姿态模型为可选
    let mut pose_model = match YOLOv8::new(pose_args) {
        Ok(m) => {
            println!("✅ 姿态模型加载成功");
            Some(m)
        }
        Err(e) => {
            eprintln!("❌ 姿态模型加载失败: {:?}", e);
            if is_fastestv2 {
                println!("⚠️  FastestV2不支持姿态估计,继续仅使用检测功能");
                None
            } else {
                eprintln!("❌ 非FastestV2模型必须有姿态模型,退出推理线程");
                return;
            }
        }
    };

    let mut count = 0;
    let mut last = Instant::now();
    let mut current_fps = 0.0;
    let mut receive_count = 0;

    println!("🔍 推理线程等待数据...");

    while let Ok(decoded_frame) = rx_decoded.recv() {
        receive_count += 1;
        if receive_count == 1 {
            println!("✅ 推理线程收到第一帧数据!");
            println!(
                "   原始尺寸: {}x{}, RGBA数据: {} 字节",
                decoded_frame.width, decoded_frame.height,
                decoded_frame.rgba_data.len()
            );
        }

        count += 1;

        // 1. RGBA → RgbaImage
        let rgba_img = match ImageBuffer::<Rgba<u8>, _>::from_raw(
            decoded_frame.width,
            decoded_frame.height,
            decoded_frame.rgba_data.clone(),
        ) {
            Some(img) => img,
            None => {
                eprintln!("❌ RGBA图像转换失败!");
                continue;
            }
        };

        // 2. CPU Resize: 动态分辨率 → 320x320 (Triangle快速算法)
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
                eprintln!("❌ RGB图像转换失败!");
                continue;
            }
        };
        let img = DynamicImage::ImageRgb8(rgb_img);


        // 首帧时检查图像数据
        if receive_count == 1 {
            let pixel_sum: u32 = img
                .to_rgb8()
                .pixels()
                .take(100)
                .map(|p| p.0[0] as u32)
                .sum();
            println!("   图像采样检查: 前100像素R通道和={} (全黑为0)", pixel_sum);
        }

        let start = Instant::now();

        // Run detection (FastestV2使用专用后处理)
        let detect_results = if let Some(ref fastestv2_pp) = fastestv2_postprocessor {
            // FastestV2: 手动执行预处理 → 推理 → 专用后处理
            let xs = detect_model
                .preprocess(&vec![img.clone()])
                .unwrap_or_default();
            let ys = detect_model.engine_mut().run(xs, false).unwrap_or_default();
            fastestv2_pp
                .postprocess(ys, &vec![img.clone()])
                .unwrap_or_default()
        } else {
            // 标准YOLO使用统一后处理
            detect_model.run(&vec![img.clone()]).unwrap_or_default()
        };

        // 首次推理后打印结果信息
        if receive_count == 1 {
            println!("   检测模型返回: {} 个结果对象", detect_results.len());
            if !detect_results.is_empty() {
                println!("   第一个结果: {:?}", detect_results[0]);
            }
        }

        // Extract bounding boxes (all classes, not just person)
        let mut bboxes = Vec::new();
        for result in &detect_results {
            if let Some(boxes) = result.bboxes() {
                // 首次推理打印原始检测信息
                if receive_count == 1 {
                    println!("   原始检测框数量: {}", boxes.len());
                    for (i, bbox) in boxes.iter().take(3).enumerate() {
                        println!(
                            "     原始框{}: class={}, conf={:.3}",
                            i + 1,
                            bbox.id(),
                            bbox.confidence()
                        );
                    }
                }

                for bbox in boxes {
                    // 🎯 只检测人(class_id=0),提高阈值过滤误检
                    if bbox.id() == 0 && bbox.confidence() >= 0.20 {
                        bboxes.push(BBox {
                            x1: bbox.xmin(),
                            y1: bbox.ymin(),
                            x2: bbox.xmax(),
                            y2: bbox.ymax(),
                            confidence: bbox.confidence(),
                            class_id: bbox.id() as u32,
                        });
                    }
                }
            } else if receive_count == 1 {
                println!("   ⚠️  result.bboxes() 返回 None");
            }
        }

        // 🚀 FastestV2: 每帧都打印检测结果(实时反馈)
        if bboxes.len() > 0 {
            println!("🎯 [帧{}] 检测到 {} 人", count, bboxes.len());

            // 打印所有人的信息
            for (i, bbox) in bboxes.iter().enumerate() {
                println!(
                    "   👤 人{}: conf={:.3}, 位置=({:.0}, {:.0}) 大小=({:.0}x{:.0})",
                    i + 1,
                    bbox.confidence,
                    bbox.x1,
                    bbox.y1,
                    bbox.x2 - bbox.x1,
                    bbox.y2 - bbox.y1
                );
            }
        } else if count % 30 == 0 {
            // 无人时每30帧提示一次,避免刷屏
            println!("⚠️  [帧{}] 当前画面无人", count);
        }

        // 姿态估计(仅当姿态模型可用时)
        let pose_results = if let Some(ref mut pose_mdl) = pose_model {
            pose_mdl.run(&vec![img]).unwrap_or_default()
        } else {
            Vec::new()
        };

        // Extract keypoints
        let mut keypoints = Vec::new();
        for result in &pose_results {
            if let Some(kpts) = result.keypoints() {
                for kp in kpts {
                    let points: Vec<(f32, f32, f32)> =
                        kp.iter().map(|k| (k.x(), k.y(), k.confidence())).collect();
                    keypoints.push(PoseKeypoints { points });
                }
            }
        }

        let inference_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Calculate FPS (基于实际处理帧数)
        if last.elapsed().as_secs_f64() >= 1.0 {
            let elapsed = last.elapsed().as_secs_f64();
            current_fps = count as f64 / elapsed;

            // 每秒打印一次统计
            let theoretical_fps = 1000.0 / inference_ms;
            println!(
                "📊 推理统计: 处理{}帧 | 实际{:.1}fps | 理论{:.0}fps | 每帧{:.1}ms",
                count, current_fps, theoretical_fps, inference_ms
            );

            last = Instant::now();
            count = 0;
        }

        // 构造渲染数据: 原始帧 + 检测结果
        let render_data = RenderData {
            rgba_data: decoded_frame.rgba_data,
            width: decoded_frame.width,
            height: decoded_frame.height,
            decode_fps: decoded_frame.decode_fps,
            decoder_name: decoded_frame.decoder_name,
            bboxes,
            keypoints,
            inference_fps: current_fps,
            inference_ms,
        };

        let _ = tx_render.try_send(render_data);
    }

    println!("✅ Inference thread exited");
}
