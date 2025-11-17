use super::bytetrack::ByteTracker;
use super::tracker::PersonTracker;
/// ggez渲染模块
/// ggez rendering module with GPU-accelerated resizing
use super::types::{DecodedFrame, InferredFrame, ResizedFrame};
use crossbeam_channel::{Receiver, Sender};
use ggez::event::EventHandler;
use ggez::graphics::{
    self, Canvas, Color, DrawParam, Image, Rect, ScreenImage, Text, TextFragment,
};
use ggez::{Context, GameResult};
use std::time::Instant;

/// 追踪算法类型
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackerType {
    DeepSort,  // DeepSort + OSNet-AIN (高精度)
    ByteTrack, // ByteTrack (高速度)
}

/// 渲染主线程: 接收解码帧 → GPU缩放显示 → GPU离屏渲染缩放到320x320 → 绘制检测结果
pub struct YoloApp {
    rx_decode: Receiver<DecodedFrame>,
    rx_result: Receiver<InferredFrame>,
    tx_to_inference: Sender<ResizedFrame>,

    current_frame: Option<DecodedFrame>, // 当前帧(动态分辨率)
    render_frame: Option<DecodedFrame>,  // 渲染用的帧(双缓冲,避免数据竞争)
    current_results: Option<InferredFrame>,

    // 调试用: 保存缩放后的推理图像 (320x320 RGBA)
    debug_inference_image: Option<Vec<u8>>,

    // 人形追踪器 (支持 DeepSort / ByteTrack)
    tracker_type: TrackerType,
    person_tracker: Option<PersonTracker>,
    byte_tracker: Option<ByteTracker>,

    frame_count: u64,
    render_count: u64,
    render_last: Instant,
    render_fps: f64,

    // Video resolution (从第一帧获取)
    video_width: u32,
    video_height: u32,

    // 解码器信息
    decoder_name: String,

    // 推理输入尺寸和窗口尺寸
    inf_size: u32,
    window_width: f32,
    window_height: f32,

    scale_image: ScreenImage,

    // 缩略图拖动状态
    thumbnail_x: f32,
    thumbnail_y: f32,
    is_dragging: bool,
    drag_offset_x: f32,
    drag_offset_y: f32,

    // 缩放模式切换 (true=GPU, false=CPU)
    use_gpu_resize: bool,

    // 缩略图显示开关
    show_thumbnail: bool,

    // 模型文件路径
    detect_model_path: String,
    pose_model_path: String,
}

impl YoloApp {
    pub fn new(
        ctx: &mut Context,
        rx_decode: Receiver<DecodedFrame>,
        rx_result: Receiver<InferredFrame>,
        tx_to_inference: Sender<ResizedFrame>,
        inf_size: u32,
        window_width: f32,
        window_height: f32,
        detect_model_path: String,
        pose_model_path: String,
        tracker_type: TrackerType,
    ) -> Self {
        // 创建320x320的离屏缓冲区用于GPU缩放
        // ScreenImage的width/height参数是相对于窗口的比例
        let scale_ratio_x = inf_size as f32 / window_width;
        let scale_ratio_y = inf_size as f32 / window_height;
        // 使用1个采样(标准渲染),GPU线性插值速度快但质量略低于CPU的Lanczos3
        let scale_image = ScreenImage::new(ctx, None, scale_ratio_x, scale_ratio_y, 1);

        // 初始化缩略图位置(右下角)
        let thumbnail_x = window_width - inf_size as f32 - 10.0;
        let thumbnail_y = window_height - inf_size as f32 - 10.0;

        // 根据类型初始化追踪器
        let (person_tracker, byte_tracker) = match tracker_type {
            TrackerType::DeepSort => (Some(PersonTracker::new()), None),
            TrackerType::ByteTrack => (None, Some(ByteTracker::new())),
        };

        Self {
            rx_decode,
            rx_result,
            tx_to_inference,
            current_frame: None,
            render_frame: None,
            current_results: None,
            debug_inference_image: None,
            tracker_type,
            person_tracker,
            byte_tracker,
            frame_count: 0,
            render_count: 0,
            render_last: Instant::now(),
            render_fps: 0.0,
            video_width: 0,
            video_height: 0,
            decoder_name: String::from("Unknown"),
            inf_size,
            window_width,
            window_height,
            scale_image,
            thumbnail_x,
            thumbnail_y,
            is_dragging: false,
            drag_offset_x: 0.0,
            drag_offset_y: 0.0,
            use_gpu_resize: false, // 默认使用CPU (质量更高)
            show_thumbnail: false, // 默认不显示缩略图
            detect_model_path,
            pose_model_path,
        }
    }

    /// GPU离屏渲染缩放: 动态分辨率 → 320x320 推理尺寸
    fn resize_for_inference_gpu(&mut self, ctx: &mut Context, rgba_data: &[u8]) -> GameResult {
        // 1. Create source image from RGBA buffer
        let src_image = Image::from_pixels(
            ctx,
            rgba_data,
            graphics::ImageFormat::Rgba8UnormSrgb,
            self.video_width,
            self.video_height,
        );

        // 2. 创建离屏Canvas (黑色背景)
        let mut canvas = Canvas::from_screen_image(
            ctx,
            &mut self.scale_image,
            Color::new(0.0, 0.0, 0.0, 1.0), // 黑色不透明背景
        );

        // 设置Canvas的屏幕坐标为320x320 (覆盖默认的窗口尺寸)
        canvas.set_screen_coordinates(graphics::Rect::new(
            0.0,
            0.0,
            self.inf_size as f32,
            self.inf_size as f32,
        ));

        // 3. 计算缩放比例并绘制
        let scale_x = self.inf_size as f32 / self.video_width as f32;
        let scale_y = self.inf_size as f32 / self.video_height as f32;

        canvas.draw(&src_image, DrawParam::default().scale([scale_x, scale_y]));
        canvas.finish(ctx)?;

        // 4. Read back GPU pixel data
        let mut rgba_data = self.scale_image.image(ctx).to_pixels(ctx)?;

        // GPU可能返回BGRA格式,需要转换为RGBA (交换R和B通道)
        for chunk in rgba_data.chunks_exact_mut(4) {
            chunk.swap(0, 2); // 交换R和B通道 (BGRA → RGBA)
        }

        // 保存缩放后的图像用于调试显示
        self.debug_inference_image = Some(rgba_data.clone());

        // 5. Convert RGBA → RGB (极速批量复制)
        let pixel_count = (self.inf_size * self.inf_size) as usize;
        let mut resized_rgb: Vec<u8> = Vec::with_capacity(pixel_count * 3);

        unsafe {
            let src = rgba_data.as_ptr();
            let dst_ptr = resized_rgb.as_mut_ptr();

            // 批量处理: 每次4个像素
            let chunks = pixel_count / 4;
            for i in 0..chunks {
                let base_src = i * 16; // 4像素 * 4字节RGBA
                let base_dst = i * 12; // 4像素 * 3字节RGB

                // 使用copy_nonoverlapping批量复制RGB通道
                std::ptr::copy_nonoverlapping(src.add(base_src), dst_ptr.add(base_dst), 3);
                std::ptr::copy_nonoverlapping(src.add(base_src + 4), dst_ptr.add(base_dst + 3), 3);
                std::ptr::copy_nonoverlapping(src.add(base_src + 8), dst_ptr.add(base_dst + 6), 3);
                std::ptr::copy_nonoverlapping(src.add(base_src + 12), dst_ptr.add(base_dst + 9), 3);
            }

            // 处理剩余像素
            let remainder = pixel_count % 4;
            let base_src = chunks * 16;
            let base_dst = chunks * 12;
            for i in 0..remainder {
                std::ptr::copy_nonoverlapping(
                    src.add(base_src + i * 4),
                    dst_ptr.add(base_dst + i * 3),
                    3,
                );
            }

            resized_rgb.set_len(pixel_count * 3);
        }

        // 6. Send to inference thread (使用阻塞send,确保数据送达)
        if let Err(e) = self.tx_to_inference.send(ResizedFrame {
            rgb_data: resized_rgb,
        }) {
            eprintln!("❌ 发送推理数据失败: {}", e);
        }

        Ok(())
    }

    /// CPU直接缩放: 动态分辨率 → 320x320 推理尺寸 (当前使用)
    fn resize_for_inference(&mut self, _ctx: &mut Context, rgba_data: &[u8]) -> GameResult {
        use image::{imageops, RgbaImage};

        // 1. 转换为RgbaImage
        let src_img =
            match RgbaImage::from_raw(self.video_width, self.video_height, rgba_data.to_vec()) {
                Some(img) => img,
                None => {
                    eprintln!("❌ 无法创建源图像");
                    return Ok(());
                }
            };

        // 2. CPU缩放到320x320 (使用快速Triangle算法,FastestV2不需要高质量)
        let resized_img = imageops::resize(
            &src_img,
            self.inf_size,
            self.inf_size,
            imageops::FilterType::Triangle, // 快速算法,适合实时检测
        );

        // 3. 获取RGBA数据
        let rgba_data = resized_img.into_raw();

        // 保存缩放后的图像用于调试显示
        self.debug_inference_image = Some(rgba_data.clone());

        // 5. Convert RGBA → RGB (极速批量复制)
        let pixel_count = (self.inf_size * self.inf_size) as usize;
        let mut resized_rgb: Vec<u8> = Vec::with_capacity(pixel_count * 3);

        unsafe {
            let src = rgba_data.as_ptr();
            let dst_ptr = resized_rgb.as_mut_ptr();

            // 批量处理: 每次4个像素
            let chunks = pixel_count / 4;
            for i in 0..chunks {
                let base_src = i * 16; // 4像素 * 4字节RGBA
                let base_dst = i * 12; // 4像素 * 3字节RGB

                // 使用copy_nonoverlapping批量复制RGB通道
                std::ptr::copy_nonoverlapping(src.add(base_src), dst_ptr.add(base_dst), 3);
                std::ptr::copy_nonoverlapping(src.add(base_src + 4), dst_ptr.add(base_dst + 3), 3);
                std::ptr::copy_nonoverlapping(src.add(base_src + 8), dst_ptr.add(base_dst + 6), 3);
                std::ptr::copy_nonoverlapping(src.add(base_src + 12), dst_ptr.add(base_dst + 9), 3);
            }

            // 处理剩余像素
            let remainder = pixel_count % 4;
            let base_src = chunks * 16;
            let base_dst = chunks * 12;
            for i in 0..remainder {
                std::ptr::copy_nonoverlapping(
                    src.add(base_src + i * 4),
                    dst_ptr.add(base_dst + i * 3),
                    3,
                );
            }

            resized_rgb.set_len(pixel_count * 3);
        }

        // 6. Send to inference thread (使用阻塞send,确保数据送达)
        if let Err(e) = self.tx_to_inference.send(ResizedFrame {
            rgb_data: resized_rgb,
        }) {
            eprintln!("❌ 发送推理数据失败: {}", e);
        }

        Ok(())
    }
}

impl EventHandler for YoloApp {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        // 处理所有缓冲帧(不丢帧,全部推理)
        let mut processed_count = 0;
        while let Ok(frame) = self.rx_decode.try_recv() {
            processed_count += 1;

            // 第一帧时更新视频分辨率
            if self.video_width == 0 {
                self.video_width = frame.width;
                self.video_height = frame.height;
            }
            // 更新解码器名称
            self.decoder_name = frame.decoder_name.clone();

            // 双缓冲: current_frame 用于推理, render_frame 用于渲染
            // 先将旧的 current_frame 移动到 render_frame
            if let Some(old_frame) = self.current_frame.take() {
                self.render_frame = Some(old_frame);
            }
            self.current_frame = Some(frame.clone());
            self.frame_count += 1;

            // FastestV2每帧都推理
            let is_fastestv2 = self.detect_model_path.contains("fastestv2");
            let should_inference = if is_fastestv2 {
                true // FastestV2每帧都推理
            } else if self.frame_count < 10 {
                self.frame_count % 3 == 0
            } else {
                self.frame_count % 8 == 0
            };

            if should_inference {
                // CPU resize并发送到推理线程
                if self.use_gpu_resize {
                    self.resize_for_inference_gpu(ctx, &frame.rgba_data)?;
                } else {
                    self.resize_for_inference(ctx, &frame.rgba_data)?;
                }
            }
        }

        // 打印统计
        if processed_count > 0 {
            println!("🎨 渲染update处理 {} 帧", processed_count);
        }

        // Update inference results when available
        if let Ok(results) = self.rx_result.try_recv() {
            static mut RESULT_COUNT: u32 = 0;
            unsafe {
                RESULT_COUNT += 1;
                if RESULT_COUNT % 30 == 1 {
                    eprintln!(
                        "📊 渲染器收到检测结果: {}人 | {}关键点组",
                        results.bboxes.len(),
                        results.keypoints.len()
                    );
                }
            }
            self.current_results = Some(results);
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::BLACK);

        // 使用 render_frame 渲染(稳定),使用 current_frame 推理(最新)
        if self.render_frame.is_none() && self.current_frame.is_some() {
            // 第一次渲染,复制一份
            self.render_frame = self.current_frame.clone();
        }

        // 推理已经在update()中处理,这里只负责渲染

        if let Some(ref decoded) = self.render_frame {
            self.render_count += 1;

            // Calculate render FPS
            if self.render_last.elapsed().as_secs_f64() >= 1.0 {
                self.render_fps =
                    self.render_count as f64 / self.render_last.elapsed().as_secs_f64();
                self.render_last = Instant::now();
                self.render_count = 0;
            }

            let decode_fps = decoded.decode_fps;
            let width = decoded.width;
            let height = decoded.height;

            // Create ggez image from RGBA data (零拷贝,直接使用!)
            let image = Image::from_pixels(
                ctx,
                &decoded.rgba_data,
                graphics::ImageFormat::Rgba8UnormSrgb,
                width,
                height,
            );

            // GPU resize: scale to window size using ggez
            let (window_width, window_height) = ctx.gfx.drawable_size();
            let scale_x = window_width / width as f32;
            let scale_y = window_height / height as f32;

            // Draw with GPU scaling
            canvas.draw(&image, DrawParam::default().scale([scale_x, scale_y]));

            // Draw detection results if available
            if let Some(ref results) = self.current_results {
                // Scale factors for coordinate mapping (inference -> 动态分辨率 -> window)
                // 使用实时窗口尺寸,支持动态缩放
                let (current_window_width, current_window_height) = ctx.gfx.drawable_size();
                let scale_x = (self.video_width as f32 / self.inf_size as f32)
                    * (current_window_width / self.video_width as f32);
                let scale_y = (self.video_height as f32 / self.inf_size as f32)
                    * (current_window_height / self.video_height as f32);

                // 1. 提取人形检测框 (class_id = 0 是"人")
                use super::types::BBox;
                let person_detections: Vec<BBox> = results
                    .bboxes
                    .iter()
                    .filter(|bbox| bbox.class_id == 0) // 过滤出"人"
                    .cloned()
                    .collect();

                // 2. 准备原始图像数据(用于ReID)
                // frame_data 带有原始帧的宽高 (用于 deep ReID 裁剪)
                let frame_data = self
                    .current_frame
                    .as_ref()
                    .map(|frame| (frame.rgba_data.as_slice(), frame.width, frame.height));

                // 3. 将检测框与关键点从推理坐标 (inf_size) 映射到原始帧坐标
                //    这样后续的 ReID 裁剪/特征提取会在正确的像素空间进行
                let scale_to_frame_x = self.video_width as f32 / self.inf_size as f32;
                let scale_to_frame_y = self.video_height as f32 / self.inf_size as f32;

                let mut scaled_detections: Vec<BBox> = Vec::with_capacity(person_detections.len());
                for det in &person_detections {
                    scaled_detections.push(BBox {
                        x1: det.x1 * scale_to_frame_x,
                        y1: det.y1 * scale_to_frame_y,
                        x2: det.x2 * scale_to_frame_x,
                        y2: det.y2 * scale_to_frame_y,
                        confidence: det.confidence,
                        class_id: det.class_id,
                    });
                }

                // 同步缩放关键点（如果有）到原始帧坐标系
                let mut scaled_keypoints: Vec<super::types::PoseKeypoints> = Vec::new();
                for kp in &results.keypoints {
                    let mut pts = Vec::with_capacity(kp.points.len());
                    for (x, y, c) in &kp.points {
                        pts.push((x * scale_to_frame_x, y * scale_to_frame_y, *c));
                    }
                    scaled_keypoints.push(super::types::PoseKeypoints { points: pts });
                }

                // 4. 更新追踪器并绘制 (传入映射到原始帧坐标的检测框、关键点和原始图像)
                // tracker 返回的 bbox 在原始帧坐标系，需要映射到窗口坐标
                let frame_to_window_x = current_window_width / self.video_width as f32;
                let frame_to_window_y = current_window_height / self.video_height as f32;

                match self.tracker_type {
                    TrackerType::DeepSort => {
                        if let Some(ref mut tracker) = self.person_tracker {
                            // DeepSort: 需要关键点和原始图像数据做 ReID
                            let tracked_persons =
                                tracker.update(&scaled_detections, &scaled_keypoints, frame_data);

                            for tracked in tracked_persons {
                                // 从原始帧坐标映射到窗口坐标
                                let x1 = tracked.bbox.x1 * frame_to_window_x;
                                let y1 = tracked.bbox.y1 * frame_to_window_y;
                                let w = (tracked.bbox.x2 - tracked.bbox.x1) * frame_to_window_x;
                                let h = (tracked.bbox.y2 - tracked.bbox.y1) * frame_to_window_y;

                                // 使用追踪对象的颜色绘制边界框 (3像素厚)
                                let color = Color::from_rgb(
                                    tracked.color.0,
                                    tracked.color.1,
                                    tracked.color.2,
                                );
                                for thickness in 0..3 {
                                    let offset = thickness as f32;
                                    let rect = graphics::Rect::new(
                                        x1 + offset,
                                        y1 + offset,
                                        w - offset * 2.0,
                                        h - offset * 2.0,
                                    );
                                    let mesh = graphics::Mesh::new_rectangle(
                                        ctx,
                                        graphics::DrawMode::stroke(1.0),
                                        rect,
                                        color,
                                    )?;
                                    canvas.draw(&mesh, DrawParam::default());
                                }

                                // 绘制追踪ID (左上角)
                                let id_text = format!("ID:{}", tracked.id);
                                let id_fragment = TextFragment::new(id_text)
                                    .font("MicrosoftYaHei")
                                    .scale(22.0)
                                    .color(color);
                                let id_text_obj = Text::new(id_fragment);
                                canvas
                                    .draw(&id_text_obj, DrawParam::default().dest([x1, y1 - 25.0]));

                                // 绘制轨迹 (最近50个点连线)
                                if tracked.trajectory.len() > 1 {
                                    let mut line_points = Vec::new();
                                    for point in &tracked.trajectory {
                                        // trajectory 中的点也在原始帧坐标系，映射到窗口
                                        line_points.push([
                                            point.x * frame_to_window_x,
                                            point.y * frame_to_window_y,
                                        ]);
                                    }

                                    // 使用Mesh绘制折线
                                    let line =
                                        graphics::Mesh::new_line(ctx, &line_points, 2.0, color)?;
                                    canvas.draw(&line, DrawParam::default());
                                }
                            }
                        }
                    }
                    TrackerType::ByteTrack => {
                        if let Some(ref mut tracker) = self.byte_tracker {
                            // ByteTrack: 只需要检测框 (纯 IOU 匹配)
                            let tracked_persons = tracker.update(&scaled_detections);

                            for tracked in tracked_persons {
                                // 从原始帧坐标映射到窗口坐标
                                let x1 = tracked.bbox.x1 * frame_to_window_x;
                                let y1 = tracked.bbox.y1 * frame_to_window_y;
                                let w = (tracked.bbox.x2 - tracked.bbox.x1) * frame_to_window_x;
                                let h = (tracked.bbox.y2 - tracked.bbox.y1) * frame_to_window_y;

                                // 使用追踪对象的颜色绘制边界框 (3像素厚)
                                let color = Color::from_rgb(
                                    tracked.color.0,
                                    tracked.color.1,
                                    tracked.color.2,
                                );
                                for thickness in 0..3 {
                                    let offset = thickness as f32;
                                    let rect = graphics::Rect::new(
                                        x1 + offset,
                                        y1 + offset,
                                        w - offset * 2.0,
                                        h - offset * 2.0,
                                    );
                                    let mesh = graphics::Mesh::new_rectangle(
                                        ctx,
                                        graphics::DrawMode::stroke(1.0),
                                        rect,
                                        color,
                                    )?;
                                    canvas.draw(&mesh, DrawParam::default());
                                }

                                // 绘制追踪ID (左上角)
                                let id_text = format!("ID:{}", tracked.id);
                                let id_fragment = TextFragment::new(id_text)
                                    .font("MicrosoftYaHei")
                                    .scale(22.0)
                                    .color(color);
                                let id_text_obj = Text::new(id_fragment);
                                canvas
                                    .draw(&id_text_obj, DrawParam::default().dest([x1, y1 - 25.0]));

                                // 绘制轨迹 (最近50个点连线)
                                if tracked.trajectory.len() > 1 {
                                    let mut line_points = Vec::new();
                                    for point in &tracked.trajectory {
                                        // trajectory 中的点也在原始帧坐标系，映射到窗口
                                        line_points.push([
                                            point.x * frame_to_window_x,
                                            point.y * frame_to_window_y,
                                        ]);
                                    }

                                    // 使用Mesh绘制折线
                                    let line =
                                        graphics::Mesh::new_line(ctx, &line_points, 2.0, color)?;
                                    canvas.draw(&line, DrawParam::default());
                                }
                            }
                        }
                    }
                }

                // 4. 绘制非人类的检测框 (原样绘制绿色)
                for bbox in &results.bboxes {
                    if bbox.class_id != 0 {
                        let x1 = bbox.x1 * scale_x;
                        let y1 = bbox.y1 * scale_y;
                        let w = (bbox.x2 - bbox.x1) * scale_x;
                        let h = (bbox.y2 - bbox.y1) * scale_y;

                        // Draw green rectangle (3 pixels thick)
                        for thickness in 0..3 {
                            let offset = thickness as f32;
                            let rect = graphics::Rect::new(
                                x1 + offset,
                                y1 + offset,
                                w - offset * 2.0,
                                h - offset * 2.0,
                            );
                            let mesh = graphics::Mesh::new_rectangle(
                                ctx,
                                graphics::DrawMode::stroke(1.0),
                                rect,
                                Color::from_rgb(0, 255, 0),
                            )?;
                            canvas.draw(&mesh, DrawParam::default());
                        }
                    }
                }

                // Draw pose keypoints using ggez
                for pose in &results.keypoints {
                    for (x, y, conf) in &pose.points {
                        if *conf > 0.5 {
                            let px = x * scale_x;
                            let py = y * scale_y;

                            // Draw keypoint as circle
                            let circle = graphics::Mesh::new_circle(
                                ctx,
                                graphics::DrawMode::fill(),
                                [px, py],
                                4.0,
                                0.1,
                                Color::from_rgb(255, 0, 0),
                            )?;
                            canvas.draw(&circle, DrawParam::default());
                        }
                    }
                }

                // Draw FPS info at top-left corner (统一样式 - 白色)
                let fps_text = format!(
                    "FPS - 解码:{:.1} | 推理:{:.1}({:.1}ms) | 渲染:{:.1} | 检测:{}人",
                    decode_fps,
                    results.inference_fps,
                    results.inference_ms,
                    self.render_fps,
                    results.bboxes.len()
                );

                let fps_fragment = TextFragment::new(fps_text)
                    .font("MicrosoftYaHei")
                    .scale(24.0); // 调大字体
                let fps_display = Text::new(fps_fragment);
                canvas.draw(
                    &fps_display,
                    DrawParam::default().dest([10.0, 10.0]).color(Color::WHITE),
                );

                // 显示解码器信息 (统一样式 - 白色)
                let decoder_text = format!("解码器: {}", self.decoder_name);
                let decoder_fragment = TextFragment::new(decoder_text)
                    .font("MicrosoftYaHei")
                    .scale(24.0); // 调大字体
                let decoder_display = Text::new(decoder_fragment);
                canvas.draw(
                    &decoder_display,
                    DrawParam::default()
                        .dest([10.0, 40.0]) // 调整间距适应更大字体
                        .color(Color::WHITE), // 白色
                );

                // 显示追踪信息
                let tracker_stats = match self.tracker_type {
                    TrackerType::DeepSort => {
                        if let Some(ref tracker) = self.person_tracker {
                            tracker.get_stats()
                        } else {
                            String::from("追踪: 未初始化")
                        }
                    }
                    TrackerType::ByteTrack => {
                        if let Some(ref tracker) = self.byte_tracker {
                            tracker.get_stats()
                        } else {
                            String::from("追踪: 未初始化")
                        }
                    }
                };
                let tracker_fragment = TextFragment::new(tracker_stats)
                    .font("MicrosoftYaHei")
                    .scale(24.0);
                let tracker_display = Text::new(tracker_fragment);
                canvas.draw(
                    &tracker_display,
                    DrawParam::default().dest([10.0, 70.0]).color(Color::WHITE),
                );

                // 显示缩放模式 (GPU/CPU)
                let resize_mode = if self.use_gpu_resize { "GPU" } else { "CPU" };
                let mode_text = format!("缩放模式: {} (空格切换)", resize_mode);
                let mode_fragment = TextFragment::new(mode_text)
                    .font("MicrosoftYaHei")
                    .scale(24.0);
                let mode_display = Text::new(mode_fragment);
                canvas.draw(
                    &mode_display,
                    DrawParam::default().dest([10.0, 100.0]).color(Color::WHITE),
                );

                // 显示追踪算法及特征
                let tracker_info = match self.tracker_type {
                    TrackerType::DeepSort => {
                        let reid_status = if let Some(ref tracker) = self.person_tracker {
                            if tracker.has_reid_model() {
                                "OSNet-AIN x1.0 (mAP 84.9%)"
                            } else {
                                "姿态ReID (64维)"
                            }
                        } else {
                            "未初始化"
                        };
                        format!("追踪: DeepSort + {}", reid_status)
                    }
                    TrackerType::ByteTrack => String::from("追踪: ByteTrack (纯IOU)"),
                };
                let tracker_info_fragment = TextFragment::new(tracker_info)
                    .font("MicrosoftYaHei")
                    .scale(24.0);
                let tracker_info_display = Text::new(tracker_info_fragment);
                canvas.draw(
                    &tracker_info_display,
                    DrawParam::default().dest([10.0, 130.0]).color(Color::WHITE),
                );

                // 显示模型名称 (只显示文件名,不显示完整路径)
                let detect_name = std::path::Path::new(&self.detect_model_path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&self.detect_model_path);
                let pose_name = std::path::Path::new(&self.pose_model_path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&self.pose_model_path);
                let model_text = format!("检测: {} | 姿态: {}", detect_name, pose_name);
                let model_fragment = TextFragment::new(model_text)
                    .font("MicrosoftYaHei")
                    .scale(20.0);
                let model_display = Text::new(model_fragment);
                canvas.draw(
                    &model_display,
                    DrawParam::default().dest([10.0, 160.0]).color(Color::WHITE), // 改为白色
                );
            }
        }

        // 🔍 调试显示: 绘制可拖动的缩略图 (320x320, 按T键切换显示/隐藏)
        if self.show_thumbnail {
            if let Some(ref debug_rgba) = self.debug_inference_image {
                // 从RGBA数据创建图像
                let thumbnail = Image::from_pixels(
                    ctx,
                    debug_rgba,
                    graphics::ImageFormat::Rgba8UnormSrgb,
                    self.inf_size,
                    self.inf_size,
                );

                // 使用拖动后的位置
                let x = self.thumbnail_x;
                let y = self.thumbnail_y;

                // 直接绘制320x320的缩略图 (无需缩放)
                canvas.draw(&thumbnail, DrawParam::default().dest([x, y]));

                // 绘制红色边框 (3像素厚)
                for thickness in 0..3 {
                    let offset = thickness as f32;
                    let border_rect = graphics::Rect::new(
                        x + offset,
                        y + offset,
                        self.inf_size as f32 - offset * 2.0,
                        self.inf_size as f32 - offset * 2.0,
                    );
                    let border_mesh = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::stroke(1.0),
                        border_rect,
                        Color::from_rgb(255, 0, 0), // 红色边框
                    )?;
                    canvas.draw(&border_mesh, DrawParam::default());
                }

                // 添加标签 (标签在缩略图上方)
                let label =
                    TextFragment::new(format!("推理输入 {}x{}", self.inf_size, self.inf_size))
                        .font("MicrosoftYaHei")
                        .scale(18.0);
                let label_text = Text::new(label);
                canvas.draw(
                    &label_text,
                    DrawParam::default()
                        .dest([x, y - 25.0]) // 标签在缩略图上方
                        .color(Color::WHITE), // 改为白色
                );
            }
        }

        canvas.finish(ctx)?;
        Ok(())
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        button: ggez::event::MouseButton,
        x: f32,
        y: f32,
    ) -> GameResult {
        if button == ggez::event::MouseButton::Left {
            // 检查是否点击在缩略图区域内
            if x >= self.thumbnail_x
                && x <= self.thumbnail_x + self.inf_size as f32
                && y >= self.thumbnail_y
                && y <= self.thumbnail_y + self.inf_size as f32
            {
                self.is_dragging = true;
                self.drag_offset_x = x - self.thumbnail_x;
                self.drag_offset_y = y - self.thumbnail_y;
            }
        }
        Ok(())
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        button: ggez::event::MouseButton,
        _x: f32,
        _y: f32,
    ) -> GameResult {
        if button == ggez::event::MouseButton::Left {
            self.is_dragging = false;
        }
        Ok(())
    }

    fn mouse_motion_event(
        &mut self,
        _ctx: &mut Context,
        x: f32,
        y: f32,
        _dx: f32,
        _dy: f32,
    ) -> GameResult {
        if self.is_dragging {
            // 更新缩略图位置,限制在窗口范围内
            self.thumbnail_x = (x - self.drag_offset_x)
                .max(0.0)
                .min(self.window_width - self.inf_size as f32);
            self.thumbnail_y = (y - self.drag_offset_y)
                .max(0.0)
                .min(self.window_height - self.inf_size as f32);
        }
        Ok(())
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        input: ggez::input::keyboard::KeyInput,
        _repeated: bool,
    ) -> GameResult {
        if let Some(keycode) = input.keycode {
            // 空格键切换CPU/GPU缩放模式
            if keycode == ggez::input::keyboard::KeyCode::Space {
                self.use_gpu_resize = !self.use_gpu_resize;
                let mode = if self.use_gpu_resize { "GPU" } else { "CPU" };
                println!("🔄 切换缩放模式: {}", mode);
            }
            // T键切换缩略图显示/隐藏
            else if keycode == ggez::input::keyboard::KeyCode::T {
                self.show_thumbnail = !self.show_thumbnail;
                let status = if self.show_thumbnail {
                    "显示"
                } else {
                    "隐藏"
                };
                println!("👁️  缩略图: {}", status);
            }
        }
        Ok(())
    }
}
