use crossbeam_channel::Receiver;
use egui_macroquad::egui;
use macroquad::prelude::*;
use std::time::Instant;
use yolov8_rs::detection::detector::DetectionResult;
use yolov8_rs::detection::types::DecodedFrame;
use yolov8_rs::input::{get_video_devices, switch_decoder_source, InputSource, VideoDevice};
use yolov8_rs::xbus::{self, Subscription};
use yolov8_rs::SKELETON;

pub struct Renderer {
    _frame_sub: Subscription,
    _result_sub: Subscription,
    render_frame_buffer: Receiver<RenderFrame>,
    last_frame: Option<Texture2D>,
    last_detection: Option<DetectionResult>,
    render_count: u64,
    render_last: Instant,
    render_fps: f64,
    preview_visible: bool,

    // 系统配置信息
    detect_model_name: String,
    pose_model_name: String,
    tracker_name: String,
    detect_fps: f64,
    decode_fps: f64,

    // egui 参数调整
    show_params_panel: bool,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_age: i32,
    pub min_hits: i32,

    // 输入源配置界面
    show_input_source_panel: bool,
    input_source_type: usize, // 0=RTSP, 1=摄像头
    rtsp_url: String,
    camera_id: i32,

    // 设备列表
    video_devices: Vec<VideoDevice>,
    selected_device_index: usize,
    devices_loaded: bool,
}

enum RenderFrame {
    Video(DecodedFrame),
    Detection(DetectionResult),
}

impl Renderer {
    pub fn new(detect_model: String, pose_model: String, tracker: String) -> Self {
        println!("渲染器启动");
        let (tx, rx) = crossbeam_channel::bounded(25);

        // 订阅DecodedFrame
        let tx1 = tx.clone();
        let frame_sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            if let Err(err) = tx1.try_send(RenderFrame::Video(frame.clone())) {
                eprintln!("渲染器通道发送DecodedFrame失败: {}", err);
            }
        });

        // 订阅DetectionResult
        let result_sub = xbus::subscribe::<DetectionResult, _>(move |result| {
            if let Err(err) = tx.try_send(RenderFrame::Detection(result.clone())) {
                eprintln!("渲染器通道发送DetectionResult失败: {}", err);
            }
        });

        Self {
            render_frame_buffer: rx,
            last_frame: None,
            last_detection: None,
            _frame_sub: frame_sub,
            _result_sub: result_sub,
            render_count: 0,
            render_last: Instant::now(),
            render_fps: 0.0,
            preview_visible: true,
            detect_model_name: detect_model,
            pose_model_name: pose_model,
            tracker_name: tracker,
            detect_fps: 0.0,
            decode_fps: 0.0,
            show_params_panel: false,
            confidence_threshold: 0.5,
            iou_threshold: 0.45,
            max_age: 30,
            min_hits: 3,
            show_input_source_panel: true,
            input_source_type: 0,
            rtsp_url: "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1&subtype=0"
                .to_string(),
            camera_id: 0,
            video_devices: Vec::new(),
            selected_device_index: 0,
            devices_loaded: false,
        }
    }

    pub fn update(&mut self) {
        // 处理帧缓冲
        if let Some(frame) = self.render_frame_buffer.try_iter().last() {
            match frame {
                RenderFrame::Video(decoded_frame) => {
                    let texture = Texture2D::from_rgba8(
                        decoded_frame.width as u16,
                        decoded_frame.height as u16,
                        &decoded_frame.rgba_data,
                    );
                    texture.set_filter(FilterMode::Linear);
                    self.last_frame = Some(texture);
                }
                RenderFrame::Detection(detection_result) => {
                    self.last_detection = Some(detection_result);
                }
            }
        }

        // 更新FPS和检测状态
        if let Some(result) = &self.last_detection {
            self.detect_fps = result.inference_fps;
            // decode_fps 从解码器获取,暂时使用推理FPS
            self.decode_fps = result.inference_fps;
        }
    }

    pub fn draw(&mut self) {
        clear_background(BLACK);

        // 绘制视频帧
        if let Some(texture) = &self.last_frame {
            let scale_x = screen_width() / texture.width();
            let scale_y = screen_height() / texture.height();

            draw_texture_ex(
                texture,
                0.0,
                0.0,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(screen_width(), screen_height())),
                    ..Default::default()
                },
            );

            // 绘制检测框
            if let Some(detection_result) = &self.last_detection {
                for bbox in &detection_result.bboxes {
                    let x1 = bbox.x1 * scale_x;
                    let y1 = bbox.y1 * scale_y;
                    let x2 = bbox.x2 * scale_x;
                    let y2 = bbox.y2 * scale_y;

                    // 绘制边框
                    draw_rectangle_lines(x1, y1, x2 - x1, y2 - y1, 3.0, GREEN);

                    // 绘制标签
                    let label = format!("ID:{} {:.2}", bbox.class_id, bbox.confidence);
                    draw_text(&label, x1, y1 - 5.0, 20.0, GREEN);
                }

                // 绘制姿态骨架
                for keypoints in &detection_result.keypoints {
                    if keypoints.points.is_empty() {
                        continue;
                    }

                    // 绘制关键点
                    for (x, y, conf) in &keypoints.points {
                        if *conf > 0.3 {
                            draw_circle(*x * scale_x, *y * scale_y, 4.0, RED);
                        }
                    }

                    // 绘制骨架连接
                    for (idx1, idx2) in &SKELETON {
                        if *idx1 < keypoints.points.len() && *idx2 < keypoints.points.len() {
                            let (x1, y1, c1) = keypoints.points[*idx1];
                            let (x2, y2, c2) = keypoints.points[*idx2];
                            if c1 > 0.3 && c2 > 0.3 {
                                draw_line(
                                    x1 * scale_x,
                                    y1 * scale_y,
                                    x2 * scale_x,
                                    y2 * scale_y,
                                    2.0,
                                    YELLOW,
                                );
                            }
                        }
                    }
                }
            }
        }

        // FPS统计
        self.render_count += 1;
        let now = Instant::now();
        if now.duration_since(self.render_last).as_secs() >= 1 {
            self.render_fps =
                self.render_count as f64 / now.duration_since(self.render_last).as_secs_f64();
            self.render_count = 0;
            self.render_last = now;
        }
    }

    pub fn draw_egui(&mut self) {
        egui_macroquad::ui(|egui_ctx| {
            // FPS 统计面板
            egui::Window::new("系统状态")
                .default_pos(egui::pos2(10.0, 10.0))
                .default_size(egui::vec2(300.0, 200.0))
                .show(egui_ctx, |ui| {
                    ui.heading("性能监控");
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("渲染 FPS:");
                        ui.colored_label(egui::Color32::GREEN, format!("{:.1}", self.render_fps));
                    });

                    ui.horizontal(|ui| {
                        ui.label("解码 FPS:");
                        ui.colored_label(egui::Color32::CYAN, format!("{:.1}", self.decode_fps));
                    });

                    ui.horizontal(|ui| {
                        ui.label("检测 FPS:");
                        ui.colored_label(egui::Color32::YELLOW, format!("{:.1}", self.detect_fps));
                    });

                    ui.separator();
                    ui.heading("配置信息");

                    ui.label(format!("检测模型: {}", self.detect_model_name));
                    if !self.pose_model_name.is_empty() {
                        ui.label(format!("姿态模型: {}", self.pose_model_name));
                    }
                    if self.tracker_name != "none" {
                        ui.label(format!("追踪算法: {}", self.tracker_name));
                    }

                    ui.separator();
                    ui.label("按 P 键切换参数面板");
                    ui.label("按 I 键切换输入源面板");
                });

            // 参数调整面板
            if self.show_params_panel {
                egui::Window::new("参数调整")
                    .default_pos(egui::pos2(screen_width() - 330.0, 10.0))
                    .default_size(egui::vec2(320.0, 280.0))
                    .show(egui_ctx, |ui| {
                        ui.heading("检测参数");
                        ui.separator();

                        ui.label("置信度阈值:");
                        ui.add(egui::Slider::new(&mut self.confidence_threshold, 0.0..=1.0));

                        ui.label("IOU阈值:");
                        ui.add(egui::Slider::new(&mut self.iou_threshold, 0.0..=1.0));

                        ui.separator();
                        ui.heading("追踪参数");

                        ui.label("最大丢失帧数:");
                        ui.add(egui::Slider::new(&mut self.max_age, 1..=100));

                        ui.label("最小命中次数:");
                        ui.add(egui::Slider::new(&mut self.min_hits, 1..=10));

                        ui.separator();
                        ui.label("按 P 键隐藏此面板");
                    });
            }

            // 输入源配置面板
            if self.show_input_source_panel {
                egui::Window::new("输入源配置")
                    .default_pos(egui::pos2(
                        screen_width() / 2.0 - 200.0,
                        screen_height() / 2.0 - 150.0,
                    ))
                    .default_size(egui::vec2(400.0, 300.0))
                    .show(egui_ctx, |ui| {
                        ui.heading("选择输入源");
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.input_source_type, 0, "RTSP 网络流");
                            ui.radio_value(&mut self.input_source_type, 1, "本地摄像头");
                        });

                        ui.separator();

                        if self.input_source_type == 0 {
                            ui.label("RTSP 流地址:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.rtsp_url)
                                    .desired_width(f32::INFINITY),
                            );
                            ui.label("示例: rtsp://admin:password@192.168.1.100/stream");
                        } else {
                            ui.label("摄像头设备:");

                            // 加载设备列表按钮
                            if !self.devices_loaded {
                                if ui.button("🔍 扫描可用设备").clicked() {
                                    self.video_devices = get_video_devices();
                                    self.devices_loaded = true;
                                    if !self.video_devices.is_empty() {
                                        self.selected_device_index = 0;
                                    }
                                }
                                ui.label("提示: 点击按钮扫描摄像头");
                            } else {
                                // 设备下拉列表
                                if self.video_devices.is_empty() {
                                    ui.label("⚠️ 未找到可用设备");
                                    if ui.button("🔄 重新扫描").clicked() {
                                        self.video_devices = get_video_devices();
                                    }
                                } else {
                                    egui::ComboBox::from_label("选择设备")
                                        .selected_text(format!(
                                            "[{}] {}",
                                            self.selected_device_index,
                                            self.video_devices
                                                .get(self.selected_device_index)
                                                .map(|d| d.name.as_str())
                                                .unwrap_or("未知设备")
                                        ))
                                        .show_ui(ui, |ui| {
                                            for (idx, device) in
                                                self.video_devices.iter().enumerate()
                                            {
                                                ui.selectable_value(
                                                    &mut self.selected_device_index,
                                                    idx,
                                                    format!("[{}] {}", idx, device.name),
                                                );
                                            }
                                        });

                                    if ui.button("🔄 重新扫描").clicked() {
                                        self.video_devices = get_video_devices();
                                    }
                                }
                            }
                        }

                        ui.separator();

                        if ui.button("🔄 立即切换输入源").clicked() {
                            let new_source = if self.input_source_type == 0 {
                                InputSource::Rtsp(self.rtsp_url.clone())
                            } else {
                                // 使用选中设备的索引
                                let device_id =
                                    if self.devices_loaded && !self.video_devices.is_empty() {
                                        self.selected_device_index as i32
                                    } else {
                                        self.camera_id
                                    };
                                InputSource::Camera(device_id)
                            };

                            // 发送切换命令
                            switch_decoder_source(new_source);
                        }

                        ui.separator();
                        ui.label("💡 使用说明:");
                        ui.label("  1. 选择输入源类型");
                        ui.label("  2. 配置相应参数");
                        ui.label("  3. 点击'立即切换'按钮");
                        ui.label("  4. 等待1秒自动完成切换");
                        ui.label("");
                        ui.label("⚡ 支持热切换,无需重启程序!");
                        ui.label("  • 按 I 键隐藏此面板");
                    });
            }

            // Resized 预览图像面板
            if self.preview_visible {
                if let Some(detection_result) = &self.last_detection {
                    egui::Window::new("推理输入预览")
                        .default_pos(egui::pos2(screen_width() - 330.0, 310.0))
                        .default_size(egui::vec2(320.0, 360.0))
                        .show(egui_ctx, |ui| {
                            ui.label(format!(
                                "输入尺寸: {}x{}",
                                detection_result.resized_size, detection_result.resized_size
                            ));
                            ui.separator();

                            // 创建或更新纹理
                            if let Some(resized_data) = &detection_result.resized_image {
                                if resized_data.len()
                                    == (detection_result.resized_size
                                        * detection_result.resized_size
                                        * 4) as usize
                                {
                                    let size = detection_result.resized_size as usize;
                                    let image = egui::ColorImage::from_rgba_unmultiplied(
                                        [size, size],
                                        resized_data,
                                    );

                                    let texture = egui_ctx.load_texture(
                                        "resized_preview",
                                        image,
                                        egui::TextureOptions::LINEAR,
                                    );

                                    ui.image(egui::ImageSource::Texture(
                                        egui::load::SizedTexture::new(
                                            texture.id(),
                                            egui::vec2(300.0, 300.0),
                                        ),
                                    ));
                                } else {
                                    ui.label("图像数据大小不匹配");
                                }
                            } else {
                                ui.label("等待图像数据...");
                            }

                            ui.separator();
                            ui.label("按 V 键隐藏/显示此面板");
                        });
                }
            }
        });

        egui_macroquad::draw();
    }

    pub fn handle_input(&mut self) {
        // 键盘输入
        if is_key_pressed(KeyCode::V) {
            self.preview_visible = !self.preview_visible;
        }
        if is_key_pressed(KeyCode::P) {
            self.show_params_panel = !self.show_params_panel;
        }
        if is_key_pressed(KeyCode::I) {
            self.show_input_source_panel = !self.show_input_source_panel;
        }
    }
}
