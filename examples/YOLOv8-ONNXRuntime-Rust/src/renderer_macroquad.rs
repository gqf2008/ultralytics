use crossbeam_channel::{Receiver, Sender};
use egui_macroquad::egui;
use macroquad::prelude::*;
use phf::phf_map;
use std::time::Instant;
use yolov8_rs::detection::detector::DetectionResult;
use yolov8_rs::detection::types::{ConfigMessage, DecodedFrame};
use yolov8_rs::input::{get_video_devices, switch_decoder_source, InputSource, VideoDevice};
use yolov8_rs::xbus::{self, Subscription};
use yolov8_rs::SKELETON;

static MODELS: [&str; 25] = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolo-fastestv2",
    "yolo-fastest-xl",
    "yolov8n-int8",
    "yolov8m-int8",
    "nanodet",
    "nanodet-plus",
    "yolox_nano",
    "yolox_tiny",
    "yolox_s",
    "yolox_m",
    "yolox_l",
];

static MODEL_INDICES: phf::Map<&'static str, usize> = phf_map! {
    "yolov8n" => 0,
    "yolov8s" => 1,
    "yolov8m" => 2,
    "yolov8l" => 3,
    "yolov8x" => 4,
    "yolov10n" => 5,
    "yolov10s" => 6,
    "yolov10m" => 7,
    "yolov11n" => 8,
    "yolov11s" => 9,
    "yolov11m" => 10,
    "yolov5n" => 11,
    "yolov5s" => 12,
    "yolov5m" => 13,
    "yolo-fastestv2" => 14,
    "yolo-fastest-xl" => 15,
    "yolov8n-int8" => 16,
    "yolov8m-int8" => 17,
    "nanodet" => 18,
    "nanodet-plus" => 19,
    "yolox_nano" => 20,
    "yolox_tiny" => 21,
    "yolox_s" => 22,
    "yolox_m" => 23,
    "yolox_l" => 24,
};

static TRACKERS: [&str; 3] = ["DeepSORT", "ByteTrack", "无"];
static TRACKER_INDICES: phf::Map<&'static str, usize> = phf_map! {
    "deepsort" => 0,
    "bytetrack" => 1,
    "none" => 2,
    "无" => 2,
};

pub struct Renderer {
    _frame_sub: Subscription,
    _result_sub: Subscription,
    render_frame_buffer: Receiver<RenderFrame>,
    config_tx: Option<Sender<ConfigMessage>>,
    last_frame: Option<Texture2D>,
    last_detection: Option<DetectionResult>,
    render_count: u64,
    render_last: Instant,
    render_fps: f64,
    preview_visible: bool,
    show_control_panel: bool,

    // 系统配置信息
    detect_model_name: String,
    pose_model_name: String,
    tracker_name: String,
    detect_fps: f64,
    decode_fps: f64,

    // egui 参数调整
    // show_params_panel: bool, // 合并到主面板
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_age: i32,
    pub min_hits: i32,

    // 输入源配置界面
    // show_input_source_panel: bool, // 合并到主面板
    input_source_type: usize, // 0=RTSP, 1=摄像头, 2=桌面捕获
    rtsp_url: String,
    rtsp_history: Vec<String>, // RTSP 历史记录
    camera_id: i32,

    // 设备列表
    video_devices: Vec<VideoDevice>,
    selected_device_index: usize,
    devices_loaded: bool,

    // 画面缩放
    zoom_scale: f32,
    pan_offset: Vec2,
    is_panning: bool,
    last_mouse_pos: Vec2,

    // 模型配置
    // show_model_config_panel: bool, // 合并到主面板
    selected_model_index: usize,
    selected_tracker_index: usize,
    pose_enabled: bool,

    // 窗口状态
    resize_window_pos: egui::Pos2,
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
            config_tx: None,
            last_frame: None,
            last_detection: None,
            _frame_sub: frame_sub,
            _result_sub: result_sub,
            render_count: 0,
            render_last: Instant::now(),
            render_fps: 0.0,
            preview_visible: true,
            show_control_panel: true,
            detect_model_name: detect_model.clone(),
            pose_model_name: pose_model,
            tracker_name: tracker.clone(),
            detect_fps: 0.0,
            decode_fps: 0.0,
            // show_params_panel: false,
            confidence_threshold: 0.5,
            iou_threshold: 0.45,
            max_age: 30,
            min_hits: 3,
            // show_input_source_panel: true,
            input_source_type: 0,
            rtsp_url: "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1&subtype=0"
                .to_string(),
            rtsp_history: {
                let mut history = vec![
                    "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1&subtype=0"
                        .to_string(),
                ];
                if let Ok(content) = std::fs::read_to_string("rtsp_history.txt") {
                    let lines: Vec<String> = content
                        .lines()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    if !lines.is_empty() {
                        history = lines;
                    }
                }
                history
            },
            camera_id: 0,
            video_devices: Vec::new(),
            selected_device_index: 0,
            devices_loaded: false,
            zoom_scale: 1.0,
            pan_offset: Vec2::ZERO,
            is_panning: false,
            last_mouse_pos: Vec2::ZERO,
            // show_model_config_panel: true,
            selected_model_index: *MODEL_INDICES.get(detect_model.as_str()).unwrap_or(&0),
            selected_tracker_index: *TRACKER_INDICES
                .get(tracker.to_lowercase().as_str())
                .unwrap_or(&2),
            pose_enabled: false,
            resize_window_pos: egui::pos2(screen_width() - 330.0, 310.0),
        }
    }

    pub fn set_config_sender(&mut self, tx: Sender<ConfigMessage>) {
        self.config_tx = Some(tx);
    }

    // 获取当前选择的模型名称
    pub fn get_selected_model(&self) -> String {
        static SHORT_NAMES: [&str; 25] = [
            "n",
            "s",
            "m",
            "l",
            "x",
            "v10n",
            "v10s",
            "v10m",
            "v11n",
            "v11s",
            "v11m",
            "v5n",
            "v5s",
            "v5m",
            "fastest",
            "fastest-xl",
            "n-int8",
            "m-int8",
            "nanodet",
            "nanodet-plus",
            "yolox_nano",
            "yolox_tiny",
            "yolox_s",
            "yolox_m",
            "yolox_l",
        ];
        SHORT_NAMES
            .get(self.selected_model_index)
            .unwrap_or(&"n")
            .to_string()
    }

    // 获取当前选择的跟踪器
    pub fn get_selected_tracker(&self) -> String {
        static SHORT_TRACKERS: [&str; 3] = ["deepsort", "bytetrack", "none"];
        SHORT_TRACKERS
            .get(self.selected_tracker_index)
            .unwrap_or(&"none")
            .to_string()
    }

    // 获取姿态估计状态
    pub fn is_pose_enabled(&self) -> bool {
        self.pose_enabled
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
            let base_scale_x = screen_width() / texture.width();
            let base_scale_y = screen_height() / texture.height();

            // 应用缩放
            let scale_x = base_scale_x * self.zoom_scale;
            let scale_y = base_scale_y * self.zoom_scale;

            // 计算缩放后的尺寸
            let scaled_width = texture.width() * scale_x;
            let scaled_height = texture.height() * scale_y;

            // 计算居中位置 + 平移偏移
            let center_x = (screen_width() - scaled_width) / 2.0 + self.pan_offset.x;
            let center_y = (screen_height() - scaled_height) / 2.0 + self.pan_offset.y;

            draw_texture_ex(
                texture,
                center_x,
                center_y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(scaled_width, scaled_height)),
                    ..Default::default()
                },
            );

            // 绘制检测框
            if let Some(detection_result) = &self.last_detection {
                for bbox in &detection_result.bboxes {
                    let x1 = bbox.x1 * scale_x + center_x;
                    let y1 = bbox.y1 * scale_y + center_y;
                    let x2 = bbox.x2 * scale_x + center_x;
                    let y2 = bbox.y2 * scale_y + center_y;

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
                            draw_circle(*x * scale_x + center_x, *y * scale_y + center_y, 4.0, RED);
                        }
                    }

                    // 绘制骨架连接
                    for (idx1, idx2) in &SKELETON {
                        if *idx1 < keypoints.points.len() && *idx2 < keypoints.points.len() {
                            let (x1, y1, c1) = keypoints.points[*idx1];
                            let (x2, y2, c2) = keypoints.points[*idx2];
                            if c1 > 0.3 && c2 > 0.3 {
                                draw_line(
                                    x1 * scale_x + center_x,
                                    y1 * scale_y + center_y,
                                    x2 * scale_x + center_x,
                                    y2 * scale_y + center_y,
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

        // 显示缩放提示
        if self.zoom_scale != 1.0 {
            let zoom_text = format!("缩放: {:.1}x (按R键重置)", self.zoom_scale);
            draw_text(&zoom_text, 10.0, screen_height() - 10.0, 20.0, WHITE);
        }
    }

    pub fn draw_egui(&mut self) {
        egui_macroquad::ui(|egui_ctx| {
            // 1. 主控制面板 (合并所有配置)
            if self.show_control_panel {
                egui::Window::new("控制面板")
                    .default_pos(egui::pos2(10.0, 10.0))
                    .default_size(egui::vec2(350.0, 600.0))
                    .resizable(true)
                    .show(egui_ctx, |ui| {
                        // --- 状态监控 ---
                        egui::CollapsingHeader::new("📊 系统状态")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("渲染 FPS:");
                                    ui.colored_label(
                                        egui::Color32::GREEN,
                                        format!("{:.1}", self.render_fps),
                                    );
                                    ui.label("| 解码 FPS:");
                                    ui.colored_label(
                                        egui::Color32::CYAN,
                                        format!("{:.1}", self.decode_fps),
                                    );
                                    ui.label("| 检测 FPS:");
                                    ui.colored_label(
                                        egui::Color32::YELLOW,
                                        format!("{:.1}", self.detect_fps),
                                    );
                                });
                                ui.label(format!("当前模型: {}", self.detect_model_name));
                            });

                        ui.separator();

                        // --- 输入源配置 ---
                        egui::CollapsingHeader::new("🎥 输入源配置")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.radio_value(&mut self.input_source_type, 0, "RTSP");
                                    if ui
                                        .radio_value(&mut self.input_source_type, 1, "摄像头")
                                        .changed()
                                    {
                                        if !self.devices_loaded {
                                            self.video_devices = get_video_devices();
                                            self.devices_loaded = true;
                                            if !self.video_devices.is_empty() {
                                                self.selected_device_index = 0;
                                            }
                                        }
                                    }
                                    ui.radio_value(&mut self.input_source_type, 2, "桌面");
                                });

                                if self.input_source_type == 0 {
                                    ui.label("RTSP 地址:");

                                    // 历史记录下拉框
                                    egui::ComboBox::from_id_salt("rtsp_history")
                                        .selected_text("选择历史记录...")
                                        .show_ui(ui, |ui| {
                                            for url in &self.rtsp_history {
                                                if ui
                                                    .selectable_label(self.rtsp_url == *url, url)
                                                    .clicked()
                                                {
                                                    self.rtsp_url = url.clone();
                                                }
                                            }
                                        });

                                    // 宽输入框
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.rtsp_url)
                                            .desired_width(ui.available_width()),
                                    );
                                } else if self.input_source_type == 1 {
                                    if !self.devices_loaded {
                                        if ui.button("🔄 刷新设备列表").clicked() {
                                            self.video_devices = get_video_devices();
                                            self.devices_loaded = true;
                                            if !self.video_devices.is_empty() {
                                                self.selected_device_index = 0;
                                            }
                                        }
                                    } else {
                                        if self.video_devices.is_empty() {
                                            ui.label("未找到设备");
                                            if ui.button("🔄 重试").clicked() {
                                                self.video_devices = get_video_devices();
                                            }
                                        } else {
                                            egui::ComboBox::from_label("选择设备")
                                                .selected_text(
                                                    self.video_devices
                                                        .get(self.selected_device_index)
                                                        .map(|d| d.name.as_str())
                                                        .unwrap_or("未知"),
                                                )
                                                .show_ui(ui, |ui| {
                                                    for (idx, device) in
                                                        self.video_devices.iter().enumerate()
                                                    {
                                                        ui.selectable_value(
                                                            &mut self.selected_device_index,
                                                            idx,
                                                            &device.name,
                                                        );
                                                    }
                                                });
                                        }
                                    }
                                } else {
                                    ui.label("桌面捕获 (gdigrab)");
                                }

                                if ui.button("🔄 切换输入源").clicked() {
                                    let new_source = if self.input_source_type == 0 {
                                        // 自动保存 RTSP 地址到历史记录
                                        if !self.rtsp_url.is_empty()
                                            && !self.rtsp_history.contains(&self.rtsp_url)
                                        {
                                            self.rtsp_history.push(self.rtsp_url.clone());
                                            // 限制历史记录数量
                                            if self.rtsp_history.len() > 10 {
                                                self.rtsp_history.remove(0);
                                            }
                                            // 保存到文件
                                            let content = self.rtsp_history.join("\n");
                                            let _ = std::fs::write("rtsp_history.txt", content);
                                        }
                                        InputSource::Rtsp(self.rtsp_url.clone())
                                    } else if self.input_source_type == 1 {
                                        let (device_index, device_name) = if self.devices_loaded
                                            && !self.video_devices.is_empty()
                                        {
                                            let dev =
                                                &self.video_devices[self.selected_device_index];
                                            (dev.index, dev.name.clone())
                                        } else {
                                            (self.camera_id as usize, format!("{}", self.camera_id))
                                        };
                                        InputSource::Camera(device_index, device_name)
                                    } else {
                                        InputSource::Desktop
                                    };
                                    switch_decoder_source(new_source);
                                }
                            });

                        ui.separator();

                        // --- 模型与参数 ---
                        egui::CollapsingHeader::new("⚙️ 模型与参数")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("检测模型:");
                                egui::ComboBox::from_label("模型")
                                    .selected_text(
                                        MODELS
                                            .get(self.selected_model_index)
                                            .copied()
                                            .unwrap_or("yolov8n"),
                                    )
                                    .show_ui(ui, |ui| {
                                        for (idx, model) in MODELS.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut self.selected_model_index,
                                                idx,
                                                *model,
                                            );
                                        }
                                    });

                                ui.label("跟踪算法:");
                                egui::ComboBox::from_label("跟踪")
                                    .selected_text(
                                        TRACKERS
                                            .get(self.selected_tracker_index)
                                            .copied()
                                            .unwrap_or("无"),
                                    )
                                    .show_ui(ui, |ui| {
                                        for (idx, tracker) in TRACKERS.iter().enumerate() {
                                            ui.selectable_value(
                                                &mut self.selected_tracker_index,
                                                idx,
                                                *tracker,
                                            );
                                        }
                                    });

                                ui.checkbox(&mut self.pose_enabled, "启用姿态估计");

                                ui.separator();
                                ui.label("阈值设置:");
                                let mut params_changed = false;
                                if ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.confidence_threshold,
                                            0.0..=1.0,
                                        )
                                        .text("置信度"),
                                    )
                                    .changed()
                                {
                                    params_changed = true;
                                }
                                if ui
                                    .add(
                                        egui::Slider::new(&mut self.iou_threshold, 0.0..=1.0)
                                            .text("IOU"),
                                    )
                                    .changed()
                                {
                                    params_changed = true;
                                }

                                if params_changed {
                                    if let Some(tx) = &self.config_tx {
                                        let _ = tx.send(ConfigMessage {
                                            conf_threshold: self.confidence_threshold,
                                            iou_threshold: self.iou_threshold,
                                        });
                                    }
                                }
                            });

                        ui.separator();

                        // --- 视图控制 ---
                        egui::CollapsingHeader::new("👁️ 视图控制")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.preview_visible, "显示 Resize 预览窗口");
                                if ui.button("重置缩放 (R)").clicked() {
                                    self.zoom_scale = 1.0;
                                    self.pan_offset = Vec2::ZERO;
                                }
                            });
                    });
            }

            // 2. Resize 预览窗口 (无边框, 可拖拽)
            if self.preview_visible {
                if let Some(detection_result) = &self.last_detection {
                    egui::Window::new("Resize Preview")
                        .title_bar(false) // 无标题栏
                        .frame(egui::Frame::window(&egui_ctx.style()).inner_margin(0.0)) // 紧凑边框
                        .current_pos(self.resize_window_pos)
                        .resizable(false)
                        .show(egui_ctx, |ui| {
                            // 整个区域作为拖拽手柄
                            let size = detection_result.resized_size as f32;
                            let (rect, response) =
                                ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::drag());

                            if response.dragged() {
                                self.resize_window_pos += response.drag_delta();
                            }

                            // 绘制图像
                            if let Some(resized_data) = &detection_result.resized_image {
                                if resized_data.len()
                                    == (detection_result.resized_size
                                        * detection_result.resized_size
                                        * 4) as usize
                                {
                                    let img_size = detection_result.resized_size as usize;
                                    let image = egui::ColorImage::from_rgba_unmultiplied(
                                        [img_size, img_size],
                                        resized_data,
                                    );

                                    let texture = egui_ctx.load_texture(
                                        "resized_preview",
                                        image,
                                        egui::TextureOptions::LINEAR,
                                    );

                                    // 在分配的矩形中绘制
                                    let painter = ui.painter();
                                    painter.image(
                                        texture.id(),
                                        rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                    // 绘制边框提示可拖拽
                                    painter.rect_stroke(
                                        rect,
                                        0.0,
                                        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(50)),
                                        egui::StrokeKind::Middle,
                                    );
                                } else {
                                    ui.painter().text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        "数据错误",
                                        egui::FontId::default(),
                                        egui::Color32::RED,
                                    );
                                }
                            } else {
                                ui.painter().text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    "无图像数据",
                                    egui::FontId::default(),
                                    egui::Color32::WHITE,
                                );
                            }
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
        if is_key_pressed(KeyCode::Tab) {
            self.show_control_panel = !self.show_control_panel;
        }
        // 移除旧的快捷键
        // if is_key_pressed(KeyCode::P) { ... }
        // if is_key_pressed(KeyCode::I) { ... }
        // if is_key_pressed(KeyCode::M) { ... }

        // 鼠标滚轮缩放
        let mouse_wheel = mouse_wheel();
        if mouse_wheel.1 != 0.0 {
            // 使用指数缩放 (更平滑自然)
            let zoom_factor = 1.1f32;
            let scale_mult = if mouse_wheel.1 > 0.0 {
                zoom_factor
            } else {
                1.0 / zoom_factor
            };

            let new_scale = (self.zoom_scale * scale_mult).clamp(0.1, 20.0);

            // 计算实际的缩放比例 (因为可能被clamp限制)
            let ratio = new_scale / self.zoom_scale;

            // 以鼠标位置为中心缩放
            // 核心公式: Pan_new = Pan_old * Ratio + Mouse_rel * (1 - Ratio)
            // 其中 Mouse_rel 是鼠标相对于屏幕中心的坐标
            let mouse_pos = mouse_position();
            let screen_center = Vec2::new(screen_width() / 2.0, screen_height() / 2.0);
            let mouse_rel = Vec2::new(mouse_pos.0, mouse_pos.1) - screen_center;

            self.pan_offset = self.pan_offset * ratio + mouse_rel * (1.0 - ratio);
            self.zoom_scale = new_scale;
        }

        // 重置缩放 (按R键)
        if is_key_pressed(KeyCode::R) {
            self.zoom_scale = 1.0;
            self.pan_offset = Vec2::ZERO;
        }

        // 鼠标中键拖动
        if is_mouse_button_down(MouseButton::Middle) {
            let mouse_pos = mouse_position();
            let current_pos = Vec2::new(mouse_pos.0, mouse_pos.1);

            if self.is_panning {
                let delta = current_pos - self.last_mouse_pos;
                self.pan_offset += delta;
            } else {
                self.is_panning = true;
            }
            self.last_mouse_pos = current_pos;
        } else {
            self.is_panning = false;
        }
    }
}
