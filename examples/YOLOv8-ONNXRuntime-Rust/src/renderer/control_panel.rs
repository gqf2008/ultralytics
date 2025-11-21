use crate::detection::types::ControlMessage;
use crate::input::decoder::DecoderPreference;
use crate::input::{get_video_devices, switch_decoder_source, InputSource, VideoDevice};
use crossbeam_channel::Sender;
use egui_macroquad::egui::{self, TextureHandle};
use macroquad::math::Vec2;
use phf::phf_map;

/// 复制文本到系统剪贴板 (Windows 专用，使用 clipboard-win)
#[cfg(windows)]
fn copy_to_clipboard(_ui: &egui::Ui, text: &str) {
    use clipboard_win::{formats, set_clipboard};

    println!("📋 复制到剪贴板: {}", text);

    match set_clipboard(formats::Unicode, text) {
        Ok(_) => {
            println!("✅ 已成功复制到系统剪贴板!");
            println!("💡 现在可以在 VS Code 等应用中按 Ctrl+V 粘贴");
        }
        Err(e) => {
            eprintln!("❌ 复制到剪贴板失败: {:?}", e);
        }
    }
}

/// 复制文本到系统剪贴板 (非 Windows 平台)
#[cfg(not(windows))]
fn copy_to_clipboard(ui: &egui::Ui, text: &str) {
    println!("📋 复制到剪贴板: {}", text);
    ui.ctx().copy_text(text.to_string());
    println!("✅ 已复制!");
}

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

/// 控制面板状态
pub struct ControlPanel {
    // 系统配置信息
    pub detect_model_name: String,
    pub tracker_name: String,
    pub detect_fps: f64,
    pub decode_fps: f64,
    pub render_fps: f64,

    // egui 参数调整
    pub confidence_threshold: f32,
    pub iou_threshold: f32,

    // 输入源配置界面
    pub input_source_type: usize, // 0=RTSP, 1=摄像头, 2=桌面捕获
    pub rtsp_url: String,
    pub rtsp_history: Vec<String>, // RTSP 历史记录

    // 设备列表
    pub video_devices: Vec<VideoDevice>,
    pub selected_device_index: usize,
    pub devices_loaded: bool,

    // 模型配置
    pub selected_model_index: usize,
    pub selected_tracker_index: usize,
    pub pose_enabled: bool,
    pub detection_enabled: bool,
    config_tx: Option<Sender<ControlMessage>>,
    // 视图控制
    pub zoom_scale: f32,
    pub pan_offset: macroquad::prelude::Vec2,

    // 背景纹理
    pub panel_bg_egui: Option<TextureHandle>,
    pub panel_bg_size: Option<(usize, usize)>,
}

impl ControlPanel {
    pub fn new(detect_model: String, tracker: String) -> Self {
        let mut bg = None;
        let mut bg_size = None;
        if let Ok(bytes) = std::fs::read("assets/images/panel_bg.jpg") {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let rgba = img.to_rgba8();
                let width = rgba.width() as usize;
                let height = rgba.height() as usize;
                bg_size = Some((width, height));
                let color_image = egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba);
                egui_macroquad::cfg(|egui_ctx| {
                    let texture = egui_ctx.load_texture(
                        "panel_bg",
                        color_image,
                        egui::TextureOptions::LINEAR,
                    );

                    bg = Some(texture);
                });
            }
        }

        Self {
            detect_model_name: detect_model.clone(),
            tracker_name: tracker.clone(),
            detect_fps: 0.0,
            decode_fps: 0.0,
            render_fps: 0.0,
            confidence_threshold: 0.5,
            iou_threshold: 0.45,
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
            video_devices: Vec::new(),
            selected_device_index: 0,
            devices_loaded: false,
            selected_model_index: *MODEL_INDICES.get(detect_model.as_str()).unwrap_or(&0),
            selected_tracker_index: *TRACKER_INDICES
                .get(tracker.to_lowercase().as_str())
                .unwrap_or(&2),
            pose_enabled: false,
            detection_enabled: true,
            zoom_scale: 1.0,
            pan_offset: macroquad::prelude::Vec2::ZERO,
            panel_bg_egui: bg,
            panel_bg_size: bg_size,
            config_tx: None,
        }
    }

    /// 保存 RTSP 历史记录到文件
    fn save_rtsp_history(&self) {
        if let Err(e) = std::fs::write("rtsp_history.txt", self.rtsp_history.join("\n")) {
            eprintln!("⚠️ 保存 RTSP 历史记录失败: {}", e);
        }
    }

    pub fn set_config_chan(&mut self, tx: Sender<ControlMessage>) {
        self.config_tx = Some(tx);
    }
    /// 添加 RTSP 地址到历史记录并保存
    fn add_rtsp_to_history(&mut self, url: String) {
        if !self.rtsp_history.contains(&url) {
            self.rtsp_history.insert(0, url.clone());
            // 限制历史记录数量
            if self.rtsp_history.len() > 10 {
                self.rtsp_history.truncate(10);
            }
            println!("📝 新增 RTSP 历史记录: {}", url);
            self.save_rtsp_history();
        }
    }

    fn resolve_model_path(&self, model_name: &str) -> String {
        match model_name {
            "yolo-fastestv2" => "models/yolo-fastestv2-opt.onnx".to_string(),
            "yolo-fastest-xl" => "models/yolo-fastest-1.1.onnx".to_string(),
            "nanodet" => "models/nanodet-plus-m_320.onnx".to_string(),
            "nanodet-plus" => "models/nanodet-plus-m_416.onnx".to_string(),
            name if name.ends_with("-int8") => format!("models/{}.onnx", name.replace("-", "_")),
            _ => format!("models/{}.onnx", model_name),
        }
    }

    fn set_style(&mut self, ctx: &egui::Context) {
        // --- 自定义 UI 样式 (透明背景) ---
        let mut visuals = egui::Visuals::dark();

        // 窗口样式 - 透明背景
        visuals.window_fill = egui::Color32::TRANSPARENT;
        visuals.window_stroke = egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, 30),
        );

        // 面板和区域背景 - 透明
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        visuals.extreme_bg_color = egui::Color32::TRANSPARENT;

        // 非交互控件（标签、文本等）- 透明背景，无圆角
        visuals.widgets.noninteractive.bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.noninteractive.weak_bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::NONE;
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 210, 220));
        visuals.widgets.noninteractive.corner_radius = 0.0.into(); // 无圆角

        // 未激活控件（按钮、输入框等）- 透明背景，无圆角
        visuals.widgets.inactive.bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.inactive.bg_stroke = egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_premultiplied(180, 190, 200, 80),
        );
        visuals.widgets.inactive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(180, 190, 200));
        visuals.widgets.inactive.corner_radius = 0.0.into(); // 无圆角

        // 悬停控件 - 透明背景+边框，无圆角
        visuals.widgets.hovered.bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.hovered.weak_bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(
            1.5,
            egui::Color32::from_rgba_premultiplied(180, 190, 200, 150),
        );
        visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
        visuals.widgets.hovered.corner_radius = 0.0.into(); // 无圆角

        // 激活/点击控件 - 透明背景+加粗边框，无圆角
        visuals.widgets.active.bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.active.weak_bg_fill = egui::Color32::TRANSPARENT;
        visuals.widgets.active.bg_stroke = egui::Stroke::new(
            2.0,
            egui::Color32::from_rgba_premultiplied(200, 210, 220, 200),
        );
        visuals.widgets.active.fg_stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
        visuals.widgets.active.corner_radius = 0.0.into(); // 无圆角

        // 选中状态 - 半透明
        visuals.selection.bg_fill = egui::Color32::from_rgba_premultiplied(100, 150, 255, 100);
        visuals.selection.stroke = egui::Stroke::new(
            1.5,
            egui::Color32::from_rgba_premultiplied(150, 200, 255, 150),
        );

        // 文本颜色
        visuals.override_text_color = Some(egui::Color32::from_rgb(230, 240, 250));

        ctx.set_visuals(visuals);
    }

    pub fn show(&mut self, ctx: &egui::Context, open: &mut bool) {
        if !*open {
            return;
        }
        self.set_style(ctx);

        // 根据背景图像尺寸确定窗口大小
        let window_size = if let Some((width, height)) = self.panel_bg_size {
            egui::vec2(width as f32, height as f32)
        } else {
            egui::vec2(350.0, 600.0) // 默认尺寸
        };

        egui::Window::new("🎯 控制面板")
            .default_pos(egui::pos2(10.0, 10.0))
            .default_size(window_size)
            .resizable(true)
            .frame(egui::Frame::NONE)
            .title_bar(true)
            .show(ctx, |ui| {
                // 先绘制背景图像到最底层,完全填充窗口
                if let Some(tex) = self.panel_bg_egui.as_ref() {
                    let painter = ui.painter();
                    let rect = ui.available_rect_before_wrap();
                    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    let tint = egui::Color32::from_rgba_premultiplied(255, 255, 255, 180);
                    painter.image(tex.id(), rect, uv, tint);
                }

                // 使用ScrollArea包裹UI内容,允许窗口垂直调整大小
                let actions = egui::ScrollArea::vertical()
                    .auto_shrink([false; 2]) // 不自动收缩,允许手动调整窗口大小
                    .show(ui, |ui| self.ui(ui))
                    .inner;

                // 处理控制面板的操作
                if actions.reset_zoom {
                    self.zoom_scale = 1.0;
                    self.pan_offset = Vec2::ZERO;
                }

                // 处理启动解码器的操作
                if let Some(input_source) = actions.start_decoder {
                    println!("🚀 从控制面板启动解码器: {:?}", input_source);
                    switch_decoder_source(input_source, DecoderPreference::Software);
                }
            });
    }
    /// 绘制控制面板UI
    fn ui(
        &mut self,
        ui: &mut egui::Ui,
        //  config_tx: &Option<Sender<ConfigMessage>>,
    ) -> ControlPanelActions {
        let mut actions = ControlPanelActions::default();

        ui.style_mut().visuals.collapsing_header_frame = false;

        // --- 状态监控 ---
        egui::CollapsingHeader::new("📊 系统状态")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("渲染 FPS:");
                    ui.colored_label(egui::Color32::GREEN, format!("{:.1}", self.render_fps));
                    ui.label("| 解码 FPS:");
                    ui.colored_label(egui::Color32::CYAN, format!("{:.1}", self.decode_fps));
                    ui.label("| 检测 FPS:");
                    ui.colored_label(egui::Color32::YELLOW, format!("{:.1}", self.detect_fps));
                });
                ui.label(format!("当前模型: {}", self.detect_model_name));
            });

        ui.separator();

        // --- 输入源配置 ---
        egui::CollapsingHeader::new("🎥 输入源配置")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // 切换到RTSP
                    if ui
                        .radio_value(&mut self.input_source_type, 0, "RTSP")
                        .changed()
                    {
                        // 立即启动RTSP解码
                        if !self.rtsp_url.trim().is_empty() {
                            actions.start_decoder = Some(InputSource::Rtsp(self.rtsp_url.clone()));
                        }
                    }

                    // 切换到摄像头
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
                        // 立即启动摄像头解码
                        if !self.video_devices.is_empty() {
                            if let Some(device) = self.video_devices.get(self.selected_device_index)
                            {
                                actions.start_decoder =
                                    Some(InputSource::Camera(device.index, device.name.clone()));
                            }
                        }
                    }

                    // 切换到桌面捕获
                    if ui
                        .radio_value(&mut self.input_source_type, 2, "桌面")
                        .changed()
                    {
                        // 立即启动桌面捕获
                        actions.start_decoder = Some(InputSource::Desktop);
                    }
                });

                if self.input_source_type == 0 {
                    ui.label("RTSP 地址:");

                    // 历史记录下拉框 - 选择后自动播放并复制到剪贴板
                    let mut url_to_copy: Option<String> = None;

                    let _combo_response = egui::ComboBox::from_id_salt("rtsp_history")
                        .selected_text("选择历史记录...")
                        .show_ui(ui, |ui| {
                            // 下拉菜单打开时重新加载历史记录
                            if let Ok(content) = std::fs::read_to_string("rtsp_history.txt") {
                                let lines: Vec<String> = content
                                    .lines()
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .collect();
                                if !lines.is_empty() {
                                    self.rtsp_history = lines;
                                }
                            }

                            for url in &self.rtsp_history.clone() {
                                let response = ui.selectable_label(self.rtsp_url == *url, url);

                                // 左键点击: 填充到输入框并自动播放
                                if response.clicked() {
                                    self.rtsp_url = url.clone();
                                    // 自动启动播放
                                    switch_decoder_source(
                                        InputSource::Rtsp(self.rtsp_url.clone()),
                                        DecoderPreference::Software,
                                    );

                                    // 移到历史记录最前面(更新访问时间)
                                    if let Some(pos) =
                                        self.rtsp_history.iter().position(|x| x == url)
                                    {
                                        if pos > 0 {
                                            let moved_url = self.rtsp_history.remove(pos);
                                            self.rtsp_history.insert(0, moved_url);
                                            self.save_rtsp_history();
                                        }
                                    }
                                }

                                // 右键点击: 标记需要复制
                                if response.secondary_clicked() {
                                    url_to_copy = Some(url.clone());
                                }

                                // 悬停提示
                                response.on_hover_text("左键:填充并播放 | 右键:仅复制到剪贴板");
                            }
                        });

                    // 同时写入 egui 和系统剪贴板
                    if let Some(url) = url_to_copy {
                        copy_to_clipboard(ui, &url);
                    }

                    // RTSP 输入框 - 支持回车键快速启动
                    let text_response = ui.add(
                        egui::TextEdit::singleline(&mut self.rtsp_url)
                            .desired_width(ui.available_width())
                            .hint_text("输入 RTSP 地址后按回车..."),
                    );

                    // 检测回车键 - 自动保存并启动播放
                    if text_response.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        && !self.rtsp_url.trim().is_empty()
                    {
                        let url = self.rtsp_url.trim().to_string();

                        // 保存到历史记录并写入文件
                        self.add_rtsp_to_history(url.clone());

                        // 更新输入框为修剪后的地址
                        self.rtsp_url = url.clone();

                        // 触发播放
                        switch_decoder_source(
                            InputSource::Rtsp(url.clone()),
                            DecoderPreference::Software,
                        );
                        println!("🚀 回车触发播放: {}", url);
                    }
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
                                    for (idx, device) in self.video_devices.iter().enumerate() {
                                        if ui
                                            .selectable_value(
                                                &mut self.selected_device_index,
                                                idx,
                                                &device.name,
                                            )
                                            .clicked()
                                        {
                                            // 选择设备后立即启动解码
                                            actions.start_decoder = Some(InputSource::Camera(
                                                device.index,
                                                device.name.clone(),
                                            ));
                                        }
                                    }
                                });
                        }
                    }
                } else {
                    ui.label("桌面捕获 (gdigrab)");
                }
            });

        ui.separator();

        // --- 模型与参数 ---
        egui::CollapsingHeader::new("⚙️ 模型与参数")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("检测模型:");
                let mut selected_model = self.selected_model_index;
                egui::ComboBox::from_label("模型")
                    .selected_text(
                        MODELS
                            .get(self.selected_model_index)
                            .copied()
                            .unwrap_or("yolov8n"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, model) in MODELS.iter().enumerate() {
                            ui.selectable_value(&mut selected_model, idx, *model);
                        }
                    });

                if selected_model != self.selected_model_index {
                    self.selected_model_index = selected_model;
                    let model_name = MODELS[selected_model];
                    self.detect_model_name = model_name.to_string();
                    let model_path = self.resolve_model_path(model_name);
                    if let Some(tx) = &self.config_tx {
                        let _ = tx.try_send(ControlMessage::SwitchModel(model_path));
                    }
                }

                ui.label("跟踪算法:");
                let mut selected_tracker = self.selected_tracker_index;
                egui::ComboBox::from_label("跟踪")
                    .selected_text(
                        TRACKERS
                            .get(self.selected_tracker_index)
                            .copied()
                            .unwrap_or("无"),
                    )
                    .show_ui(ui, |ui| {
                        for (idx, tracker) in TRACKERS.iter().enumerate() {
                            ui.selectable_value(&mut selected_tracker, idx, *tracker);
                        }
                    });

                if selected_tracker != self.selected_tracker_index {
                    self.selected_tracker_index = selected_tracker;
                    let tracker_name = TRACKERS[selected_tracker];
                    self.tracker_name = tracker_name.to_string();
                    if let Some(tx) = &self.config_tx {
                        let _ =
                            tx.try_send(ControlMessage::SwitchTracker(tracker_name.to_string()));
                    }
                }

                if ui
                    .checkbox(&mut self.pose_enabled, "启用姿态估计")
                    .changed()
                {
                    if let Some(tx) = &self.config_tx {
                        let _ = tx.try_send(ControlMessage::TogglePose(self.pose_enabled));
                    }
                }

                if ui
                    .checkbox(&mut self.detection_enabled, "启用目标检测")
                    .changed()
                {
                    if let Some(tx) = &self.config_tx {
                        let _ =
                            tx.try_send(ControlMessage::ToggleDetection(self.detection_enabled));
                    }
                }

                ui.separator();
                ui.label("阈值设置:");
                let mut params_changed = false;
                if ui
                    .add(
                        egui::Slider::new(&mut self.confidence_threshold, 0.0..=1.0).text("置信度"),
                    )
                    .changed()
                {
                    params_changed = true;
                }
                if ui
                    .add(egui::Slider::new(&mut self.iou_threshold, 0.0..=1.0).text("IOU"))
                    .changed()
                {
                    params_changed = true;
                }

                if params_changed {
                    if let Some(tx) = &self.config_tx {
                        // 使用 try_send 避免阻塞UI线程（当Detector忙碌时）
                        let _ = tx.try_send(ControlMessage::UpdateParams {
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
                if ui.button("重置缩放 (R)").clicked() {
                    actions.reset_zoom = true;
                }
            });

        actions
    }
}

/// 控制面板操作返回值
#[derive(Default)]
pub struct ControlPanelActions {
    pub reset_zoom: bool,
    pub start_decoder: Option<InputSource>,
}
