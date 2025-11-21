mod control_panel;

use crate::detection::detector::DetectionResult;
use crate::detection::types::{ConfigMessage, DecodedFrame};
use crate::input::decoder::DecoderPreference;
use crate::input::switch_decoder_source;
use crate::xbus::{self, Subscription};
use crate::SKELETON;
use control_panel::ControlPanel;
use crossbeam_channel::{Receiver, Sender};
use egui_macroquad::egui;
use macroquad::prelude::*;
use std::time::Instant;

// 引入 image crate 用于加载背景图
use image;

pub struct Renderer {
    _frame_sub: Subscription,
    _result_sub: Subscription,
    render_frame_buffer: Receiver<RenderFrame>,

    last_frame: Option<Texture2D>,
    last_detection: Option<DetectionResult>,
    render_count: u64,
    render_last: Instant,
    show_control_panel: bool,

    // 视频帧率统计
    video_count: u64,
    video_last: Instant,

    // 画面缩放
    is_panning: bool,
    last_mouse_pos: Vec2,

    // 窗口状态
    is_mouse_over_ui: bool,

    // 背景纹理
    background_texture: Option<Texture2D>,

    // 中文字体
    chinese_font: Option<Font>,

    // 检测器延迟启动参数
    detector_model_path: Option<String>,
    detector_inf_size: Option<u32>,
    detector_tracker: Option<String>,
    detector_pose_enabled: Option<bool>,
    detector_started: bool,

    // 控制面板(独立模块)
    control_panel: ControlPanel,
}

enum RenderFrame {
    Video(DecodedFrame),
    Detection(DetectionResult),
}

impl Renderer {
    pub fn new(detect_model: String, _pose_model: String, tracker: String) -> Self {
        println!("渲染器启动");
        // 进一步减小队列长度以降低内存占用 (5 -> 2)
        let (tx, rx) = crossbeam_channel::bounded(2);

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

        // 加载背景图片
        let background_texture = if let Ok(bytes) = std::fs::read("assets/images/background.jpg") {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let rgba = img.to_rgba8();
                Some(Texture2D::from_rgba8(
                    rgba.width() as u16,
                    rgba.height() as u16,
                    &rgba,
                ))
            } else {
                println!("⚠️ 背景图片解码失败");
                None
            }
        } else {
            println!("⚠️ 未找到背景图片: assets/images/background.jpg");
            None
        };
        let mut control_panel = ControlPanel::new(detect_model, tracker);
        // 加载控制面板背景纹理
        if let Ok(bytes) = std::fs::read("assets/images/panel_bg.jpg") {
            if let Ok(img) = image::load_from_memory(&bytes) {
                let rgba = img.to_rgba8();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [rgba.width() as usize, rgba.height() as usize],
                    &rgba,
                );
                egui_macroquad::cfg(|egui_ctx| {
                    let texture = egui_ctx.load_texture(
                        "panel_bg",
                        color_image,
                        egui::TextureOptions::LINEAR,
                    );
                    control_panel.register_background_texture(texture);
                });
            }
        }

        // 加载中文字体
        let chinese_font = if let Ok(bytes) = std::fs::read("assets/font/msyh.ttc") {
            match load_ttf_font_from_bytes(&bytes) {
                Ok(font) => {
                    println!("✅ 中文字体加载成功");
                    Some(font)
                }
                Err(e) => {
                    println!("⚠️ 中文字体加载失败: {}", e);
                    None
                }
            }
        } else {
            println!("⚠️ 未找到中文字体文件: assets/font/msyh.ttc");
            None
        };

        Self {
            render_frame_buffer: rx,
            last_frame: None,
            last_detection: None,
            _frame_sub: frame_sub,
            _result_sub: result_sub,
            render_count: 0,
            render_last: Instant::now(),
            show_control_panel: true,
            video_count: 0,
            video_last: Instant::now(),
            is_panning: false,
            last_mouse_pos: Vec2::ZERO,
            is_mouse_over_ui: false,
            background_texture,

            chinese_font,
            detector_model_path: None,
            detector_inf_size: None,
            detector_tracker: None,
            detector_pose_enabled: None,
            detector_started: false,
            control_panel,
        }
    }

    pub fn set_config_sender(&mut self, tx: Sender<ConfigMessage>) {
        self.control_panel.set_config_chan(tx);
    }

    /// 保存检测器启动参数(延迟启动)
    pub fn set_detector_params(
        &mut self,
        model_path: String,
        inf_size: u32,
        tracker: String,
        pose_enabled: bool,
    ) {
        self.detector_model_path = Some(model_path);
        self.detector_inf_size = Some(inf_size);
        self.detector_tracker = Some(tracker);
        self.detector_pose_enabled = Some(pose_enabled);
    }

    /// 启动检测器线程(首次启动解码器时调用)
    fn start_detector_if_needed(&mut self) {
        if self.detector_started {
            return; // 已启动,跳过
        }

        // 检查是否有保存的参数
        if let (Some(model_path), Some(inf_size), Some(tracker), Some(pose_enabled)) = (
            self.detector_model_path.clone(),
            self.detector_inf_size,
            self.detector_tracker.clone(),
            self.detector_pose_enabled,
        ) {
            println!("🔍 检测模块启动");

            // 创建配置通道
            let (config_tx, config_rx) = crossbeam_channel::bounded(5);

            // 启动检测线程
            std::thread::spawn(move || {
                use crate::detection;
                let mut det = detection::Detector::new(model_path, inf_size, tracker, pose_enabled);
                det.set_config_receiver(config_rx);
                det.run();
            });

            // 保存配置发送器
            self.control_panel.set_config_chan(config_tx.clone());

            // 发送初始参数
            if let Err(e) = config_tx.try_send(ConfigMessage::UpdateParams {
                conf_threshold: self.control_panel.confidence_threshold,
                iou_threshold: self.control_panel.iou_threshold,
            }) {
                eprintln!("⚠️ 发送初始参数失败: {}", e);
            }

            self.detector_started = true;
        }
    }

    pub fn update(&mut self) {
        // 首次收到视频帧时启动检测器(在处理帧之前检查)
        let should_start_detector = !self.detector_started;

        // 处理帧缓冲 - 统计所有接收到的帧以计算FPS，但只渲染最新一帧
        let mut latest_video_frame = None;
        let mut latest_detection_result = None;
        let mut video_frames_received = 0;
        let mut has_video_frame = false;

        for frame in self.render_frame_buffer.try_iter() {
            match frame {
                RenderFrame::Video(decoded_frame) => {
                    has_video_frame = true;
                    latest_video_frame = Some(decoded_frame);
                    video_frames_received += 1;
                }
                RenderFrame::Detection(detection_result) => {
                    latest_detection_result = Some(detection_result);
                }
            }
        }

        // 收到第一帧视频时启动检测器
        if should_start_detector && has_video_frame {
            self.start_detector_if_needed();
        }

        // 更新解码FPS统计
        self.video_count += video_frames_received;
        let now = Instant::now();
        if now.duration_since(self.video_last).as_secs() >= 1 {
            self.control_panel.decode_fps =
                self.video_count as f64 / now.duration_since(self.video_last).as_secs_f64();
            self.video_count = 0;
            self.video_last = now;
        }

        // 更新视频纹理
        if let Some(decoded_frame) = latest_video_frame {
            // 释放旧纹理（macroquad会自动管理）
            // 只在分辨率变化时重建纹理，否则更新像素数据
            let needs_rebuild = if let Some(ref tex) = self.last_frame {
                tex.width() != decoded_frame.width as f32
                    || tex.height() != decoded_frame.height as f32
            } else {
                true
            };

            if needs_rebuild {
                let texture = Texture2D::from_rgba8(
                    decoded_frame.width as u16,
                    decoded_frame.height as u16,
                    &decoded_frame.rgba_data,
                );
                texture.set_filter(FilterMode::Linear);
                self.last_frame = Some(texture);
            } else if let Some(ref tex) = self.last_frame {
                // 更新现有纹理的像素数据（避免重新分配GPU内存）
                let img = Image {
                    bytes: decoded_frame.rgba_data.to_vec(),
                    width: decoded_frame.width as u16,
                    height: decoded_frame.height as u16,
                };
                tex.update(&img);
            }
        }

        // 更新检测结果
        if let Some(result) = latest_detection_result {
            self.last_detection = Some(result);
        }

        // 更新检测FPS
        if let Some(result) = &self.last_detection {
            self.control_panel.detect_fps = result.inference_fps;
        }
    }

    pub fn draw(&mut self) {
        // 先绘制背景图（如果没有视频帧）
        if self.last_frame.is_none() {
            if let Some(bg) = &self.background_texture {
                draw_texture_ex(
                    bg,
                    0.0,
                    0.0,
                    WHITE,
                    DrawTextureParams {
                        dest_size: Some(vec2(screen_width(), screen_height())),
                        ..Default::default()
                    },
                );
                // 叠加半透明遮罩
                draw_rectangle(
                    0.0,
                    0.0,
                    screen_width(),
                    screen_height(),
                    Color::new(0.0, 0.0, 0.0, 0.5),
                );
            } else {
                clear_background(Color::from_rgba(20, 20, 30, 255));
            }
        } else {
            clear_background(BLACK);
        }

        // 绘制视频帧
        if let Some(texture) = &self.last_frame {
            let base_scale_x = screen_width() / texture.width();
            let base_scale_y = screen_height() / texture.height();

            // 应用缩放
            let scale_x = base_scale_x * self.control_panel.zoom_scale;
            let scale_y = base_scale_y * self.control_panel.zoom_scale;

            // 计算缩放后的尺寸
            let scaled_width = texture.width() * scale_x;
            let scaled_height = texture.height() * scale_y;

            // 计算居中位置 + 平移偏移
            let center_x = (screen_width() - scaled_width) / 2.0 + self.control_panel.pan_offset.x;
            let center_y =
                (screen_height() - scaled_height) / 2.0 + self.control_panel.pan_offset.y;

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
            if self.control_panel.detection_enabled {
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
                                draw_circle(
                                    *x * scale_x + center_x,
                                    *y * scale_y + center_y,
                                    4.0,
                                    RED,
                                );
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
        }

        // 没有视频时显示提示文字
        if self.last_frame.is_none() {
            let text = "请在右侧控制面板选择输入源并启动";
            let font_size = 40.0;
            let text_params = TextParams {
                font: self.chinese_font.as_ref(),
                font_size: font_size as u16,
                color: WHITE,
                ..Default::default()
            };
            let text_dims = measure_text(text, self.chinese_font.as_ref(), font_size as u16, 1.0);
            draw_text_ex(
                text,
                (screen_width() - text_dims.width) / 2.0,
                (screen_height() - text_dims.height) / 2.0,
                text_params,
            );

            if self.background_texture.is_none() {
                let warning_params = TextParams {
                    font: self.chinese_font.as_ref(),
                    font_size: 24,
                    color: YELLOW,
                    ..Default::default()
                };
                draw_text_ex("⚠️ 背景图片加载失败", 10.0, 30.0, warning_params);
            }
        }

        // FPS统计
        self.render_count += 1;
        let now = Instant::now();
        if now.duration_since(self.render_last).as_secs() >= 1 {
            self.control_panel.render_fps =
                self.render_count as f64 / now.duration_since(self.render_last).as_secs_f64();
            self.render_count = 0;
            self.render_last = now;
        }

        // 显示缩放提示
        if self.control_panel.zoom_scale != 1.0 {
            let zoom_text = format!("缩放: {:.1}x (按R键重置)", self.control_panel.zoom_scale);
            let zoom_params = TextParams {
                font: self.chinese_font.as_ref(),
                font_size: 20,
                color: WHITE,
                ..Default::default()
            };
            draw_text_ex(&zoom_text, 10.0, screen_height() - 10.0, zoom_params);
        }
    }

    pub fn draw_egui(&mut self) {
        egui_macroquad::ui(|egui_ctx| {
            self.is_mouse_over_ui = egui_ctx.wants_pointer_input();
            self.control_panel
                .show(egui_ctx, &mut self.show_control_panel);
        });

        egui_macroquad::draw();
    }

    pub fn handle_input(&mut self) {
        // 键盘输入
        if is_key_pressed(KeyCode::Tab) {
            self.show_control_panel = !self.show_control_panel;
        }

        // 鼠标滚轮缩放
        let mouse_wheel = mouse_wheel();
        if mouse_wheel.1 != 0.0 && !self.is_mouse_over_ui {
            // 使用指数缩放 (更平滑自然)
            let zoom_factor = 1.1f32;
            let scale_mult = if mouse_wheel.1 > 0.0 {
                zoom_factor
            } else {
                1.0 / zoom_factor
            };

            let new_scale = (self.control_panel.zoom_scale * scale_mult).clamp(0.1, 20.0);

            // 计算实际的缩放比例 (因为可能被clamp限制)
            let ratio = new_scale / self.control_panel.zoom_scale;

            // 以鼠标位置为中心缩放
            // 核心公式: Pan_new = Pan_old * Ratio + Mouse_rel * (1 - Ratio)
            // 其中 Mouse_rel 是鼠标相对于屏幕中心的坐标
            let mouse_pos = mouse_position();
            let screen_center = Vec2::new(screen_width() / 2.0, screen_height() / 2.0);
            let mouse_rel = Vec2::new(mouse_pos.0, mouse_pos.1) - screen_center;

            self.control_panel.pan_offset =
                self.control_panel.pan_offset * ratio + mouse_rel * (1.0 - ratio);
            self.control_panel.zoom_scale = new_scale;
        }

        // 重置缩放 (按R键)
        if is_key_pressed(KeyCode::R) {
            self.control_panel.zoom_scale = 1.0;
            self.control_panel.pan_offset = Vec2::ZERO;
        }

        // 鼠标中键拖动
        if is_mouse_button_down(MouseButton::Middle) {
            let mouse_pos = mouse_position();
            let current_pos = Vec2::new(mouse_pos.0, mouse_pos.1);

            if self.is_panning {
                let delta = current_pos - self.last_mouse_pos;
                self.control_panel.pan_offset += delta;
            } else {
                self.is_panning = true;
            }
            self.last_mouse_pos = current_pos;
        } else {
            self.is_panning = false;
        }
    }
}
