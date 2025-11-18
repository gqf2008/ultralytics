use crate::detection::detector::DetectionResult;
use crate::detection::types::DecodedFrame;
/// 渲染器 (Renderer)
/// 数字卫兵主程序的渲染模块
use crate::xbus::{self, Subscription};
use crate::SKELETON;
use crossbeam_channel::Receiver;
use ggez::event::{EventHandler, MouseButton};
use ggez::graphics::{Canvas, Color, DrawMode, DrawParam, Image, Mesh, Rect, Text, TextFragment};
use ggez::input::keyboard::KeyCode;
use ggez::mint::Point2;
use ggez::{Context, GameResult};
use std::time::Instant;

// ========== 公共常量 ==========

pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;

pub struct Renderer {
    _frame_sub: Subscription,
    _result_sub: Subscription,
    render_frame_buffer: Receiver<RenderFrame>,
    last_frame: Option<Image>,
    last_detection: Option<DetectionResult>,
    render_count: u64,
    render_last: Instant,
    render_fps: f64,
    // Resize预览窗口拖动相关
    preview_pos: (f32, f32),         // 预览窗口位置 (默认右下角)
    preview_dragging: bool,          // 是否正在拖动
    preview_drag_offset: (f32, f32), // 拖动时鼠标相对预览窗口的偏移
    preview_visible: bool,           // 是否显示预览窗口 (Space键切换)
    // 系统配置信息
    detect_model_name: String, // 检测模型名称
    pose_model_name: String,   // 姿态模型名称
    tracker_name: String,      // 追踪器名称
    #[allow(dead_code)]
    detect_fps: f64, // 检测FPS (保留用于未来统计)
    #[allow(dead_code)]
    decode_fps: f64, // 解码FPS (保留用于未来统计)
}

/// 系统控制
#[derive(Clone, Debug)]
pub enum SystemControl {
    PauseDecode,
    ResumeDecode,
    Shutdown,
    SwitchTracker(String),
}
enum RenderFrame {
    Video(DecodedFrame),
    Detection(DetectionResult),
}

impl Renderer {
    pub fn new(detect_model: String, pose_model: String, tracker: String) -> GameResult<Self> {
        println!("渲染器启动");
        let (tx, rx) = crossbeam_channel::bounded(120);
        // 订阅DecodedFrame
        let tx1 = tx.clone();
        let frame_sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            if let Err(err) = tx1.try_send(RenderFrame::Video(frame.clone())) {
                eprintln!("渲染器通道发送DecodedFrame失败: {}", err);
            }
        });

        // 订阅Det ectionResult
        let result_sub = xbus::subscribe::<DetectionResult, _>(move |result| {
            if let Err(err) = tx.try_send(RenderFrame::Detection(result.clone())) {
                eprintln!("渲染器通道发送DetectionResult失败: {}", err);
            }
        });

        Ok(Self {
            render_frame_buffer: rx,
            last_frame: None,
            last_detection: None,
            _frame_sub: frame_sub,
            _result_sub: result_sub,
            render_count: 0,
            render_last: Instant::now(),
            render_fps: 0.0,
            preview_pos: (0.0, 0.0), // 初始化为(0,0),在draw时设置为右下角
            preview_dragging: false,
            preview_drag_offset: (0.0, 0.0),
            preview_visible: true, // 默认显示预览窗口
            detect_model_name: detect_model,
            pose_model_name: pose_model,
            tracker_name: tracker,
            detect_fps: 0.0,
            decode_fps: 0.0,
        })
    }

    /// 验证帧质量 - 防止灰屏/单调帧渲染
    fn is_valid_frame(frame: &DecodedFrame) -> bool {
        // 采样检查RGBA数据 (检查RGB,忽略Alpha)
        let data = &frame.rgba_data;
        let pixel_count = (frame.width * frame.height) as usize;

        if pixel_count == 0 || data.len() < pixel_count * 4 {
            return false;
        }

        // 采样25个点检查对比度
        let sample_indices = [
            pixel_count / 6,
            pixel_count / 3,
            pixel_count / 2,
            pixel_count * 2 / 3,
            pixel_count * 5 / 6,
            pixel_count / 4,
            pixel_count * 3 / 8,
            pixel_count * 5 / 8,
            pixel_count * 3 / 4,
            pixel_count / 5,
        ];

        let mut r_min = 255u8;
        let mut r_max = 0u8;
        let mut g_min = 255u8;
        let mut g_max = 0u8;
        let mut b_min = 255u8;
        let mut b_max = 0u8;

        for &idx in &sample_indices {
            if idx * 4 + 2 < data.len() {
                let r = data[idx * 4];
                let g = data[idx * 4 + 1];
                let b = data[idx * 4 + 2];

                r_min = r_min.min(r);
                r_max = r_max.max(r);
                g_min = g_min.min(g);
                g_max = g_max.max(g);
                b_min = b_min.min(b);
                b_max = b_max.max(b);
            }
        }

        let r_range = r_max - r_min;
        let g_range = g_max - g_min;
        let b_range = b_max - b_min;
        let total_range = r_range.max(g_range).max(b_range);

        // 降低阈值避免窗口无响应 (从10降到3)
        if total_range < 3 {
            // 不再打印警告,避免刷屏
            return false;
        }

        true
    }

    /// 绘制系统统计信息面板 (左上角)
    fn draw_stats_panel(&self, ctx: &mut Context, canvas: &mut Canvas) -> GameResult {
        let margin = 10.0;
        let panel_width = 280.0;
        let line_height = 22.0;
        let font_size = 16.0;

        // 准备统计信息文本
        let mut lines = vec![
            format!("数字卫兵 Digital Sentinel"),
            format!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━"),
        ];

        // 检测模型
        lines.push(format!("检测: {}", self.detect_model_name));

        // 姿态模型
        if !self.pose_model_name.is_empty() {
            lines.push(format!("姿态: {}", self.pose_model_name));
        }

        // 追踪器
        lines.push(format!("追踪: {}", self.tracker_name));

        lines.push(format!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━"));

        // 性能统计
        if let Some(detection) = &self.last_detection {
            lines.push(format!("检测FPS: {:.1}", detection.inference_fps));
            lines.push(format!("检测耗时: {:.1}ms", detection.inference_ms));
            lines.push(format!("追踪FPS: {:.1}", detection.tracker_fps));
            lines.push(format!("追踪耗时: {:.2}ms", detection.tracker_ms));
            lines.push(format!("人数: {}", detection.bboxes.len()));
        } else {
            lines.push(format!("检测FPS: --"));
            lines.push(format!("检测耗时: --"));
            lines.push(format!("追踪FPS: --"));
            lines.push(format!("追踪耗时: --"));
            lines.push(format!("人数: 0"));
        }

        lines.push(format!("渲染FPS: {:.1}", self.render_fps));

        // 计算面板高度
        let panel_height = lines.len() as f32 * line_height + 20.0;

        // 绘制半透明背景
        let bg_rect = Rect::new(margin, margin, panel_width, panel_height);
        let bg_mesh = Mesh::new_rectangle(
            ctx,
            DrawMode::fill(),
            bg_rect,
            Color::from_rgba(0, 0, 0, 180), // 半透明黑色
        )?;
        canvas.draw(&bg_mesh, DrawParam::default());

        // 绘制边框
        let border_mesh = Mesh::new_rectangle(
            ctx,
            DrawMode::stroke(2.0),
            bg_rect,
            Color::from_rgb(0, 200, 255), // 青蓝色边框
        )?;
        canvas.draw(&border_mesh, DrawParam::default());

        // 绘制文本
        for (i, line) in lines.iter().enumerate() {
            let y_pos = margin + 10.0 + i as f32 * line_height;
            let text = Text::new(
                TextFragment::new(line.clone())
                    .font("MicrosoftYaHei")
                    .scale(font_size),
            );
            canvas.draw(
                &text,
                DrawParam::default()
                    .dest(Point2 {
                        x: margin + 10.0,
                        y: y_pos,
                    })
                    .color(Color::from_rgb(255, 255, 255)),
            );
        }

        Ok(())
    }
}

impl EventHandler for Renderer {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        if let Some(frame) = self.render_frame_buffer.try_iter().last() {
            match frame {
                RenderFrame::Video(decoded_frame) => {
                    // 只更新有效帧,低质量帧保留上一帧 (不要return,保持事件循环)
                    if Self::is_valid_frame(&decoded_frame) {
                        let image = Image::from_pixels(
                            ctx,
                            &decoded_frame.rgba_data,
                            ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                            decoded_frame.width,
                            decoded_frame.height,
                        );
                        self.last_frame.replace(image);
                    }
                    // 即使跳过帧也继续,不要return,保持窗口响应
                }
                RenderFrame::Detection(detection_result) => {
                    self.last_detection.replace(detection_result);
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::BLACK);

        // 绘制视频帧
        if let Some(image) = &self.last_frame {
            let scale_x = ctx.gfx.drawable_size().0 / image.width() as f32;
            let scale_y = ctx.gfx.drawable_size().1 / image.height() as f32;
            canvas.draw(image, DrawParam::default().scale([scale_x, scale_y]));

            // 绘制检测结果
            if let Some(detection_result) = &self.last_detection {
                // bbox坐标在原始视频分辨率下,需要缩放到窗口尺寸
                for (idx, bbox) in detection_result.bboxes.iter().enumerate() {
                    let x1 = bbox.x1 * scale_x;
                    let y1 = bbox.y1 * scale_y;
                    let w = (bbox.x2 - bbox.x1) * scale_x;
                    let h = (bbox.y2 - bbox.y1) * scale_y;

                    // 边界框矩形
                    let rect = Rect::new(x1, y1, w, h);
                    let box_mesh = Mesh::new_rectangle(
                        ctx,
                        DrawMode::stroke(3.0),
                        rect,
                        Color::from_rgb(0, 255, 0), // 绿色边框
                    )?;
                    canvas.draw(&box_mesh, DrawParam::default());

                    // 置信度标签
                    let label = format!("ID:{} {:.2}", bbox.class_id, bbox.confidence);
                    let text =
                        Text::new(TextFragment::new(label).font("MicrosoftYaHei").scale(18.0));
                    canvas.draw(
                        &text,
                        DrawParam::default()
                            .dest(Point2 {
                                x: x1,
                                y: y1 - 20.0,
                            })
                            .color(Color::from_rgb(0, 255, 0)),
                    );

                    // 绘制ReID特征可视化 (色块)
                    if idx < detection_result.reid_features.len() {
                        let features = &detection_result.reid_features[idx];
                        if !features.is_empty() {
                            // 使用前3个特征维度作为RGB颜色
                            let r =
                                ((features.get(0).unwrap_or(&0.0).abs() * 255.0).min(255.0)) as u8;
                            let g =
                                ((features.get(1).unwrap_or(&0.0).abs() * 255.0).min(255.0)) as u8;
                            let b =
                                ((features.get(2).unwrap_or(&0.0).abs() * 255.0).min(255.0)) as u8;

                            // 在bbox右上角绘制ReID特征色块
                            let reid_rect = Rect::new(x1 + w - 30.0, y1, 30.0, 30.0);
                            let reid_mesh = Mesh::new_rectangle(
                                ctx,
                                DrawMode::fill(),
                                reid_rect,
                                Color::from_rgb(r, g, b),
                            )?;
                            canvas.draw(&reid_mesh, DrawParam::default());

                            // 绘制边框
                            let reid_border = Mesh::new_rectangle(
                                ctx,
                                DrawMode::stroke(2.0),
                                reid_rect,
                                Color::WHITE,
                            )?;
                            canvas.draw(&reid_border, DrawParam::default());
                        }
                    }
                }

                // 绘制姿态骨架
                for keypoints in &detection_result.keypoints {
                    if keypoints.points.is_empty() {
                        continue;
                    }

                    // 绘制关键点
                    for (x, y, conf) in &keypoints.points {
                        if *conf > 0.3 {
                            let point_mesh = Mesh::new_circle(
                                ctx,
                                DrawMode::fill(),
                                Point2 {
                                    x: *x * scale_x,
                                    y: *y * scale_y,
                                },
                                4.0,
                                0.1,
                                Color::from_rgb(255, 0, 0), // 红色关键点
                            )?;
                            canvas.draw(&point_mesh, DrawParam::default());
                        }
                    }

                    // 绘制骨架连接
                    for (idx1, idx2) in &SKELETON {
                        if *idx1 < keypoints.points.len() && *idx2 < keypoints.points.len() {
                            let (x1, y1, c1) = keypoints.points[*idx1];
                            let (x2, y2, c2) = keypoints.points[*idx2];
                            if c1 > 0.3 && c2 > 0.3 {
                                let line = Mesh::new_line(
                                    ctx,
                                    &[
                                        Point2 {
                                            x: x1 * scale_x,
                                            y: y1 * scale_y,
                                        },
                                        Point2 {
                                            x: x2 * scale_x,
                                            y: y2 * scale_y,
                                        },
                                    ],
                                    2.0,
                                    Color::from_rgb(255, 255, 0), // 黄色骨架
                                )?;
                                canvas.draw(&line, DrawParam::default());
                            }
                        }
                    }
                }

                // 在右下角显示resize后的图像 (可拖动, Space键切换显示)
                if self.preview_visible {
                    if let Some(ref resized_data) = detection_result.resized_image {
                        let (window_width, window_height) = ctx.gfx.drawable_size();

                        let resized_img = Image::from_pixels(
                            ctx,
                            resized_data,
                            ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                            detection_result.resized_size,
                            detection_result.resized_size,
                        );

                        // 计算预览窗口位置 (首次默认右下角,之后使用拖动位置)
                        let margin = 10.0;
                        let preview_size = detection_result.resized_size.min(300) as f32; // 动态尺寸,最大300
                        let preview_scale = preview_size / detection_result.resized_size as f32;

                        // 如果还未初始化位置,设为右下角
                        if self.preview_pos == (0.0, 0.0) {
                            self.preview_pos = (
                                window_width - preview_size - margin,
                                window_height - preview_size - margin,
                            );
                        }

                        let x = self.preview_pos.0;
                        let y = self.preview_pos.1;

                        // 绘制边框
                        let border_rect =
                            Rect::new(x - 2.0, y - 2.0, preview_size + 4.0, preview_size + 4.0);
                        let border_mesh = Mesh::new_rectangle(
                            ctx,
                            DrawMode::stroke(2.0),
                            border_rect,
                            Color::from_rgb(0, 255, 255), // 青色边框
                        )?;
                        canvas.draw(&border_mesh, DrawParam::default());

                        // 绘制resize后的图像
                        canvas.draw(
                            &resized_img,
                            DrawParam::default()
                                .dest(Point2 { x, y })
                                .scale([preview_scale, preview_scale]),
                        );

                        // 添加标签
                        let label_text = format!(
                            "推理输入 {}x{}",
                            detection_result.resized_size, detection_result.resized_size
                        );
                        let label = Text::new(
                            TextFragment::new(label_text)
                                .font("MicrosoftYaHei")
                                .scale(18.0),
                        );
                        canvas.draw(
                            &label,
                            DrawParam::default()
                                .dest(Point2 { x, y: y - 25.0 })
                                .color(Color::from_rgb(0, 255, 255)),
                        );
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

        // 绘制左上角系统统计信息面板
        self.draw_stats_panel(ctx, &mut canvas)?;

        canvas.finish(ctx)?;
        Ok(())
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        button: MouseButton,
        x: f32,
        y: f32,
    ) -> GameResult {
        if button == MouseButton::Left {
            // 检查是否点击在resize预览区域
            let preview_size = 200.0;
            let px = self.preview_pos.0;
            let py = self.preview_pos.1;

            if x >= px && x <= px + preview_size && y >= py && y <= py + preview_size {
                self.preview_dragging = true;
                self.preview_drag_offset = (x - px, y - py);
            }
        }
        Ok(())
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        button: MouseButton,
        _x: f32,
        _y: f32,
    ) -> GameResult {
        if button == MouseButton::Left {
            self.preview_dragging = false;
        }
        Ok(())
    }

    fn mouse_motion_event(
        &mut self,
        ctx: &mut Context,
        x: f32,
        y: f32,
        _dx: f32,
        _dy: f32,
    ) -> GameResult {
        if self.preview_dragging {
            let preview_size = 200.0;
            let (window_width, window_height) = ctx.gfx.drawable_size();

            // 计算新位置 (考虑拖动偏移)
            let mut new_x = x - self.preview_drag_offset.0;
            let mut new_y = y - self.preview_drag_offset.1;

            // 限制在窗口范围内
            new_x = new_x.max(0.0).min(window_width - preview_size);
            new_y = new_y.max(0.0).min(window_height - preview_size);

            self.preview_pos = (new_x, new_y);
        }
        Ok(())
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        input: ggez::input::keyboard::KeyInput,
        _repeated: bool,
    ) -> GameResult {
        // Space键切换预览窗口显示/隐藏
        if input.keycode == Some(KeyCode::Space) {
            self.preview_visible = !self.preview_visible;
            println!(
                "🔲 Resize预览窗口: {}",
                if self.preview_visible {
                    "显示"
                } else {
                    "隐藏"
                }
            );
        }
        Ok(())
    }
}
