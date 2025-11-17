use crate::detection::detector::DetectionResult;
use crate::rtsp::DecodedFrame;
/// 渲染器 (Renderer)
/// 数字卫兵主程序的渲染模块
use crate::xbus::{self, Subscription};
use crate::SKELETON;
use crossbeam_channel::Receiver;
use ggez::event::EventHandler;
use ggez::graphics::{Canvas, Color, DrawMode, DrawParam, Image, Mesh, Rect, Text, TextFragment};
use ggez::mint::Point2;
use ggez::{Context, GameResult};
use std::time::Instant;

pub struct Renderer {
    _frame_sub: Subscription,
    _result_sub: Subscription,
    render_frame_buffer: Receiver<RenderFrame>,
    last_frame: Option<Image>,
    last_detection: Option<DetectionResult>,
    render_count: u64,
    render_last: Instant,
    render_fps: f64,
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
    pub fn new() -> GameResult<Self> {
        println!("🎨 渲染器启动");
        let (tx, rx) = crossbeam_channel::bounded(120);
        // 订阅DecodedFrame
        let tx1 = tx.clone();
        let frame_sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            if let Err(err) = tx1.try_send(RenderFrame::Video(frame.clone())) {
                eprintln!("⚠️ 渲染器通道发送DecodedFrame失败: {}", err);
            }
        });

        // 订阅Det ectionResult
        let result_sub = xbus::subscribe::<DetectionResult, _>(move |result| {
            if let Err(err) = tx.try_send(RenderFrame::Detection(result.clone())) {
                eprintln!("⚠️ 渲染器通道发送DetectionResult失败: {}", err);
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

        // 如果RGB对比度都很低(<10),认为是灰屏/单调帧
        if total_range < 10 {
            eprintln!(
                "⚠️ 渲染器跳过低对比度帧 (RGB范围: {}/{}/{})",
                r_range, g_range, b_range
            );
            return false;
        }

        true
    }
}

impl EventHandler for Renderer {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        if let Some(frame) = self.render_frame_buffer.try_iter().last() {
            match frame {
                RenderFrame::Video(decoded_frame) => {
                    // ✅ 在渲染前做二次质量检查,防止灰屏
                    if !Self::is_valid_frame(&decoded_frame) {
                        // 跳过低质量帧,保留上一帧
                        return Ok(());
                    }

                    // 渲染视频帧
                    let image = Image::from_pixels(
                        ctx,
                        &decoded_frame.rgba_data,
                        ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                        decoded_frame.width,
                        decoded_frame.height,
                    );
                    self.last_frame.replace(image);
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
                for bbox in &detection_result.bboxes {
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
                    let label = format!("Person {:.2}", bbox.confidence);
                    let text = Text::new(TextFragment::new(label).scale(18.0));
                    canvas.draw(
                        &text,
                        DrawParam::default()
                            .dest(Point2 {
                                x: x1,
                                y: y1 - 20.0,
                            })
                            .color(Color::from_rgb(0, 255, 0)),
                    );
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
        canvas.finish(ctx)?;
        Ok(())
    }
}
