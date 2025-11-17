/// 渲染模块 - 主线程运行 (ggez事件循环)
/// 负责: 订阅DecodedFrame + DetectionResult → GPU渲染
use crate::rtsp_xbus::{DecodedFrame, DetectionResult};
use crate::xbus::{self, Subscription};
use ggez::event::EventHandler;
use ggez::graphics::{self, Canvas, Color, DrawParam, Image, Text, TextFragment};
use ggez::{Context, GameResult};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct RenderModule {
    // 当前帧数据 (通过Arc<Mutex>在订阅回调和ggez线程间共享)
    current_frame: Arc<Mutex<Option<FrameData>>>,
    current_result: Arc<Mutex<Option<DetectionResult>>>,

    // 订阅凭证 (保持订阅活跃)
    _frame_sub: Subscription,
    _result_sub: Subscription,

    // 渲染统计
    render_count: u64,
    render_last: Instant,
    render_fps: f64,
}

struct FrameData {
    rgba_data: Vec<u8>,
    width: u32,
    height: u32,
    frame_id: u64,
    decode_fps: f64,
    decoder_name: String,
}

impl RenderModule {
    pub fn new(_ctx: &mut Context) -> GameResult<Self> {
        println!("🎨 渲染模块启动");

        let current_frame = Arc::new(Mutex::new(None));
        let current_result = Arc::new(Mutex::new(None));

        // 订阅解码帧
        let frame_clone = current_frame.clone();
        let frame_sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            *frame_clone.lock().unwrap() = Some(FrameData {
                rgba_data: frame.rgba_data.clone(),
                width: frame.width,
                height: frame.height,
                frame_id: frame.frame_id,
                decode_fps: frame.decode_fps,
                decoder_name: frame.decoder_name.clone(),
            });
        });

        // 订阅检测结果
        let result_clone = current_result.clone();
        let result_sub = xbus::subscribe::<DetectionResult, _>(move |result| {
            *result_clone.lock().unwrap() = Some(result.clone());
        });

        println!("✅ 渲染模块已订阅消息");

        Ok(Self {
            current_frame,
            current_result,
            _frame_sub: frame_sub,
            _result_sub: result_sub,
            render_count: 0,
            render_last: Instant::now(),
            render_fps: 0.0,
        })
    }
}

impl EventHandler for RenderModule {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        // ggez的update不需要做什么,数据通过订阅回调更新
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::BLACK);

        // 获取当前帧
        let frame_opt = self.current_frame.lock().unwrap().clone();
        let result_opt = self.current_result.lock().unwrap().clone();

        if let Some(frame_data) = frame_opt {
            self.render_count += 1;

            // 计算渲染FPS
            if self.render_last.elapsed().as_secs_f64() >= 1.0 {
                self.render_fps =
                    self.render_count as f64 / self.render_last.elapsed().as_secs_f64();
                self.render_last = Instant::now();
                self.render_count = 0;
            }

            // 创建图像
            let image = Image::from_pixels(
                ctx,
                &frame_data.rgba_data,
                graphics::ImageFormat::Rgba8UnormSrgb,
                frame_data.width,
                frame_data.height,
            );

            // GPU缩放到窗口
            let (window_width, window_height) = ctx.gfx.drawable_size();
            let scale_x = window_width / frame_data.width as f32;
            let scale_y = window_height / frame_data.height as f32;

            canvas.draw(&image, DrawParam::default().scale([scale_x, scale_y]));

            // 绘制检测框
            if let Some(result) = result_opt {
                for bbox in &result.bboxes {
                    let rect = graphics::Rect::new(
                        bbox.x1 * scale_x,
                        bbox.y1 * scale_y,
                        (bbox.x2 - bbox.x1) * scale_x,
                        (bbox.y2 - bbox.y1) * scale_y,
                    );

                    let mesh = graphics::Mesh::new_rectangle(
                        ctx,
                        graphics::DrawMode::stroke(2.0),
                        rect,
                        Color::from_rgb(0, 255, 0),
                    )?;

                    canvas.draw(&mesh, DrawParam::default());
                }

                // 显示FPS和统计信息
                let fps_text = format!(
                    "FPS - 解码:{:.1} | 推理:{:.1}({:.1}ms) | 渲染:{:.1} | 检测:{}人",
                    frame_data.decode_fps,
                    result.inference_fps,
                    result.inference_ms,
                    self.render_fps,
                    result.bboxes.len()
                );

                let fps_fragment = TextFragment::new(fps_text)
                    .font("MicrosoftYaHei")
                    .scale(24.0);
                let fps_display = Text::new(fps_fragment);
                canvas.draw(
                    &fps_display,
                    DrawParam::default().dest([10.0, 10.0]).color(Color::WHITE),
                );

                // 解码器信息
                let decoder_text = format!("解码器: {}", frame_data.decoder_name);
                let decoder_fragment = TextFragment::new(decoder_text)
                    .font("MicrosoftYaHei")
                    .scale(24.0);
                let decoder_display = Text::new(decoder_fragment);
                canvas.draw(
                    &decoder_display,
                    DrawParam::default().dest([10.0, 40.0]).color(Color::WHITE),
                );
            }
        } else {
            // 无数据时显示等待提示
            let wait_text = "等待RTSP数据...";
            let text_fragment = TextFragment::new(wait_text)
                .font("MicrosoftYaHei")
                .scale(48.0);
            let text_display = Text::new(text_fragment);
            canvas.draw(
                &text_display,
                DrawParam::default()
                    .dest([400.0, 300.0])
                    .color(Color::WHITE),
            );
        }

        canvas.finish(ctx)?;
        Ok(())
    }
}
