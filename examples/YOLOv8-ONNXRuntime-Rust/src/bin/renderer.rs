/// 渲染器 (Renderer)
/// 数字卫兵主程序的渲染模块

use yolov8_rs::systems::{DecodedFrame, DetectionResult};
use yolov8_rs::xbus::{self, Subscription};
use ggez::event::EventHandler;
use ggez::graphics::{self, Canvas, Color, DrawParam, Image, Text, TextFragment};
use ggez::{Context, GameResult};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone)]
pub struct FrameData {
    pub rgba_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frame_id: u64,
}

pub struct Renderer {
    current_frame: Arc<Mutex<Option<FrameData>>>,
    current_result: Arc<Mutex<Option<DetectionResult>>>,
    _frame_sub: Subscription,
    _result_sub: Subscription,
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

impl Renderer {
    pub fn new(_ctx: &mut Context) -> GameResult<Self> {
        println!("🎨 渲染器启动");

        let current_frame = Arc::new(Mutex::new(None));
        let current_result = Arc::new(Mutex::new(None));

        // 订阅DecodedFrame
        let frame_clone = current_frame.clone();
        let frame_sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
            *frame_clone.lock().unwrap() = Some(FrameData {
                rgba_data: frame.rgba_data,
                width: frame.width,
                height: frame.height,
                frame_id: frame.frame_id,
                decode_fps: frame.decode_fps,
                decoder_name: frame.decoder_name,
            });
        });

        // 订阅DetectionResult
        let result_clone = current_result.clone();
        let result_sub = xbus::subscribe::<DetectionResult, _>(move |result| {
            *result_clone.lock().unwrap() = Some(result);
        });

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

impl EventHandler for Renderer {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, Color::BLACK);

        // 渲染视频帧
        if let Some(frame_data) = self.current_frame.lock().unwrap().as_ref() {
            if let Ok(image) = Image::from_pixels(
                ctx,
                &frame_data.rgba_data,
                ggez::graphics::ImageFormat::Rgba8UnormSrgb,
                frame_data.width,
                frame_data.height,
            ) {
                canvas.draw(&image, DrawParam::default());
            }
        }

        // TODO: 绘制检测结果

        // FPS统计
        self.render_count += 1;
        let now = Instant::now();
        if now.duration_since(self.render_last).as_secs() >= 1 {
            self.render_fps = self.render_count as f64 / now.duration_since(self.render_last).as_secs_f64();
            self.render_count = 0;
            self.render_last = now;
        }

        canvas.finish(ctx)?;
        Ok(())
    }
}
