use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_filter::FrameFilter;
use ez_ffmpeg::filter::frame_filter_context::FrameFilterContext;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::Frame;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};
use macroquad::prelude::*;
use std::sync::mpsc::{sync_channel, SyncSender, TryRecvError};
use std::thread;
use yolov8_rs::utils::color_convert::nv12_to_rgba_simd;

/// 帧数据过滤器
struct FrameCapture {
    sender: SyncSender<Vec<u8>>,
    width: u32,
    height: u32,
    frame_count: usize,
}

impl FrameCapture {
    fn new(sender: SyncSender<Vec<u8>>) -> Self {
        Self {
            sender,
            width: 0,
            height: 0,
            frame_count: 0,
        }
    }
}

impl FrameFilter for FrameCapture {
    fn media_type(&self) -> AVMediaType {
        AVMediaType::AVMEDIA_TYPE_VIDEO
    }

    fn init(&mut self, _ctx: &FrameFilterContext) -> Result<(), String> {
        println!("✅ 解码过滤器初始化");
        Ok(())
    }

    fn filter_frame(
        &mut self,
        frame: Frame,
        _ctx: &FrameFilterContext,
    ) -> Result<Option<Frame>, String> {
        unsafe {
            if frame.as_ptr().is_null() || frame.is_empty() {
                return Ok(None);
            }

            let format = (*frame.as_ptr()).format;
            let w = (*frame.as_ptr()).width as u32;
            let h = (*frame.as_ptr()).height as u32;

            // 记录首帧分辨率
            if self.frame_count == 0 {
                self.width = w;
                self.height = h;
            }

            // NV12格式 (format=23) - QSV硬件解码输出,SIMD转RGBA
            if format == 23 {
                let y_plane = (*frame.as_ptr()).data[0];
                let uv_plane = (*frame.as_ptr()).data[1];
                let y_stride = (*frame.as_ptr()).linesize[0] as usize;
                let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

                if !y_plane.is_null() && !uv_plane.is_null() {
                    // 使用 AVX2 优化的 NV12->RGBA 转换
                    let rgba_vec = nv12_to_rgba_simd(y_plane, uv_plane, w, h, y_stride, uv_stride);

                    // 非阻塞发送
                    let _ = self.sender.try_send(rgba_vec);

                    self.frame_count += 1;
                    if self.frame_count == 1 {
                        println!("🎨 NV12格式 - AVX2 SIMD色彩转换");
                    }
                    if self.frame_count % 60 == 0 {
                        println!("📺 已解码 {} 帧", self.frame_count);
                    }
                }
            }
        }
        Ok(Some(frame))
    }

    fn uninit(&mut self, _ctx: &FrameFilterContext) {
        println!("✅ 解码过滤器退出");
    }
}

/// QSV硬件解码RTSP流并实时渲染
#[macroquad::main("QSV RTSP实时显示")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 QSV硬件解码 + Macroquad实时渲染");

    let rtsp_url = std::env::args().nth(1).unwrap_or_else(|| {
        "rtsp://admin:sqb12345@172.19.42.187/cam/realmonitor?channel=1subtype=0".to_string()
    });

    println!("📹 RTSP地址: {}", rtsp_url);
    println!("🎨 QSV Pipeline: HEVC解码(GPU) → NV12 → RGBA (AVX2加速)");
    println!("📝 优化: 无锁通道 + 双缓冲");

    // 创建有界通道 (容量=2,实现双缓冲)
    let (tx, rx) = sync_channel::<Vec<u8>>(2);

    // 启动解码线程
    thread::spawn(move || {
        decode_thread(rtsp_url, tx);
    });

    // 纹理对象
    let mut texture: Option<Texture2D> = None;
    let mut frame_count = 0u64;
    let mut last_fps_time = get_time();
    let mut dropped_frames = 0u64;
    let mut width = 0u32; // 移到外层,保持状态
    let mut height = 0u32;

    println!("✅ 解码线程已启动,等待第一帧...");

    loop {
        // 非阻塞接收最新帧 (丢弃中间帧)
        let mut latest_frame: Option<Vec<u8>> = None;

        loop {
            match rx.try_recv() {
                Ok(rgba_data) => {
                    // 首帧获取分辨率
                    if texture.is_none() {
                        let pixels = rgba_data.len() / 4;
                        // 假设是 4K (3840x2160) 或 1080p (1920x1080)
                        if pixels == 3840 * 2160 {
                            width = 3840;
                            height = 2160;
                        } else if pixels == 1920 * 1080 {
                            width = 1920;
                            height = 1080;
                        } else {
                            // 其他分辨率,尝试推断
                            width = (pixels as f64).sqrt() as u32 * 16 / 9;
                            height = pixels as u32 / width;
                        }
                    }
                    latest_frame = Some(rgba_data);
                    dropped_frames += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    println!("⚠️ 解码线程已退出");
                    return Ok(());
                }
            }
        }

        // 有新帧时更新纹理
        if let Some(rgba_data) = latest_frame {
            dropped_frames = dropped_frames.saturating_sub(1);

            if texture.is_none() {
                texture = Some(Texture2D::from_rgba8(
                    width as u16,
                    height as u16,
                    &rgba_data,
                ));
                println!("✅ 创建纹理: {}x{}", width, height);
            } else {
                let image = Image {
                    bytes: rgba_data,
                    width: width as u16,
                    height: height as u16,
                };
                texture.as_ref().unwrap().update(&image);
            }

            frame_count += 1;

            let now = get_time();
            if now - last_fps_time >= 1.0 {
                let fps = frame_count as f64 / (now - last_fps_time);
                println!("🎬 渲染FPS: {:.1} | 丢帧: {}", fps, dropped_frames);
                frame_count = 0;
                dropped_frames = 0;
                last_fps_time = now;
            }
        }

        clear_background(BLACK);

        // 只有在纹理创建后才渲染
        if let Some(ref tex) = texture {
            let screen_w = screen_width();
            let screen_h = screen_height();
            let tex_w = tex.width();
            let tex_h = tex.height();

            let scale = (screen_w / tex_w).min(screen_h / tex_h);
            let draw_w = tex_w * scale;
            let draw_h = tex_h * scale;
            let x = (screen_w - draw_w) * 0.5;
            let y = (screen_h - draw_h) * 0.5;

            draw_texture_ex(
                tex,
                x,
                y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(draw_w, draw_h)),
                    ..Default::default()
                },
            );

            draw_text(
                &format!("分辨率: {}x{}", tex_w as u32, tex_h as u32),
                10.0,
                30.0,
                30.0,
                GREEN,
            );
            draw_text("QSV硬件解码", 10.0, 60.0, 30.0, GREEN);
            draw_text("按ESC退出", 10.0, 90.0, 30.0, GREEN);
        }

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }

    println!("👋 程序退出");
    Ok(())
}

/// 解码线程
fn decode_thread(rtsp_url: String, sender: SyncSender<Vec<u8>>) {
    println!("🔧 初始化QSV解码器...");

    // 构建输入 - QSV硬件解码
    let input = Input::new(&rtsp_url)
        .set_hwaccel("qsv")
        .set_hwaccel_output_format("nv12") // QSV解码后下载NV12到CPU
        .set_video_codec("hevc_qsv")
        .set_input_opts(
            [
                ("rtsp_transport", "tcp"),
                ("buffer_size", "67108864"),
                ("rtsp_flags", "prefer_tcp"),
            ]
            .into(),
        );

    // 构建过滤器管道
    let filter = FrameCapture::new(sender);

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("capture", Box::new(filter));
    let output = create_null_output().add_frame_pipeline(pipe);

    println!("🔍 QSV硬件解码(GPU) + AVX2色彩转换");
    println!("📌 双缓冲机制,简化内存管理");

    // 构建并运行FFmpeg上下文
    match FfmpegContext::builder()
        .input(input)
        // 不使用filter_desc - QSV无法直接输出RGBA
        .output(output)
        .build()
    {
        Ok(ctx) => {
            println!("✅ FFmpeg上下文构建成功");
            match ctx.start() {
                Ok(sch) => {
                    println!("✅ QSV解码启动成功");
                    let _ = sch.wait();
                }
                Err(e) => eprintln!("❌ 启动失败: {:?}", e),
            }
        }
        Err(e) => eprintln!("❌ 构建失败: {:?}", e),
    }

    println!("⚠️ 解码线程退出");
}
