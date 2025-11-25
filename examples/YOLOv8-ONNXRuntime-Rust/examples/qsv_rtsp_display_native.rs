// QSV 硬件解码 + Macroquad 渲染
// 使用优雅的 Rust 风格封装 - 双线程版本

use crossbeam::channel::{bounded, Receiver, Sender};
use macroquad::prelude::*;
use std::thread;
use std::time::Instant;
use yolov8_rs::utils::color_convert::nv12_to_rgba_simd_into;
use yolov8_rs::video::{DecoderConfig, HardwareAccel, QsvDecoder, RtspTransport};

// 渲染帧数据
struct RenderFrame {
    rgba_data: Vec<u8>,
    width: u32,
    height: u32,
}

// 解码线程
fn decode_thread(config: DecoderConfig, sender: Sender<RenderFrame>) {
    println!("🔄 解码线程启动");

    let mut decoder = match QsvDecoder::new(config) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("❌ 创建解码器失败: {}", e);
            return;
        }
    };

    let info = decoder.video_info();
    println!(
        "📊 视频信息: {}x{}, {}",
        info.width, info.height, info.pixel_format
    );

    let mut rgba_buffer = Vec::new();
    let mut frame_count = 0u64;

    loop {
        match decoder.decode_next_frame() {
            Ok(Some(video_frame)) => {
                // NV12 → RGBA 转换
                unsafe {
                    nv12_to_rgba_simd_into(
                        video_frame.y_data.as_ptr(),
                        video_frame.uv_data.as_ptr(),
                        video_frame.width,
                        video_frame.height,
                        video_frame.y_stride,
                        video_frame.uv_stride,
                        &mut rgba_buffer,
                    );
                }

                let render_frame = RenderFrame {
                    rgba_data: rgba_buffer.clone(),
                    width: video_frame.width,
                    height: video_frame.height,
                };

                // 非阻塞发送,自动丢弃旧帧
                let _ = sender.try_send(render_frame);

                frame_count += 1;
                if frame_count % 60 == 0 {
                    println!("📺 已解码 {} 帧", frame_count);
                }
            }
            Ok(None) => {
                println!("⚠️ 流结束");
                break;
            }
            Err(e) => {
                eprintln!("❌ 解码错误: {}", e);
                break;
            }
        }
    }

    println!("⏹️  解码线程退出");
}

#[macroquad::main("QSV 硬件解码 + Macroquad")]
async fn main() {
    println!("🚀 QSV 硬件解码 + Macroquad 渲染");
    println!("📹 使用优雅的 Rust 风格封装 (双线程版本)");

    let rtsp_url = std::env::args().nth(1).unwrap_or_else(|| {
        "rtsp://admin:sqb12345@172.19.42.187/cam/realmonitor?channel=1&subtype=0".to_string()
    });

    println!("📡 RTSP: {}", rtsp_url);

    // 配置解码器
    let config = DecoderConfig {
        url: rtsp_url,
        hardware_accel: HardwareAccel::Qsv,
        rtsp_transport: RtspTransport::Tcp,
        buffer_size: 5 * 1024 * 1024,
        max_delay: 0,
    };

    // 创建 channel (缓冲 1 帧)
    let (tx, rx): (Sender<RenderFrame>, Receiver<RenderFrame>) = bounded(1);

    // 启动解码线程
    thread::spawn(move || decode_thread(config, tx));

    // 渲染状态
    let mut texture: Option<Texture2D> = None;
    let mut frame_count = 0u64;
    let mut last_fps_time = Instant::now();
    let mut video_width = 0u32;
    let mut video_height = 0u32;

    // 渲染循环
    loop {
        // 接收最新帧
        let mut latest_frame = None;
        while let Ok(frame) = rx.try_recv() {
            latest_frame = Some(frame);
        }

        // 只在有新帧时才更新纹理和渲染
        if let Some(frame) = latest_frame {
            if texture.is_none() {
                video_width = frame.width;
                video_height = frame.height;
                texture = Some(Texture2D::from_rgba8(
                    video_width as u16,
                    video_height as u16,
                    &frame.rgba_data,
                ));
                println!("✅ 创建纹理: {}x{}", video_width, video_height);
            } else {
                texture.as_ref().unwrap().update(&Image {
                    bytes: frame.rgba_data,
                    width: video_width as u16,
                    height: video_height as u16,
                });
            }
            frame_count += 1;

            // FPS 统计
            let elapsed = last_fps_time.elapsed();
            if elapsed.as_secs() >= 1 {
                let fps = frame_count as f64 / elapsed.as_secs_f64();
                println!("🎬 FPS: {:.1}", fps);
                frame_count = 0;
                last_fps_time = Instant::now();
            }

            // 渲染
            clear_background(BLACK);

            if let Some(ref tex) = texture {
                let screen_w = screen_width();
                let screen_h = screen_height();

                // 计算缩放比例 (保持宽高比)
                let scale = (screen_w / video_width as f32).min(screen_h / video_height as f32);
                let draw_w = video_width as f32 * scale;
                let draw_h = video_height as f32 * scale;
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

                // 信息显示
                draw_text(
                    &format!("{}x{}", video_width, video_height),
                    10.0,
                    30.0,
                    30.0,
                    GREEN,
                );
                draw_text("QSV 硬件解码 + SIMD", 10.0, 60.0, 30.0, GREEN);
                draw_text("ESC 退出", 10.0, 90.0, 30.0, GREEN);
            }
        } // 只有收到新帧才渲染

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }

    println!("👋 退出");
}
