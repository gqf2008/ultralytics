use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_filter::FrameFilter;
use ez_ffmpeg::filter::frame_filter_context::FrameFilterContext;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::Frame;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};
use macroquad::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;

/// 帧数据过滤器 - 提取RGBA数据并存入共享缓冲区
struct FrameCapture {
    frame_buffer: Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>,
    frame_count: usize,
}

impl FrameCapture {
    fn new(frame_buffer: Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>) -> Self {
        Self {
            frame_buffer,
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

            // NV12格式 (format=23) - QSV硬件解码输出,CPU转RGBA
            if format == 23 {
                let y_plane = (*frame.as_ptr()).data[0];
                let uv_plane = (*frame.as_ptr()).data[1]; // NV12: UV交错存储
                let y_stride = (*frame.as_ptr()).linesize[0] as usize;
                let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

                if !y_plane.is_null() && !uv_plane.is_null() {
                    let pixel_count = (w * h) as usize;
                    let mut rgba_vec = vec![255u8; pixel_count * 4];

                    // NV12->RGBA转换 (UV交错格式)
                    for y in 0..(h as usize) {
                        for x in 0..(w as usize) {
                            let y_val = *y_plane.add(y * y_stride + x) as i32;

                            // NV12: UV平面交错存储 [U0,V0,U1,V1,...]
                            let uv_idx = (y >> 1) * uv_stride + (x & !1);
                            let u_val = *uv_plane.add(uv_idx) as i32 - 128;
                            let v_val = *uv_plane.add(uv_idx + 1) as i32 - 128;

                            let r = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
                            let g = (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7))
                                .clamp(0, 255) as u8;
                            let b = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;

                            let idx = (y * w as usize + x) * 4;
                            rgba_vec[idx] = r;
                            rgba_vec[idx + 1] = g;
                            rgba_vec[idx + 2] = b;
                        }
                    }

                    // 更新共享缓冲区
                    if let Ok(mut buffer) = self.frame_buffer.lock() {
                        *buffer = Some((rgba_vec, w, h));
                    }

                    self.frame_count += 1;
                    if self.frame_count == 1 {
                        println!("🎨 检测到NV12格式 - QSV硬件解码,CPU色彩转换");
                    }
                    if self.frame_count % 60 == 0 {
                        println!("📺 已解码 {} 帧 ({}x{})", self.frame_count, w, h);
                    }
                }
            }
            // RGBA格式 (format=26) - GPU已转换好,直接复制
            else if format == 26 {
                let rgba_plane = (*frame.as_ptr()).data[0];
                let stride = (*frame.as_ptr()).linesize[0] as usize;

                if !rgba_plane.is_null() {
                    let pixel_count = (w * h) as usize;
                    let mut rgba_vec = vec![255u8; pixel_count * 4];

                    // 直接复制RGBA数据(GPU VPP已转换好)
                    for y in 0..(h as usize) {
                        let src = std::slice::from_raw_parts(
                            rgba_plane.add(y * stride),
                            (w as usize) * 4,
                        );
                        let dst_offset = y * (w as usize) * 4;
                        rgba_vec[dst_offset..dst_offset + src.len()].copy_from_slice(src);
                    }

                    // 更新共享缓冲区
                    if let Ok(mut buffer) = self.frame_buffer.lock() {
                        *buffer = Some((rgba_vec, w, h));
                    }

                    self.frame_count += 1;
                    if self.frame_count == 1 {
                        println!("🎨 检测到RGBA格式 - GPU VPP色彩转换");
                    }
                    if self.frame_count % 60 == 0 {
                        println!("📺 已解码 {} 帧 ({}x{}) - GPU转换", self.frame_count, w, h);
                    }
                }
            }
            // YUV420P格式 (format=0) - CPU转换(备用)
            else if format == 0 {
                let y_plane = (*frame.as_ptr()).data[0];
                let u_plane = (*frame.as_ptr()).data[1];
                let v_plane = (*frame.as_ptr()).data[2];
                let y_stride = (*frame.as_ptr()).linesize[0] as usize;
                let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

                if !y_plane.is_null() && !u_plane.is_null() && !v_plane.is_null() {
                    let pixel_count = (w * h) as usize;
                    let mut rgba_vec = vec![255u8; pixel_count * 4];

                    // 简单的YUV->RGBA转换
                    for y in 0..(h as usize) {
                        for x in 0..(w as usize) {
                            let y_val = *y_plane.add(y * y_stride + x) as i32;
                            let u_val = *u_plane.add((y >> 1) * uv_stride + (x >> 1)) as i32 - 128;
                            let v_val = *v_plane.add((y >> 1) * uv_stride + (x >> 1)) as i32 - 128;

                            let r = (y_val + ((v_val * 179) >> 7)).clamp(0, 255) as u8;
                            let g = (y_val - ((u_val * 44) >> 7) - ((v_val * 91) >> 7))
                                .clamp(0, 255) as u8;
                            let b = (y_val + ((u_val * 227) >> 7)).clamp(0, 255) as u8;

                            let idx = (y * w as usize + x) * 4;
                            rgba_vec[idx] = r;
                            rgba_vec[idx + 1] = g;
                            rgba_vec[idx + 2] = b;
                        }
                    }

                    // 更新共享缓冲区
                    if let Ok(mut buffer) = self.frame_buffer.lock() {
                        *buffer = Some((rgba_vec, w, h));
                    }

                    self.frame_count += 1;
                    if self.frame_count % 60 == 0 {
                        println!("📺 已解码 {} 帧 ({}x{})", self.frame_count, w, h);
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
    println!("🎨 QSV Pipeline: HEVC解码(GPU) -> NV12 -> RGBA (CPU转换)");
    println!("📝 注: 真正的GPU色彩转换需要保持QSV surface不下载到CPU");

    // 共享帧缓冲区
    let frame_buffer: Arc<Mutex<Option<(Vec<u8>, u32, u32)>>> = Arc::new(Mutex::new(None));
    let frame_buffer_clone = frame_buffer.clone();

    // 启动解码线程
    thread::spawn(move || {
        decode_thread(rtsp_url, frame_buffer_clone);
    });

    // 纹理对象
    let mut texture: Option<Texture2D> = None;
    let mut frame_count = 0u64;
    let mut last_fps_time = get_time();

    println!("✅ 解码线程已启动,等待第一帧...");

    loop {
        // 尝试获取新帧
        if let Ok(mut buffer) = frame_buffer.lock() {
            if let Some((rgba_data, width, height)) = buffer.take() {
                let image = Image {
                    bytes: rgba_data,
                    width: width as u16,
                    height: height as u16,
                };

                texture = Some(Texture2D::from_image(&image));
                frame_count += 1;

                let now = get_time();
                if now - last_fps_time >= 1.0 {
                    let fps = frame_count as f64 / (now - last_fps_time);
                    println!("🎬 渲染FPS: {:.1}", fps);
                    frame_count = 0;
                    last_fps_time = now;
                }
            }
        }

        clear_background(BLACK);

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
        } else {
            draw_text("等待视频流...", 100.0, 100.0, 40.0, WHITE);
        }

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }

    println!("👋 程序退出");
    Ok(())
}

/// 解码线程 - 直接使用FFmpeg API(参考qsv_rtsp_to_mp4.rs)
fn decode_thread(rtsp_url: String, frame_buffer: Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>) {
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

    // 构建过滤器管道 - 接收RGBA帧
    let filter = FrameCapture::new(frame_buffer);

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("capture", Box::new(filter));
    let output = create_null_output().add_frame_pipeline(pipe);

    println!("🔍 使用Intel QSV硬件解码(GPU) + CPU色彩转换(NV12→RGBA)");
    println!("📌 注: QSV VPP不支持RGBA输出,CPU转换不可避免");
    println!("📌 已测试的滤镜:");
    println!("   ❌ hwaccel_output_format(\"rgba\") - 解码器崩溃");
    println!("   ❌ filter_desc(\"scale_qsv=format=rgba\") - 不支持RGBA");
    println!("   ❌ filter_desc(\"hwdownload,format=rgba\") - 卡死");
    println!("   ❌ filter_desc(\"colorspace=range=pc:format=rgb24\") - 语法错误");
    println!("   ❌ filter_desc(\"format=pix_fmts=rgb24\") - 卡死无输出");
    println!("   ✅ hwaccel_output_format(\"nv12\") + CPU转换 - 正常工作");

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
