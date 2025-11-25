use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_filter::FrameFilter;
use ez_ffmpeg::filter::frame_filter_context::FrameFilterContext;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::Frame;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};
use macroquad::prelude::*;
use std::sync::mpsc::{sync_channel, SyncSender, TryRecvError};
use std::thread;

/// NV12帧数据
struct NV12Frame {
    y_data: Vec<u8>,
    uv_data: Vec<u8>,
    width: u32,
    height: u32,
}

/// 帧捕获过滤器
struct FrameCapture {
    sender: SyncSender<NV12Frame>,
    width: u32,
    height: u32,
    frame_count: usize,
}

impl FrameCapture {
    fn new(sender: SyncSender<NV12Frame>) -> Self {
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

            if self.frame_count == 0 {
                self.width = w;
                self.height = h;
            }

            // NV12格式 (format=23)
            if format == 23 {
                let y_plane = (*frame.as_ptr()).data[0];
                let uv_plane = (*frame.as_ptr()).data[1];
                let y_stride = (*frame.as_ptr()).linesize[0] as usize;
                let uv_stride = (*frame.as_ptr()).linesize[1] as usize;

                if !y_plane.is_null() && !uv_plane.is_null() {
                    // 直接复制NV12数据,不转换
                    // Y平面: width x height (单通道)
                    let y_size = (w as usize) * (h as usize);
                    // UV平面: (width/2) x (height/2) x 2 = width x height / 2 (双通道交错)
                    let uv_size = (w as usize) * (h as usize) / 2;

                    let mut y_data = vec![0u8; y_size];
                    let mut uv_data = vec![0u8; uv_size];

                    // 逐行复制Y平面
                    for y in 0..(h as usize) {
                        let src = y_plane.add(y * y_stride);
                        let dst = y_data.as_mut_ptr().add(y * w as usize);
                        std::ptr::copy_nonoverlapping(src, dst, w as usize);
                    }

                    // 逐行复制UV平面
                    for y in 0..(h as usize / 2) {
                        let src = uv_plane.add(y * uv_stride);
                        let dst = uv_data.as_mut_ptr().add(y * w as usize);
                        std::ptr::copy_nonoverlapping(src, dst, w as usize);
                    }

                    let nv12_frame = NV12Frame {
                        y_data,
                        uv_data,
                        width: w,
                        height: h,
                    };

                    let _ = self.sender.try_send(nv12_frame);

                    self.frame_count += 1;
                    if self.frame_count == 1 {
                        println!("🎨 NV12格式 - GPU渲染模式");
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

// NV12到RGB的Shader
const NV12_VERTEX_SHADER: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
varying lowp vec2 uv;

void main() {
    gl_Position = vec4(position, 1);
    uv = texcoord;
}"#;

const NV12_FRAGMENT_SHADER: &str = r#"#version 100
precision mediump float;

varying vec2 uv;
uniform sampler2D y_texture;
uniform sampler2D uv_texture;

void main() {
    // 从RGBA纹理中提取Y值(我们把Y存在了R通道)
    float y = texture2D(y_texture, uv).r / 255.0;
    
    // 从RGBA纹理中提取UV值(U在R通道,V在G通道)
    vec4 uv_rgba = texture2D(uv_texture, uv);
    float u = (uv_rgba.r / 255.0) - 0.5;
    float v = (uv_rgba.g / 255.0) - 0.5;
    
    // BT.601 YUV到RGB转换
    float r = y + 1.402 * v;
    float g = y - 0.344 * u - 0.714 * v;
    float b = y + 1.772 * u;
    
    gl_FragColor = vec4(r, g, b, 1.0);
}"#;

/// QSV硬件解码RTSP流并GPU渲染
#[macroquad::main("QSV RTSP GPU渲染")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 QSV硬件解码 + GPU NV12渲染");

    let rtsp_url = std::env::args().nth(1).unwrap_or_else(|| {
        "rtsp://admin:sqb12345@172.19.42.187/cam/realmonitor?channel=1subtype=0".to_string()
    });

    println!("📹 RTSP地址: {}", rtsp_url);
    println!("🎨 QSV Pipeline: HEVC解码(GPU) → NV12 → GPU Shader渲染");
    println!("📝 零CPU转换,极低内存占用");

    let (tx, rx) = sync_channel::<NV12Frame>(2);

    thread::spawn(move || {
        decode_thread(rtsp_url, tx);
    });

    // 创建自定义Shader材质
    let material = load_material(
        ShaderSource::Glsl {
            vertex: NV12_VERTEX_SHADER,
            fragment: NV12_FRAGMENT_SHADER,
        },
        MaterialParams {
            textures: vec!["y_texture".to_string(), "uv_texture".to_string()],
            ..Default::default()
        },
    )
    .unwrap();

    let mut y_texture: Option<Texture2D> = None;
    let mut uv_texture: Option<Texture2D> = None;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut frame_count = 0u64;
    let mut last_fps_time = get_time();

    println!("✅ 解码线程已启动,等待第一帧...");

    loop {
        // 非阻塞接收最新帧
        let mut latest_frame: Option<NV12Frame> = None;

        loop {
            match rx.try_recv() {
                Ok(nv12_frame) => {
                    latest_frame = Some(nv12_frame);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    println!("⚠️ 解码线程已退出");
                    return Ok(());
                }
            }
        }

        // 更新纹理
        if let Some(nv12_frame) = latest_frame {
            width = nv12_frame.width;
            height = nv12_frame.height;

            // Y纹理: 扩展为RGBA格式 (R=Y, G=Y, B=Y, A=255)
            if y_texture.is_none() {
                let mut rgba_y = Vec::with_capacity((width * height * 4) as usize);
                for &y in &nv12_frame.y_data {
                    rgba_y.extend_from_slice(&[y, y, y, 255]);
                }
                y_texture = Some(Texture2D::from_rgba8(width as u16, height as u16, &rgba_y));
                y_texture.as_mut().unwrap().set_filter(FilterMode::Linear);
                println!("✅ 创建Y纹理: {}x{}", width, height);
            } else {
                let mut rgba_y = Vec::with_capacity((width * height * 4) as usize);
                for &y in &nv12_frame.y_data {
                    rgba_y.extend_from_slice(&[y, y, y, 255]);
                }
                y_texture.as_ref().unwrap().update(&Image {
                    bytes: rgba_y,
                    width: width as u16,
                    height: height as u16,
                });
            }

            // UV纹理: 扩展为RGBA格式 (R=U, G=V, B=0, A=255)
            if uv_texture.is_none() {
                let mut rgba_uv = Vec::with_capacity(((width / 2) * (height / 2) * 4) as usize);
                for chunk in nv12_frame.uv_data.chunks_exact(2) {
                    rgba_uv.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
                }
                uv_texture = Some(Texture2D::from_rgba8(
                    (width / 2) as u16,
                    (height / 2) as u16,
                    &rgba_uv,
                ));
                uv_texture.as_mut().unwrap().set_filter(FilterMode::Linear);
                println!("✅ 创建UV纹理: {}x{}", width / 2, height / 2);
            } else {
                let mut rgba_uv = Vec::with_capacity(((width / 2) * (height / 2) * 4) as usize);
                for chunk in nv12_frame.uv_data.chunks_exact(2) {
                    rgba_uv.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
                }
                uv_texture.as_ref().unwrap().update(&Image {
                    bytes: rgba_uv,
                    width: (width / 2) as u16,
                    height: (height / 2) as u16,
                });
            }

            frame_count += 1;

            let now = get_time();
            if now - last_fps_time >= 1.0 {
                let fps = frame_count as f64 / (now - last_fps_time);
                println!("🎬 GPU渲染FPS: {:.1}", fps);
                frame_count = 0;
                last_fps_time = now;
            }
        }

        clear_background(BLACK);

        // GPU渲染
        if let (Some(ref y_tex), Some(ref uv_tex)) = (&y_texture, &uv_texture) {
            let screen_w = screen_width();
            let screen_h = screen_height();
            let tex_w = width as f32;
            let tex_h = height as f32;

            let scale = (screen_w / tex_w).min(screen_h / tex_h);
            let draw_w = tex_w * scale;
            let draw_h = tex_h * scale;
            let x = (screen_w - draw_w) * 0.5;
            let y = (screen_h - draw_h) * 0.5;

            // 使用自定义材质渲染Y纹理(实际会在Shader中组合Y和UV)
            gl_use_material(&material);
            material.set_texture("y_texture", y_tex.clone());
            material.set_texture("uv_texture", uv_tex.clone());

            draw_texture_ex(
                y_tex,
                x,
                y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(draw_w, draw_h)),
                    ..Default::default()
                },
            );

            gl_use_default_material();

            draw_text(
                &format!("分辨率: {}x{}", width, height),
                10.0,
                30.0,
                30.0,
                GREEN,
            );
            draw_text("QSV硬件解码 + GPU渲染", 10.0, 60.0, 30.0, GREEN);
            draw_text("零CPU转换", 10.0, 90.0, 30.0, GREEN);
            draw_text("按ESC退出", 10.0, 120.0, 30.0, GREEN);
        }

        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        next_frame().await;
    }

    println!("👋 程序退出");
    Ok(())
}

fn decode_thread(rtsp_url: String, sender: SyncSender<NV12Frame>) {
    println!("🔧 初始化QSV解码器...");

    let input = Input::new(&rtsp_url)
        .set_hwaccel("qsv")
        .set_hwaccel_output_format("nv12")
        .set_video_codec("hevc_qsv")
        .set_input_opts(
            [
                ("rtsp_transport", "tcp"),
                ("buffer_size", "67108864"),
                ("rtsp_flags", "prefer_tcp"),
            ]
            .into(),
        );

    let filter = FrameCapture::new(sender);

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("capture", Box::new(filter));
    let output = create_null_output().add_frame_pipeline(pipe);

    println!("🔍 QSV硬件解码(GPU) + GPU直接渲染NV12");
    println!("📌 内存优化: 无CPU转换,最小内存占用");

    match FfmpegContext::builder().input(input).output(output).build() {
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
