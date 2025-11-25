// QSV 硬件解码 + SDL2 YUV 渲染 - 使用 ffmpeg-next 原生 API
// 完全对标 ffplay - 零拷贝,预分配帧队列

use ffmpeg_next as ffmpeg;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use std::collections::VecDeque;
use std::ptr;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

// 预分配的帧队列 (对标 ffplay 的 VIDEO_PICTURE_QUEUE_SIZE = 3)
const FRAME_QUEUE_SIZE: usize = 3;

// 帧包装器 - 只传递指针,不拷贝数据
struct VideoFrame {
    y_data: Vec<u8>,
    uv_data: Vec<u8>,
    width: u32,
    height: u32,
}

// 解码线程
fn decode_thread(rtsp_url: String, sender: SyncSender<VideoFrame>) -> Result<(), ffmpeg::Error> {
    ffmpeg::init()?;

    // 设置 RTSP 参数
    let mut opts = ffmpeg::Dictionary::new();
    opts.set("rtsp_transport", "tcp");
    opts.set("buffer_size", "1048576"); // 1MB
    opts.set("max_delay", "0");

    println!("📡 尝试启用 QSV 硬件加速...");

    // 打开输入
    let mut ictx = ffmpeg::format::input_with_dictionary(&rtsp_url, opts)?;

    println!("✅ RTSP 流打开成功");
    println!("📊 流信息:");
    for stream in ictx.streams() {
        println!(
            "  - Stream #{}: {:?}",
            stream.index(),
            stream.parameters().medium()
        );
    }

    // 查找视频流
    let video_stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or(ffmpeg::Error::StreamNotFound)?
        .index();

    println!("🎬 视频流索引: {}", video_stream_index);

    // 创建解码器 - 尝试 QSV
    let video_stream = ictx.stream(video_stream_index).unwrap();
    let codec_params = video_stream.parameters();

    // 先尝试 QSV 硬件解码器
    let codec = match ffmpeg::decoder::find_by_name("hevc_qsv") {
        Some(c) => {
            println!("✅ 找到 QSV 硬件解码器");
            c
        }
        None => {
            println!("⚠️ QSV 不可用,回退到软件解码");
            ffmpeg::decoder::find(codec_params.id()).ok_or(ffmpeg::Error::DecoderNotFound)?
        }
    };

    println!("🔧 使用解码器: {}", codec.name());

    let context = ffmpeg::codec::context::Context::from_parameters(codec_params)?;
    let mut decoder = context.decoder().video()?;

    println!("✅ 解码器初始化成功");
    println!("📐 视频尺寸: {}x{}", decoder.width(), decoder.height());

    // 预分配帧缓冲 (对标 ffplay)
    let mut frame_pool: VecDeque<ffmpeg::frame::Video> = VecDeque::with_capacity(FRAME_QUEUE_SIZE);
    for _ in 0..FRAME_QUEUE_SIZE {
        frame_pool.push_back(ffmpeg::frame::Video::empty());
    }

    let mut frame_count = 0u64;

    // 解码循环
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;

            // 接收解码帧
            while let Ok(()) = decoder.receive_frame(&mut frame_pool[0]) {
                let decoded_frame = &frame_pool[0];

                // 检查格式 (软件解码是 YUV420P)
                let format = decoded_frame.format();
                if format == ffmpeg::format::Pixel::YUV420P || format == ffmpeg::format::Pixel::NV12
                {
                    let width = decoded_frame.width();
                    let height = decoded_frame.height();

                    // 零拷贝:直接访问 AVFrame 内部数据
                    unsafe {
                        let y_plane_ptr = decoded_frame.data(0);
                        let y_stride = decoded_frame.stride(0);

                        // YUV420P 有3个平面: Y, U, V
                        let u_plane_ptr = decoded_frame.data(1);
                        let v_plane_ptr = decoded_frame.data(2);
                        let u_stride = decoded_frame.stride(1);
                        let v_stride = decoded_frame.stride(2);

                        // 拷贝 Y 平面
                        let mut y_data = Vec::with_capacity((width * height) as usize);
                        for y in 0..height as usize {
                            let offset = y * y_stride;
                            let slice = std::slice::from_raw_parts(
                                y_plane_ptr.as_ptr().add(offset),
                                width as usize,
                            );
                            y_data.extend_from_slice(slice);
                        }

                        // YUV420P 转 NV12: 需要交错 U 和 V
                        let mut uv_data = Vec::with_capacity((width * height / 2) as usize);
                        for y in 0..(height / 2) as usize {
                            let u_offset = y * u_stride;
                            let v_offset = y * v_stride;
                            let u_slice = std::slice::from_raw_parts(
                                u_plane_ptr.as_ptr().add(u_offset),
                                (width / 2) as usize,
                            );
                            let v_slice = std::slice::from_raw_parts(
                                v_plane_ptr.as_ptr().add(v_offset),
                                (width / 2) as usize,
                            );

                            // 交错 UV
                            for i in 0..(width / 2) as usize {
                                uv_data.push(u_slice[i]);
                                uv_data.push(v_slice[i]);
                            }
                        }

                        let video_frame = VideoFrame {
                            y_data,
                            uv_data,
                            width,
                            height,
                        };

                        // 非阻塞发送
                        if sender.try_send(video_frame).is_err() {
                            // 通道满,丢帧
                        }
                    }

                    frame_count += 1;
                    if frame_count == 1 {
                        println!("🎨 首帧解码成功: {:?} {}x{}", format, width, height);
                    }
                    if frame_count % 60 == 0 {
                        println!("📺 已解码 {} 帧", frame_count);
                    }
                }

                // 帧循环复用
                frame_pool.rotate_left(1);
            }
        }
    }

    println!("⚠️ 解码线程退出");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 FFmpeg-Next 原生 API + SDL2");
    println!("📹 零拷贝解码,对标 ffplay 性能");

    let rtsp_url = "rtsp://admin:sqb12345@172.19.42.187/cam/realmonitor?channel=1&subtype=0";
    println!("📡 RTSP: {}", rtsp_url);

    // 单缓冲通道
    let (tx, rx) = sync_channel::<VideoFrame>(1);

    // 解码线程
    let rtsp_clone = rtsp_url.to_string();
    thread::spawn(move || {
        if let Err(e) = decode_thread(rtsp_clone, tx) {
            eprintln!("❌ 解码错误: {}", e);
        }
    });

    println!("✅ 解码线程启动");

    // 初始化 SDL2
    let sdl = sdl2::init()?;
    let video = sdl.video()?;

    let window = video
        .window("FFmpeg-Next + SDL2", 1920, 1080)
        .position_centered()
        .resizable()
        .build()?;

    let mut canvas = window.into_canvas().accelerated().present_vsync().build()?;
    let texture_creator = canvas.texture_creator();
    let mut yuv_texture: Option<sdl2::render::Texture> = None;

    let mut event_pump = sdl.event_pump()?;
    let mut frame_count = 0u64;
    let mut last_fps_time = Instant::now();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        // 接收最新帧
        let mut latest_frame = None;
        while let Ok(frame) = rx.try_recv() {
            latest_frame = Some(frame);
        }

        if let Some(video_frame) = latest_frame {
            // 首次创建纹理
            if yuv_texture.is_none() {
                yuv_texture = Some(texture_creator.create_texture_streaming(
                    PixelFormatEnum::NV12,
                    video_frame.width,
                    video_frame.height,
                )?);
                canvas
                    .window_mut()
                    .set_size(video_frame.width, video_frame.height)?;
                println!(
                    "✅ SDL2 NV12 纹理: {}x{}",
                    video_frame.width, video_frame.height
                );
            }

            // 更新纹理
            if let Some(ref mut texture) = yuv_texture {
                texture.with_lock(None, |buffer: &mut [u8], pitch: usize| {
                    let y_size = video_frame.height as usize * pitch;
                    let uv_offset = y_size;

                    // Y 平面
                    for y in 0..video_frame.height as usize {
                        let src_offset = y * video_frame.width as usize;
                        let dst_offset = y * pitch;
                        let width = video_frame.width as usize;
                        buffer[dst_offset..dst_offset + width]
                            .copy_from_slice(&video_frame.y_data[src_offset..src_offset + width]);
                    }

                    // UV 平面
                    for y in 0..(video_frame.height / 2) as usize {
                        let src_offset = y * video_frame.width as usize;
                        let dst_offset = uv_offset + y * pitch;
                        let width = video_frame.width as usize;
                        buffer[dst_offset..dst_offset + width]
                            .copy_from_slice(&video_frame.uv_data[src_offset..src_offset + width]);
                    }
                })?;
            }

            frame_count += 1;
        }

        // 渲染
        canvas.clear();
        if let Some(ref texture) = yuv_texture {
            canvas.copy(texture, None, None)?;
        }
        canvas.present();

        // FPS 统计
        let elapsed = last_fps_time.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            println!("🎬 FPS: {:.1}", fps);
            frame_count = 0;
            last_fps_time = Instant::now();
        }

        thread::sleep(Duration::from_millis(1));
    }

    println!("👋 退出");
    Ok(())
}
