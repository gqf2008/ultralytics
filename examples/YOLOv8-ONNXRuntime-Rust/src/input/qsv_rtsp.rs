/// QSV RTSP 解码器 (双线程架构,Rust风格封装)
use crate::video::{DecoderConfig, HardwareAccel, QsvDecoder, RtspTransport, VideoFrame};
use crossbeam::channel::{bounded, Receiver, Sender};
use std::thread;

/// QSV RTSP 解码器
pub struct QsvRtspDecoder {
    rtsp_url: String,
    generation: usize,
}

impl QsvRtspDecoder {
    /// 创建 QSV RTSP 解码器
    pub fn new(rtsp_url: String, generation: usize) -> Self {
        Self {
            rtsp_url,
            generation,
        }
    }

    /// 运行 QSV RTSP 解码 (双线程架构)
    pub fn run(&mut self) {
        println!("🎬 QSV RTSP 解码器启动 (Gen: {})", self.generation);
        println!("📹 流地址: {}", self.rtsp_url);
        println!("⚡ 架构: 双线程 (解码 + 处理)");

        let rtsp_url = self.rtsp_url.clone();
        let generation = self.generation;

        // 创建 crossbeam channel (缓冲 1 帧)
        let (tx, rx): (Sender<VideoFrame>, Receiver<VideoFrame>) = bounded(1);

        // 启动解码线程
        thread::spawn(move || {
            decode_thread(rtsp_url, tx, generation);
        });

        // 主线程处理帧
        process_thread(rx, generation);
    }
}

/// 解码线程 - 使用 Rust 风格封装的 QsvDecoder
fn decode_thread(rtsp_url: String, sender: Sender<VideoFrame>, generation: usize) {
    println!("🔄 解码线程启动 (Gen: {})", generation);

    // 配置解码器
    let config = DecoderConfig {
        url: rtsp_url,
        hardware_accel: HardwareAccel::Qsv,
        rtsp_transport: RtspTransport::Tcp,
        buffer_size: 5 * 1024 * 1024,
        max_delay: 0,
    };

    // 创建解码器
    let mut decoder = match QsvDecoder::new(config) {
        Ok(d) => {
            let info = d.video_info();
            println!(
                "✅ QSV 解码器初始化成功: {}x{}, {}",
                info.width, info.height, info.pixel_format
            );
            d
        }
        Err(e) => {
            eprintln!("❌ 创建解码器失败: {}", e);
            return;
        }
    };

    let mut frame_count = 0u64;

    // 解码循环
    loop {
        // 检查是否应该退出
        if should_stop_generation(generation) {
            println!("🛑 解码线程收到退出信号 (Gen: {})", generation);
            break;
        }

        match decoder.decode_next_frame() {
            Ok(Some(video_frame)) => {
                // 非阻塞发送,自动丢弃旧帧
                let _ = sender.try_send(video_frame);

                frame_count += 1;
                if frame_count % 300 == 0 {
                    println!("📺 解码线程已解码 {} 帧 (Gen: {})", frame_count, generation);
                }
            }
            Ok(None) => {
                println!("⚠️ 流结束 (Gen: {})", generation);
                break;
            }
            Err(e) => {
                eprintln!("❌ 解码错误: {} (Gen: {})", e, generation);
                break;
            }
        }
    }

    println!("⏹️ 解码线程退出 (Gen: {})", generation);
}

/// 处理线程 - 转换并通过 xbus 发送帧
fn process_thread(receiver: Receiver<VideoFrame>, generation: usize) {
    println!("🔄 处理线程启动 (Gen: {})", generation);

    use crate::detection::types::DecodedFrame;
    use crate::xbus;
    use std::sync::Arc;

    let mut frame_count = 0u64;
    let mut rgba_buffer = Vec::with_capacity(1920 * 1080 * 4);
    let mut last_fps_time = std::time::Instant::now();
    let mut fps_counter = 0u64;
    let mut current_fps = 0.0;

    loop {
        // 检查是否应该退出
        if should_stop_generation(generation) {
            println!("🛑 处理线程收到退出信号 (Gen: {})", generation);
            break;
        }

        // 接收帧 (阻塞等待,最多 100ms)
        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(video_frame) => {
                // NV12 → RGBA 转换
                unsafe {
                    crate::utils::color_convert::nv12_to_rgba_simd_into(
                        video_frame.y_data.as_ptr(),
                        video_frame.uv_data.as_ptr(),
                        video_frame.width,
                        video_frame.height,
                        video_frame.y_stride,
                        video_frame.uv_stride,
                        &mut rgba_buffer,
                    );
                }

                // 计算 FPS
                fps_counter += 1;
                if last_fps_time.elapsed().as_secs_f64() >= 1.0 {
                    let elapsed = last_fps_time.elapsed().as_secs_f64();
                    current_fps = fps_counter as f64 / elapsed;
                    println!("📺 QSV处理: {:.1} fps (Gen: {})", current_fps, generation);
                    fps_counter = 0;
                    last_fps_time = std::time::Instant::now();
                }

                // 通过 xbus 发送解码帧 (使用 Arc 包装交换缓冲区)
                let capacity = rgba_buffer.capacity();
                let rgba_arc = Arc::new(std::mem::replace(
                    &mut rgba_buffer,
                    Vec::with_capacity(capacity),
                ));
                
                let decoded = DecodedFrame {
                    rgba_data: rgba_arc,
                    width: video_frame.width,
                    height: video_frame.height,
                    decode_fps: current_fps,
                    decoder_name: "Intel QSV (Rust封装)".to_string(),
                };

                xbus::post(decoded);

                frame_count += 1;
                if frame_count % 300 == 0 {
                    println!("📊 处理线程已处理 {} 帧 (Gen: {})", frame_count, generation);
                }
            }
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                // 超时,继续等待
                continue;
            }
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                println!("📡 解码线程已断开 (Gen: {})", generation);
                break;
            }
        }
    }

    println!("⏹️ 处理线程退出 (Gen: {})", generation);
}

/// 检查当前代数是否应该停止
fn should_stop_generation(generation: usize) -> bool {
    use super::decoder_manager::ACTIVE_DECODER_GENERATION;
    use std::sync::atomic::Ordering;

    let current = ACTIVE_DECODER_GENERATION.load(Ordering::SeqCst);
    current != generation
}
