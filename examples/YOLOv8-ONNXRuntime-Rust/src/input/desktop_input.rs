/// 桌面捕获输入 (基于 ffmpeg-next desktop_decoder)
use crate::detection::types::DecodedFrame;
use crate::video::{DesktopConfig, DesktopDecoder};
use crate::xbus;
use std::sync::Arc;

pub struct DesktopInput {
    generation: usize,
}

impl DesktopInput {
    pub fn new(generation: usize) -> Self {
        Self { generation }
    }

    pub fn run(&mut self) {
        println!("🖥️ 桌面捕获启动 (Gen: {})", self.generation);

        let config = DesktopConfig::default();

        let mut decoder = match DesktopDecoder::new(config) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("❌ 创建桌面捕获解码器失败: {}", e);
                return;
            }
        };

        let (width, height) = decoder.video_info();
        let mut rgba_buffer = Vec::with_capacity(1920 * 1080 * 4);
        let mut frame_count = 0u64;
        let mut last_fps_time = std::time::Instant::now();
        let mut fps_counter = 0u64;
        let mut current_fps = 0.0;

        loop {
            // 检查是否应该退出
            if should_stop_generation(self.generation) {
                println!("🛑 桌面捕获退出 (Gen: {})", self.generation);
                break;
            }

            match decoder.decode_next_frame() {
                Ok(Some(video_frame)) => {
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
                        fps_counter = 0;
                        last_fps_time = std::time::Instant::now();
                    }

                    // 通过 xbus 发送 (使用 Arc 包装交换缓冲区)
                    let capacity = rgba_buffer.capacity();
                    let rgba_arc = Arc::new(std::mem::replace(
                        &mut rgba_buffer,
                        Vec::with_capacity(capacity),
                    ));

                    let decoded = DecodedFrame {
                        rgba_data: rgba_arc,
                        width,
                        height,
                        decode_fps: current_fps,
                        decoder_name: "桌面捕获".to_string(),
                    };

                    xbus::post(decoded);

                    frame_count += 1;
                    if frame_count % 300 == 0 {
                        println!("🖥️ 已处理 {} 帧 (Gen: {})", frame_count, self.generation);
                    }
                }
                Ok(None) => {
                    println!("⚠️ 桌面捕获流结束 (Gen: {})", self.generation);
                    break;
                }
                Err(e) => {
                    eprintln!("❌ 桌面捕获解码错误: {} (Gen: {})", e, self.generation);
                    break;
                }
            }
        }

        println!("⏹️ 桌面捕获退出 (Gen: {})", self.generation);
    }
}

fn should_stop_generation(generation: usize) -> bool {
    use crate::input::ACTIVE_DECODER_GENERATION;
    use std::sync::atomic::Ordering;

    let current = ACTIVE_DECODER_GENERATION.load(Ordering::SeqCst);
    current != generation
}
