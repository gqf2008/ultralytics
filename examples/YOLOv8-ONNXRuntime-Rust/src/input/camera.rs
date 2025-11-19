//! 摄像头输入模块 - 独立的摄像头解码器
//!
//! 处理本地摄像头输入,支持 DirectShow(Windows) / AVFoundation(macOS) / V4L2(Linux)

use super::decode_filter::DecodeFilter;
use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};

/// 摄像头解码器结构
pub struct CameraDecoder {
    device_index: usize,
    device_name: String,
    generation: usize,
}

impl CameraDecoder {
    /// 创建新的摄像头解码器
    pub fn new(device_index: usize, device_name: String, generation: usize) -> Self {
        Self {
            device_index,
            device_name,
            generation,
        }
    }

    /// 启动摄像头解码
    pub fn run(&mut self) {
        println!(
            "\n🎥 ============ 摄像头解码器 (Gen: {}) ============",
            self.generation
        );
        println!("📷 设备索引: {}", self.device_index);
        println!("📷 设备名称: {}", self.device_name);

        let camera_url = Self::format_camera_url(self.device_index, &self.device_name);
        println!("🔗 摄像头URL: {}", camera_url);

        // 创建解码滤镜
        let filter = DecodeFilter::new(self.generation);

        // 开始解码
        Self::decode_camera(&camera_url, filter);
    }

    /// 格式化摄像头URL - 根据平台选择
    fn format_camera_url(_index: usize, name: &str) -> String {
        #[cfg(target_os = "windows")]
        {
            format!("video={}", name)
        }
        #[cfg(target_os = "macos")]
        {
            format!("{}", index)
        }
        #[cfg(target_os = "linux")]
        {
            format!("/dev/video{}", index)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            format!("{}", index)
        }
    }

    /// 摄像头解码实现
    fn decode_camera(camera_input: &str, filter: DecodeFilter) {
        println!("📹 启动摄像头解码");

        #[cfg(target_os = "windows")]
        let format = "dshow"; // DirectShow

        #[cfg(target_os = "macos")]
        let format = "avfoundation"; // AVFoundation

        #[cfg(target_os = "linux")]
        let format = "v4l2"; // Video4Linux2

        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        let format = "video4linux2"; // 默认

        let mut retry_count = 0;
        let max_retries = 3;

        println!("🔍 使用格式: {}, 输入: {}", format, camera_input);

        loop {
            // 构建帧处理管线
            let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
            let pipe = pipe.filter("decode", Box::new(filter.clone()));
            let out = create_null_output().add_frame_pipeline(pipe);

            // 配置摄像头输入
            // 注意: 移除硬编码的分辨率和帧率,让 dshow 自动协商默认值
            // 很多摄像头不支持特定的 1280x720 或 30fps, 导致打开失败
            let input = Input::new(camera_input)
                .set_format(format)
                .set_input_opts([("framerate", "30"), ("video_size", "1280x720")].into());

            // 构建FFmpeg上下文
            let ctx_result = FfmpegContext::builder().input(input).output(out).build();

            let ctx = match ctx_result {
                Ok(c) => c,
                Err(e) => {
                    retry_count += 1;
                    eprintln!("❌ 摄像头构建错误详情: {}", e);
                    if retry_count >= max_retries {
                        eprintln!("❌ 摄像头构建失败 (重试{}次)", max_retries);
                        eprintln!("💡 提示: 请检查设备名称是否正确,或尝试关闭其他占用摄像头的程序");
                        return;
                    }
                    println!(
                        "⚠️ 摄像头忙或无法打开, 1秒后重试... ({}/{})",
                        retry_count, max_retries
                    );
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
            };

            // 启动并运行解码循环
            let sch = match ctx.start() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("❌ 摄像头启动失败: {}", e);
                    return;
                }
            };

            println!("✅ 摄像头连接成功,开始解码!");

            // 等待解码完成
            let _ = sch.wait();
            println!("📹 摄像头解码循环结束");
            break;
        }
    }
}

/// 获取可用的摄像头设备列表
pub fn get_camera_devices() -> Vec<(usize, String)> {
    match ez_ffmpeg::device::get_input_video_devices() {
        Ok(devices) => devices.into_iter().enumerate().collect(),
        Err(e) => {
            eprintln!("⚠️ 获取摄像头列表失败: {}", e);
            vec![]
        }
    }
}
