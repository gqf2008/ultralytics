//! 桌面捕获模块
//!
//! 处理桌面屏幕捕获,支持 Windows (gdigrab)

use super::decode_filter::DecodeFilter;
use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};

/// 桌面解码器结构
pub struct DesktopDecoder {
    generation: usize,
}

impl DesktopDecoder {
    /// 创建新的桌面解码器
    pub fn new(generation: usize) -> Self {
        Self { generation }
    }

    /// 启动桌面捕获
    pub fn run(&mut self) {
        println!(
            "\n🖥️ ============ 桌面捕获解码器 (Gen: {}) ============",
            self.generation
        );

        // 创建解码滤镜
        let filter = DecodeFilter::new(self.generation);

        // 开始解码
        Self::decode_desktop(filter);
    }

    /// 桌面解码实现
    fn decode_desktop(filter: DecodeFilter) {
        println!("🖥️ 启动桌面捕获");

        #[cfg(target_os = "windows")]
        {
            // 1. 尝试 gdigrab (通常性能更好)
            println!("Trying gdigrab...");
            if Self::try_run_desktop("gdigrab", "desktop", filter.clone()).is_ok() {
                return;
            }

            // 2. 尝试 dshow screen-capture-recorder (如果安装了 OBS 或 screen-capture-recorder)
            println!("⚠️ gdigrab 失败, 尝试 dshow screen-capture-recorder...");
            if Self::try_run_desktop("dshow", "video=screen-capture-recorder", filter).is_ok() {
                return;
            }

            eprintln!("❌ 所有桌面捕获方式均失败");
        }

        #[cfg(not(target_os = "windows"))]
        {
            eprintln!("❌ 桌面捕获目前仅支持 Windows");
        }
    }

    /// 尝试运行桌面捕获
    fn try_run_desktop(format: &str, input_name: &str, filter: DecodeFilter) -> Result<(), String> {
        println!("🔍 尝试: format={}, input={}", format, input_name);

        // 构建帧处理管线
        let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
        let pipe = pipe.filter("decode", Box::new(filter));
        let out = create_null_output().add_frame_pipeline(pipe);

        // 配置输入
        let input = Input::new(input_name)
            .set_format(format)
            .set_input_opts([("framerate", "30"), ("video_size", "1280x720")].into());

        // 构建FFmpeg上下文
        let ctx = FfmpegContext::builder()
            .input(input)
            .output(out)
            .build()
            .map_err(|e| {
                eprintln!("❌ 构建错误详情: {}", e);
                format!("构建失败: {}", e)
            })?;

        // 启动并运行解码循环
        let sch = ctx.start().map_err(|e| {
            eprintln!("❌ 启动错误详情: {}", e);
            format!("启动失败: {}", e)
        })?;

        println!("✅ 桌面捕获连接成功 ({}), 开始解码!", format);

        // 等待解码完成
        let _ = sch.wait();
        println!("🖥️ 桌面捕获循环结束");
        Ok(())
    }
}
