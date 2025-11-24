/// RTSP主动拉流解码器
/// RTSP active pulling decoder with software decoding only
use super::decode_filter::DecodeFilter;
use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};

/// RTSP解码器
pub struct Decoder {
    rtsp_url: String,
    generation: usize,
    preference: DecoderPreference,
}

impl Decoder {
    /// 创建RTSP解码器
    pub fn new(rtsp_url: String, generation: usize, preference: DecoderPreference) -> Self {
        Self {
            rtsp_url,
            generation,
            preference,
        }
    }

    /// 运行RTSP解码
    pub fn run(&mut self) {
        println!("🎬 RTSP解码器启动 (Gen: {})", self.generation);
        println!("📹 流地址: {}", self.rtsp_url);
        println!("⚙️ 解码偏好: {:?}", self.preference);

        let filter = DecodeFilter::new(self.generation);
        adaptive_decode(&self.rtsp_url, filter, &self.preference);

        println!("❌ RTSP解码器退出");
    }
}

/// 解码器偏好设置 (仅CPU软件解码)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecoderPreference {
    Software,
}

impl DecoderPreference {
    pub fn name(&self) -> &str {
        "CPU软件解码"
    }
}

/// CPU软件解码
fn software_decode(
    rtsp_url: &str,
    mut filter: DecodeFilter,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 使用CPU软件解码");

    filter.decoder_name = "CPU软件解码".to_string();

    // 清除可能存在的硬件加速环境变量
    std::env::remove_var("FFMPEG_HWACCEL");

    // RTSP传输优化
    std::env::set_var("FFMPEG_RTSP_TRANSPORT", "tcp");
    std::env::set_var("FFMPEG_RTSP_FLAGS", "prefer_tcp");
    std::env::set_var("FFMPEG_BUFFER_SIZE", "8192000");

    // 低延迟参数
    std::env::set_var("FFMPEG_FLAGS", "low_delay");
    std::env::set_var("FFMPEG_FFLAGS", "nobuffer");

    // 解码质量优化
    std::env::set_var("FFMPEG_SKIP_FRAME", "noref");
    std::env::set_var("FFMPEG_SKIP_LOOP_FILTER", "noref");
    std::env::set_var("FFMPEG_ERR_DETECT", "careful");

    // 多线程解码
    std::env::set_var("FFMPEG_THREADS", "auto");
    std::env::set_var("FFMPEG_THREAD_TYPE", "frame+slice");

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("decode", Box::new(filter));
    let out = create_null_output().add_frame_pipeline(pipe);

    let input = Input::new(rtsp_url).set_input_opts(
        [
            ("rtsp_transport", "tcp"),
            ("buffer_size", "67108864"),
            ("rtsp_flags", "prefer_tcp"),
            // ("thread", "4"),
            // ("thread_queue_size", "1024"),
        ]
        .into(),
    );
    // 构建FFmpeg上下文
    let ctx = FfmpegContext::builder()
        .input(input)
        .filter_descs(["scale=1920x1080"].into()) // 让FFmpeg用sws_scale转换YUV→RGBA
        .output(out)
        .build()
        .map_err(|e| format!("构建失败: {}", e))?;

    let sch = ctx.start().map_err(|e| format!("启动失败: {}", e))?;
    println!("✅ CPU软件解码启动成功");

    let _ = sch.wait();
    Ok(())
}

/// CPU软件解码(简化版)
pub fn adaptive_decode(rtsp_url: &str, filter: DecodeFilter, _preference: &DecoderPreference) {
    println!("🔄 解码策略: CPU软件解码");

    match software_decode(rtsp_url, filter) {
        Ok(_) => {
            println!("✅ 解码线程正常退出");
        }
        Err(e) => {
            eprintln!("❌ CPU软件解码失败: {}", e);
        }
    }
}
