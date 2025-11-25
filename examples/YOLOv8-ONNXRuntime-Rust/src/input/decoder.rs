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

        // 根据偏好选择解码方式
        let result = match self.preference {
            DecoderPreference::Qsv | DecoderPreference::Auto => qsv_decode(&self.rtsp_url, filter),
            _ => software_decode(&self.rtsp_url, filter),
        };

        if let Err(e) = result {
            eprintln!("❌ 解码出错: {}", e);
        }

        println!("❌ RTSP解码器退出");
    }
}

/// 解码器偏好设置
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecoderPreference {
    Software, // CPU软件解码
    Nvdec,    // NVIDIA硬件解码
    Cuvid,    // CUDA视频解码
    Qsv,      // Intel Quick Sync
    Auto,     // 自动选择最优解码器
}

impl DecoderPreference {
    pub fn name(&self) -> &str {
        match self {
            Self::Software => "CPU软件解码",
            Self::Nvdec => "NVDEC硬件解码",
            Self::Cuvid => "CUVID硬件解码",
            Self::Qsv => "Intel QSV硬件解码",
            Self::Auto => "自动检测",
        }
    }

    pub fn decoder_name(&self, codec: &str) -> &str {
        match self {
            Self::Software => match codec {
                "hevc" | "h265" => "hevc",
                _ => "h264",
            },
            Self::Nvdec => match codec {
                "hevc" | "h265" => "hevc_cuvid",
                _ => "h264_cuvid",
            },
            Self::Cuvid => match codec {
                "hevc" | "h265" => "hevc_cuvid",
                _ => "h264_cuvid",
            },
            Self::Qsv | Self::Auto => match codec {
                "hevc" | "h265" => "hevc_qsv", // Intel QSV HEVC硬解码
                _ => "h264_qsv",
            },
        }
    }

    pub fn hwaccel_filter(&self) -> &'static str {
        match self {
            Self::Nvdec | Self::Cuvid => "hwdownload,format=nv12",
            Self::Qsv | Self::Auto => "hwdownload,format=nv12", // QSV先下载为NV12,CPU转换
            Self::Software => "",
        }
    }
}

/// CPU软件解码
fn software_decode(
    rtsp_url: &str,
    mut filter: DecodeFilter,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 使用CPU软件解码");

    filter.decoder_name = "CPU软件解码".to_string();

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("decode", Box::new(filter));
    let out = create_null_output().add_frame_pipeline(pipe);

    let input = Input::new(rtsp_url).set_input_opts(
        [
            ("rtsp_transport", "tcp"),
            ("buffer_size", "67108864"),
            ("rtsp_flags", "prefer_tcp"),
        ]
        .into(),
    );

    let ctx = FfmpegContext::builder().input(input).output(out).build();
    let ctx = match ctx {
        Ok(c) => {
            println!("✅ FFmpeg上下文构建成功");
            c
        }
        Err(e) => {
            eprintln!("❌ FFmpeg上下文构建失败: {:?}", e);
            return Err(format!("构建失败: {:?}", e).into());
        }
    };

    let sch = match ctx.start() {
        Ok(s) => {
            println!("✅ FFmpeg调度器启动成功");
            s
        }
        Err(e) => {
            eprintln!("❌ FFmpeg调度器启动失败: {:?}", e);
            return Err(format!("启动失败: {:?}", e).into());
        }
    };
    println!("✅ CPU软件解码启动成功,开始接收帧...");

    let _ = sch.wait();
    Ok(())
}

/// Intel QSV硬件解码
fn qsv_decode(rtsp_url: &str, mut filter: DecodeFilter) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 使用Intel QSV硬件解码");

    filter.decoder_name = "Intel QSV".to_string();

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("decode", Box::new(filter));
    let out = create_null_output().add_frame_pipeline(pipe);

    let input = Input::new(rtsp_url)
        .set_hwaccel("qsv")
        .set_hwaccel_output_format("yuv420p") // GPU解码后转为YUV420P
        .set_video_codec("hevc_qsv")
        .set_input_opts(
            [
                ("rtsp_transport", "tcp"),
                ("buffer_size", "67108864"),
                ("rtsp_flags", "prefer_tcp"),
            ]
            .into(),
        );

    let ctx = FfmpegContext::builder().input(input).output(out).build();
    let ctx = match ctx {
        Ok(c) => {
            println!("✅ FFmpeg上下文构建成功");
            c
        }
        Err(e) => {
            eprintln!("❌ FFmpeg上下文构建失败: {:?}", e);
            return Err(format!("构建失败: {:?}", e).into());
        }
    };

    let sch = match ctx.start() {
        Ok(s) => {
            println!("✅ FFmpeg调度器启动成功");
            s
        }
        Err(e) => {
            eprintln!("❌ FFmpeg调度器启动失败: {:?}", e);
            return Err(format!("启动失败: {:?}", e).into());
        }
    };
    println!("✅ QSV硬件解码启动成功,开始接收帧...");

    let _ = sch.wait();
    Ok(())
}
