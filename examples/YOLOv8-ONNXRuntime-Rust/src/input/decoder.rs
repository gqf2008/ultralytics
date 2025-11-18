/// RTSP主动拉流解码器
/// RTSP active pulling decoder with adaptive hardware detection
use super::decode_filter::DecodeFilter;
use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::{AVMediaType, FfmpegContext, Input};

#[cfg(windows)]
use wmi::{COMLibrary, WMIConnection};

/// 解码器包装器
pub struct Decoder {
    rtsp_url: String,
}

impl Decoder {
    pub fn new(rtsp_url: String) -> Self {
        Self { rtsp_url }
    }

    pub fn run(&mut self) {
        println!("🎬 解码器启动");
        let filter = DecodeFilter::new();
        let rtsp_url = self.rtsp_url.clone();
        adaptive_decode(&rtsp_url, filter);
        println!("❌ 解码器退出");
    }
}

/// 解码器类型
pub enum DecoderType {
    NvidiaCuda, // NVIDIA GPU硬件解码
    IntelQsv,   // Intel QuickSync硬件解码
    AmdAmf,     // AMD GPU硬件解码
    Dxva2,      // Windows DXVA2通用硬件解码
    Software,   // CPU软件解码
}

impl DecoderType {
    pub fn name(&self) -> &str {
        match self {
            DecoderType::NvidiaCuda => "尝试CUDA(若无N卡则软解)",
            DecoderType::IntelQsv => "尝试QuickSync(若无Intel核显则软解)",
            DecoderType::AmdAmf => "尝试AMF(若无AMD卡则软解)",
            DecoderType::Dxva2 => "DXVA2通用硬解",
            DecoderType::Software => "CPU软件解码",
        }
    }

    fn env_vars(&self) -> Vec<(&str, &str)> {
        match self {
            DecoderType::NvidiaCuda => vec![("FFMPEG_HWACCEL", "cuda")],
            DecoderType::IntelQsv => vec![("FFMPEG_HWACCEL", "qsv")],
            DecoderType::AmdAmf => vec![("FFMPEG_HWACCEL", "amf")],
            DecoderType::Dxva2 => vec![("FFMPEG_HWACCEL", "dxva2")],
            DecoderType::Software => vec![], // 无需设置环境变量
        }
    }

    /// 检测硬件是否可用 (使用WMI API)
    fn is_hardware_available(&self) -> bool {
        match self {
            DecoderType::NvidiaCuda => {
                #[cfg(windows)]
                {
                    check_gpu_vendor("nvidia")
                }
                #[cfg(not(windows))]
                {
                    false
                }
            }
            DecoderType::IntelQsv => {
                #[cfg(windows)]
                {
                    check_gpu_vendor("intel")
                }
                #[cfg(not(windows))]
                {
                    false
                }
            }
            DecoderType::AmdAmf => {
                #[cfg(windows)]
                {
                    check_gpu_vendor("amd") || check_gpu_vendor("radeon")
                }
                #[cfg(not(windows))]
                {
                    false
                }
            }
            DecoderType::Dxva2 => {
                // DXVA2在Windows上总是可用
                #[cfg(windows)]
                {
                    true
                }
                #[cfg(not(windows))]
                {
                    false
                }
            }
            DecoderType::Software => true, // 软解总是可用
        }
    }
}

/// Windows平台检测显卡厂商 (使用WMI)
#[cfg(windows)]
fn check_gpu_vendor(vendor: &str) -> bool {
    use serde::Deserialize;

    #[derive(Deserialize)]
    #[allow(non_camel_case_types)]
    struct Win32_VideoController {
        #[serde(rename = "Name")]
        name: String,
    }

    match COMLibrary::new() {
        Ok(com_con) => match WMIConnection::new(com_con) {
            Ok(wmi_con) => {
                if let Ok(gpus) = wmi_con
                    .raw_query::<Win32_VideoController>("SELECT Name FROM Win32_VideoController")
                {
                    for gpu in gpus {
                        if gpu.name.to_lowercase().contains(vendor) {
                            return true;
                        }
                    }
                }
            }
            Err(_) => return false,
        },
        Err(_) => return false,
    }
    false
}

/// 尝试使用指定解码器启动FFmpeg
fn try_decoder(
    rtsp_url: &str,
    decoder: &DecoderType,
    mut filter: DecodeFilter,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 尝试解码器: {}", decoder.name());

    // 先检测硬件是否可用
    if !decoder.is_hardware_available() {
        return Err(format!("硬件不可用").into());
    }
    println!("   ✅ 硬件检测通过");

    // 设置解码器名称
    filter.decoder_name = decoder.name().to_string();

    // 清除之前的环境变量
    std::env::remove_var("FFMPEG_HWACCEL");

    // 设置新的环境变量
    for (key, val) in decoder.env_vars() {
        std::env::set_var(key, val);
    }

    // ========== 🔥 VLC级别画质优化参数 (CSDN优化方案) ==========
    // RTSP传输优化
    std::env::set_var("FFMPEG_RTSP_TRANSPORT", "tcp"); // 强制TCP传输(可靠,防止UDP丢包)
    std::env::set_var("FFMPEG_RTSP_FLAGS", "prefer_tcp");
    std::env::set_var("FFMPEG_BUFFER_SIZE", "8192000"); // 8MB缓冲区

    // 🎯 低延迟参数
    std::env::set_var("FFMPEG_FLAGS", "low_delay"); // 低延迟模式
    std::env::set_var("FFMPEG_FFLAGS", "nobuffer"); // 无缓冲(降低延迟)

    // 🎯 解码质量优化 - 关键! (保留完整环路滤波)
    std::env::set_var("FFMPEG_SKIP_FRAME", "noref"); // 只跳过非参考帧(保留画质)
    std::env::set_var("FFMPEG_SKIP_LOOP_FILTER", "noref"); // 保留环路滤波(去块效应核心)
    std::env::set_var("FFMPEG_ERR_DETECT", "careful"); // 错误检测但不丢弃可修复帧

    // 🔧 H.264/HEVC解码器优化选项
    std::env::set_var("FFMPEG_THREADS", "auto"); // 多线程解码
    std::env::set_var("FFMPEG_THREAD_TYPE", "frame+slice"); // 帧级+切片级并行

    // 🎨 后处理滤镜(去块+降噪)
    std::env::set_var("FFMPEG_POST_PROCESS", "1"); // 启用后处理

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("decode", Box::new(filter));
    let out = create_null_output().add_frame_pipeline(pipe);
    let input = Input::new(rtsp_url).set_input_opts(
        [
            ("rtsp_transport", "tcp"),
            ("buffer_size", "67108864"),
            ("rtsp_flags", "prefer_tcp "),
        ]
        .into(),
    ); //4,194,304
       // 构建FFmpeg上下文 - 添加画质滤镜
    let ctx = FfmpegContext::builder()
        .input(input)
        .filter_descs(["format=yuv420p"].into())
        .output(out)
        .build()
        .map_err(|e| format!("构建失败: {}", e))?;

    // 尝试启动
    let sch = ctx.start().map_err(|e| format!("启动失败: {}", e))?;
    println!("✅ {} 连接成功,开始解码! (画质增强模式)", decoder.name());
    let _ = sch.wait();
    Ok(())
}

/// 自适应解码器选择: 优先硬件,失败则降级
pub fn adaptive_decode(rtsp_url: &str, filter: DecodeFilter) {
    let decoders = vec![
        DecoderType::NvidiaCuda, // 优先NVIDIA (最快)
        DecoderType::IntelQsv,   // 次选Intel
        DecoderType::AmdAmf,     // 再次AMD
        DecoderType::Dxva2,      // 通用硬件解码
        DecoderType::Software,   // 最后软解
    ];

    println!("� 自适应解码器选择 (优先硬件加速)");
    println!("📋 尝试顺序: NVIDIA CUDA > Intel QSV > AMD AMF > DXVA2 > 软件解码");

    for decoder in &decoders {
        match try_decoder(rtsp_url, decoder, filter.clone()) {
            Ok(_) => {
                println!("✅ 解码线程正常退出");
                return;
            }
            Err(e) => {
                println!("⚠️  {} 失败: {}", decoder.name(), e);
                println!("   正在尝试下一个解码器...");
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
    }

    eprintln!("❌ 所有解码器均失败!");
}
