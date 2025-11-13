/// 自适应解码器选择模块
/// Adaptive decoder selection module with hardware detection

use super::decode_filter::DecodeFilter;
use ez_ffmpeg::core::context::null_output::create_null_output;
use ez_ffmpeg::filter::frame_pipeline_builder::FramePipelineBuilder;
use ez_ffmpeg::{AVMediaType, FfmpegContext};

#[cfg(windows)]
use wmi::{COMLibrary, WMIConnection};

/// 解码器类型
pub enum DecoderType {
    NvidiaCuda,    // NVIDIA GPU硬件解码
    IntelQsv,      // Intel QuickSync硬件解码
    AmdAmf,        // AMD GPU硬件解码
    Dxva2,         // Windows DXVA2通用硬件解码
    Software,      // CPU软件解码
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
    struct Win32_VideoController {
        Name: String,
    }
    
    match COMLibrary::new() {
        Ok(com_con) => {
            match WMIConnection::new(com_con) {
                Ok(wmi_con) => {
                    if let Ok(gpus) = wmi_con.raw_query::<Win32_VideoController>(
                        "SELECT Name FROM Win32_VideoController"
                    ) {
                        for gpu in gpus {
                            if gpu.Name.to_lowercase().contains(vendor) {
                                return true;
                            }
                        }
                    }
                }
                Err(_) => return false,
            }
        }
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

    let pipe: FramePipelineBuilder = AVMediaType::AVMEDIA_TYPE_VIDEO.into();
    let pipe = pipe.filter("decode", Box::new(filter));
    let out = create_null_output().add_frame_pipeline(pipe);

    // 尝试构建FFmpeg上下文
    let ctx = FfmpegContext::builder()
        .input(rtsp_url)
        .filter_desc("format=yuv420p")
        .output(out)
        .build()
        .map_err(|e| format!("构建失败: {}", e))?;

    // 尝试启动
    let sch = ctx.start()
        .map_err(|e| format!("启动失败: {}", e))?;
    
    println!("✅ {} 连接成功,开始解码!", decoder.name());
    let _ = sch.wait();
    Ok(())
}

/// 自适应解码器选择: 优先硬件,失败则降级
pub fn adaptive_decode(rtsp_url: &str, filter: DecodeFilter) {
    let decoders = vec![
        DecoderType::NvidiaCuda,  // 优先NVIDIA (最快)
        DecoderType::IntelQsv,    // 次选Intel
        DecoderType::AmdAmf,      // 再次AMD
        DecoderType::Dxva2,       // 通用硬件解码
        DecoderType::Software,    // 最后软解
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
