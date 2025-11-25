use ez_ffmpeg::{FfmpegContext, Input, Output};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Intel QSV硬件加速示例启动");

    // RTSP流地址(可通过命令行参数指定)
    let rtsp_url = env::args().nth(1).unwrap_or_else(|| {
        "rtsp://admin:sqb12345@172.19.42.187/cam/realmonitor?channel=1subtype=0".to_string()
    });

    // 输出文件路径
    let output_path = env::args()
        .nth(2)
        .unwrap_or_else(|| "output_qsv.mp4".to_string());

    println!("📹 RTSP地址: {}", rtsp_url);
    println!("💾 输出文件: {}", output_path);

    // 构建输入 - 启用QSV硬件解码
    let input = Input::new(&rtsp_url)
        .set_hwaccel("qsv") // 启用QSV硬件加速
        .set_hwaccel_output_format("nv12") // QSV输出NV12格式(可用vpp_qsv转RGBA)
        .set_video_codec("hevc_qsv") // 使用HEVC QSV解码器
        .set_input_opts(
            [
                ("rtsp_transport", "tcp"),
                ("buffer_size", "67108864"),
                ("rtsp_flags", "prefer_tcp"),
            ]
            .into(),
        );
    // 构建输出 - QSV硬件编码(高兼容性配置)
    let output = Output::new(&output_path)
        .set_video_codec("h264_qsv") // 使用H264 QSV编码器
        .set_video_codec_opts(
            [
                ("preset", "medium"),     // 使用medium预设(平衡质量和速度)
                ("global_quality", "23"), // QSV质量控制(18-28,越小越好)
                ("look_ahead", "0"),      // 禁用前瞻
                ("profile", "high"),      // H.264 High Profile(广泛兼容)
                ("level", "4.1"),         // H.264 Level 4.1
                ("async_depth", "1"),     // 异步深度
            ]
            .into(),
        )
        .set_format_opts(
            [
                ("movflags", "+faststart+frag_keyframe"), // 快速启动+分片关键帧
                ("frag_duration", "1000000"),             // 分片时长1秒
            ]
            .into(),
        );

    println!("⚙️ 开始处理...");
    println!("💡 提示: 按Ctrl+C停止录制");
    println!("🎨 QSV Pipeline: HEVC解码(GPU) -> NV12 -> H.264编码(GPU)");
    println!("📝 注: NV12->RGBA转换可用vpp_qsv滤镜在GPU完成(零CPU消耗)");

    // 构建并运行FFmpeg上下文
    let ctx = FfmpegContext::builder()
        .input(input)
        .output(output)
        .build()?;

    // 运行(会阻塞直到流结束或出错)
    match ctx.start()?.wait() {
        Ok(_) => {
            println!("📁 文件已保存: {}", output_path);
        }
        Err(e) => {
            eprintln!("❌ 处理出错: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
