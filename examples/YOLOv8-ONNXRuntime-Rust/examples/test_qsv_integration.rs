/// 测试 QSV 解码器集成到 input 模块
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use yolov8_rs::detection::types::DecodedFrame;
use yolov8_rs::input::decoder::DecoderPreference;
use yolov8_rs::input::{switch_decoder_source, InputSource};
use yolov8_rs::xbus;

fn main() {
    println!("🧪 测试 QSV RTSP 解码器集成");

    let rtsp_url = std::env::args().nth(1).unwrap_or_else(|| {
        "rtsp://admin:sqb12345@172.19.42.187/cam/realmonitor?channel=1&subtype=0".to_string()
    });

    println!("📡 RTSP URL: {}", rtsp_url);

    // 统计计数器
    let frame_count = Arc::new(AtomicU64::new(0));
    let frame_count_clone = frame_count.clone();

    // 订阅解码帧
    let _sub = xbus::subscribe::<DecodedFrame, _>(move |frame| {
        let count = frame_count_clone.fetch_add(1, Ordering::SeqCst) + 1;

        if count % 30 == 0 {
            println!(
                "✅ 帧 #{}: {}x{} | 解码器: {} | FPS: {:.1}",
                count, frame.width, frame.height, frame.decoder_name, frame.decode_fps
            );
        }
    });

    // 启动 QSV 解码器
    println!("🚀 启动 QSV 解码器...\n");
    switch_decoder_source(InputSource::Rtsp(rtsp_url), DecoderPreference::Qsv);

    // 等待接收帧
    let start_time = std::time::Instant::now();

    loop {
        std::thread::sleep(Duration::from_secs(1));

        let count = frame_count.load(Ordering::SeqCst);
        let elapsed = start_time.elapsed().as_secs_f64();

        if count >= 300 {
            println!("\n🎉 成功接收 300 帧,测试通过!");
            println!("\n📈 统计:");
            println!("   总帧数: {}", count);
            println!("   总时长: {:.1}s", elapsed);
            println!("   平均FPS: {:.1}", count as f64 / elapsed);
            break;
        }

        if elapsed > 30.0 && count == 0 {
            eprintln!("\n⚠️ 30秒内未收到帧,测试失败");
            break;
        }
    }
}
