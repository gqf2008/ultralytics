/// 测试设备列表获取功能
use yolov8_rs::input::get_video_devices;

fn main() {
    println!("🔍 开始扫描视频设备...\n");
    
    let devices = get_video_devices();
    
    if devices.is_empty() {
        println!("⚠️  未找到任何视频设备");
    } else {
        println!("✅ 找到 {} 个视频设备:\n", devices.len());
        for device in &devices {
            println!("  📹 [{}] {}", device.index, device.name);
        }
    }
    
    println!("\n✅ 设备扫描完成");
}
