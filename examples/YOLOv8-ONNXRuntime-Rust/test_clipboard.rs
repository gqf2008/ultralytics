// 测试剪贴板功能
#[cfg(windows)]
fn main() {
    use clipboard_win::{formats, set_clipboard};

    let test_text = "测试剪贴板内容: rtsp://admin:password@192.168.1.1/stream";

    println!("📋 尝试复制: {}", test_text);

    match set_clipboard(formats::Unicode, test_text) {
        Ok(_) => {
            println!("✅ 复制成功!");
            println!("💡 现在请在任意应用中按 Ctrl+V 测试");
        }
        Err(e) => {
            eprintln!("❌ 复制失败: {:?}", e);
        }
    }

    // 等待用户测试
    println!("\n按 Enter 继续...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).ok();
}

#[cfg(not(windows))]
fn main() {
    println!("此测试仅适用于 Windows 平台");
}
