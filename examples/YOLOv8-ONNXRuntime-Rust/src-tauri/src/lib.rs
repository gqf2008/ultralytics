// Tauri library exports
// 这个文件用于导出库接口,但我们的主要逻辑在 main.rs (二进制入口)

#[cfg(target_os = "android")]
pub use mobile::*;

#[cfg(target_os = "android")]
mod mobile {
    // Android 特定导出
}
