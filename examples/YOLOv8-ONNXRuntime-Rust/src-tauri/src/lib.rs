// Tauri library exports
// 这个文件用于导出库接口,但我们的主要逻辑在 main.rs (二进制入口)
// 如果需要导出库函数,在这里添加

// pub mod shared_memory;  // TODO: 暂时禁用，后续实现

// Windows Graphics Capture 模块
#[cfg(windows)]
pub mod wgc_capture;

#[cfg(target_os = "android")]
pub use mobile::*;

#[cfg(target_os = "android")]
mod mobile {
    // Android 特定导出
}
