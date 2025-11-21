// 构建脚本: 链接FFmpeg依赖库
fn main() {
    // 仅在Windows MSVC环境下添加FFmpeg相关库
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        // Intel QSV (Quick Sync Video) 硬件加速
        println!("cargo:rustc-link-lib=dylib=libmfx");

        // x264 编码器
        println!("cargo:rustc-link-lib=dylib=libx264");

        // OLE 自动化和VFW
        println!("cargo:rustc-link-lib=dylib=oleaut32");
        println!("cargo:rustc-link-lib=dylib=vfw32");

        // Secure Channel (TLS/SSL)
        println!("cargo:rustc-link-lib=dylib=secur32");
    }
}
