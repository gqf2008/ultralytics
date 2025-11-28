fn main() {
    tauri_build::build();

    // FFmpeg QSV 硬件加速依赖
    // 使用 x64-windows-static-md 版本 (静态库 + 动态CRT, 与 ONNX Runtime 兼容)
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        let vcpkg_lib = r"D:\workspace\vcpkg\installed\x64-windows-static-md\lib";
        println!("cargo:rustc-link-search=native={}", vcpkg_lib);

        // Windows 系统库
        println!("cargo:rustc-link-lib=gdi32");
        println!("cargo:rustc-link-lib=user32");
        println!("cargo:rustc-link-lib=shell32");
        println!("cargo:rustc-link-lib=ole32");
        println!("cargo:rustc-link-lib=oleaut32");
        println!("cargo:rustc-link-lib=vfw32");
        println!("cargo:rustc-link-lib=secur32");
        println!("cargo:rustc-link-lib=ws2_32");
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=bcrypt");

        // Intel Media SDK (QSV) 和 x264
        println!("cargo:rustc-link-arg={}/libmfx.lib", vcpkg_lib);
        println!("cargo:rustc-link-arg={}/libx264.lib", vcpkg_lib);
    }
}
