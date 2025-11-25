// 构建脚本: 链接FFmpeg依赖库
fn main() {
    // 仅在Windows MSVC环境下添加FFmpeg相关库
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        // vcpkg库路径
        let vcpkg_lib = r"D:\workspace\vcpkg\installed\x64-windows-static-md\lib";
        println!("cargo:rustc-link-search=native={}", vcpkg_lib);

        // Windows 系统库必须先链接
        println!("cargo:rustc-link-lib=gdi32"); // GDI绘图函数
        println!("cargo:rustc-link-lib=user32"); // 窗口管理函数
        println!("cargo:rustc-link-lib=shell32"); // Shell API (SHCreateStreamOnFileA)
        println!("cargo:rustc-link-lib=ole32"); // OLE基础库
        println!("cargo:rustc-link-lib=oleaut32"); // OLE自动化
        println!("cargo:rustc-link-lib=vfw32"); // Video for Windows
        println!("cargo:rustc-link-lib=secur32"); // 安全通道
        println!("cargo:rustc-link-lib=ws2_32"); // Windows Sockets
        println!("cargo:rustc-link-lib=advapi32"); // 高级API
        println!("cargo:rustc-link-lib=bcrypt"); // 加密API

        // 然后链接vcpkg静态库(使用绝对路径)
        println!("cargo:rustc-link-arg={}/libmfx.lib", vcpkg_lib);
        println!("cargo:rustc-link-arg={}/libx264.lib", vcpkg_lib);
    }
}
