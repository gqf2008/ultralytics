/// GPU加速仿射变换性能测试
/// 比较 CPU标准实现、SIMD优化、GPU加速的性能差异

use std::time::Instant;
use yolov8_rs::utils::affine_transform::*;

#[cfg(target_arch = "x86_64")]
use yolov8_rs::utils::affine_transform_simd::warp_affine_rgb_simd;

#[cfg(feature = "gpu")]
use yolov8_rs::utils::affine_transform_wgpu::WgpuAffineTransform;

fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut img = vec![0u8; width * height * 3];
    
    // 创建渐变图案
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            img[idx] = ((x * 255) / width) as u8;
            img[idx + 1] = ((y * 255) / height) as u8;
            img[idx + 2] = (((x + y) * 255) / (width + height)) as u8;
        }
    }
    
    // 添加图案
    for y in height / 4..3 * height / 4 {
        for x in width / 4..3 * width / 4 {
            let idx = (y * width + x) * 3;
            img[idx] = 255;
            img[idx + 1] = 128;
            img[idx + 2] = 64;
        }
    }
    
    img
}

fn benchmark_standard(
    src: &[u8],
    width: usize,
    height: usize,
    iterations: usize,
) -> f64 {
    let matrix = AffineMatrix::rotation_around_center(
        (width / 2) as f32,
        (height / 2) as f32,
        30.0,
    );
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _ = warp_affine_rgb(
            src,
            width,
            height,
            &matrix,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );
    }
    
    start.elapsed().as_secs_f64()
}

#[cfg(target_arch = "x86_64")]
fn benchmark_simd(
    src: &[u8],
    width: usize,
    height: usize,
    iterations: usize,
) -> f64 {
    let matrix = AffineMatrix::rotation_around_center(
        (width / 2) as f32,
        (height / 2) as f32,
        30.0,
    );
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _ = warp_affine_rgb_simd(
            src,
            width,
            height,
            &matrix,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );
    }
    
    start.elapsed().as_secs_f64()
}

#[cfg(feature = "gpu")]
fn benchmark_gpu(
    gpu_context: &WgpuAffineTransform,
    src: &[u8],
    width: u32,
    height: u32,
    iterations: usize,
) -> f64 {
    let matrix = AffineMatrix::rotation_around_center(
        (width / 2) as f32,
        (height / 2) as f32,
        30.0,
    );
    
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _ = gpu_context.warp_affine_rgb(
            src,
            width,
            height,
            &matrix,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );
    }
    
    start.elapsed().as_secs_f64()
}

#[cfg(feature = "gpu")]
fn main() {
    println!("=== GPU加速仿射变换性能对比 ===\n");

    // 初始化GPU上下文 (同步调用)
    println!("正在初始化GPU...");
    let gpu_context = match WgpuAffineTransform::new() {
        Ok(ctx) => {
            println!("GPU初始化成功!\n");
            ctx
        }
        Err(e) => {
            eprintln!("GPU初始化失败: {}", e);
            eprintln!("请确保系统支持Vulkan/Metal/DX12");
            return;
        }
    };

    let test_sizes = vec![
        (320, 240, "QVGA", 50),
        (640, 480, "VGA", 50),
        (1280, 720, "HD", 30),
        (1920, 1080, "Full HD", 10),
    ];

    for (width, height, name, iterations) in test_sizes {
        println!("=" .repeat(60));
        println!("测试分辨率: {} ({}x{})", name, width, height);
        println!("迭代次数: {}", iterations);
        println!("-".repeat(60));
        
        let img = create_test_image(width as usize, height as usize);
        
        // CPU标准实现
        println!("\n📊 CPU标准实现:");
        let time_standard = benchmark_standard(&img, width as usize, height as usize, iterations);
        let fps_standard = iterations as f64 / time_standard;
        println!("  总时间: {:.3}s", time_standard);
        println!("  平均每帧: {:.3}ms", time_standard * 1000.0 / iterations as f64);
        println!("  处理速度: {:.2} FPS", fps_standard);

        // SIMD优化
        #[cfg(target_arch = "x86_64")]
        {
            println!("\n⚡ SIMD优化 (AVX2):");
            let time_simd = benchmark_simd(&img, width as usize, height as usize, iterations);
            let fps_simd = iterations as f64 / time_simd;
            println!("  总时间: {:.3}s", time_simd);
            println!("  平均每帧: {:.3}ms", time_simd * 1000.0 / iterations as f64);
            println!("  处理速度: {:.2} FPS", fps_simd);
            
            let speedup_simd = time_standard / time_simd;
            println!("  vs CPU: {:.2}x 加速", speedup_simd);
        }

        // GPU加速
        println!("\n🚀 GPU加速 (wgpu):");
        let time_gpu = benchmark_gpu(&gpu_context, &img, width, height, iterations);
        let fps_gpu = iterations as f64 / time_gpu;
        println!("  总时间: {:.3}s", time_gpu);
        println!("  平均每帧: {:.3}ms", time_gpu * 1000.0 / iterations as f64);
        println!("  处理速度: {:.2} FPS", fps_gpu);
        
        let speedup_gpu = time_standard / time_gpu;
        println!("  vs CPU: {:.2}x 加速", speedup_gpu);

        #[cfg(target_arch = "x86_64")]
        {
            let time_simd = benchmark_simd(&img, width as usize, height as usize, iterations);
            let speedup_gpu_vs_simd = time_simd / time_gpu;
            println!("  vs SIMD: {:.2}x 加速", speedup_gpu_vs_simd);
        }

        println!();
    }

    println!("=" .repeat(60));
    println!("=== 测试完成 ===");
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("错误: GPU功能未启用!");
    eprintln!("请使用以下命令编译:");
    eprintln!("  cargo run --example affine_gpu_benchmark --features gpu --release");
}
