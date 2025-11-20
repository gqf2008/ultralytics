/// 仿射变换性能基准测试
/// 比较标准实现和SIMD优化实现的性能差异

use std::time::Instant;
use yolov8_rs::utils::affine_transform::*;
use yolov8_rs::utils::affine_transform_simd::warp_affine_rgb_simd;

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
    
    // 添加一些图案
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

fn main() {
    println!("=== 仿射变换性能基准测试 ===\n");

    let test_sizes = vec![
        (320, 240, "QVGA"),
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ];

    for (width, height, name) in test_sizes {
        println!("测试分辨率: {} ({}x{})", name, width, height);
        
        let img = create_test_image(width, height);
        let iterations = if width * height > 1000000 { 10 } else { 50 };
        
        // 预热
        let _ = warp_affine_rgb(
            &img,
            width,
            height,
            &AffineMatrix::identity(),
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );

        // 标准实现测试
        println!("  标准实现:");
        let time_standard = benchmark_standard(&img, width, height, iterations);
        let fps_standard = iterations as f64 / time_standard;
        println!("    总时间: {:.3}s", time_standard);
        println!("    平均每帧: {:.3}ms", time_standard * 1000.0 / iterations as f64);
        println!("    处理速度: {:.2} FPS", fps_standard);

        // SIMD实现测试
        #[cfg(target_arch = "x86_64")]
        {
            println!("  SIMD优化:");
            let time_simd = benchmark_simd(&img, width, height, iterations);
            let fps_simd = iterations as f64 / time_simd;
            println!("    总时间: {:.3}s", time_simd);
            println!("    平均每帧: {:.3}ms", time_simd * 1000.0 / iterations as f64);
            println!("    处理速度: {:.2} FPS", fps_simd);
            
            let speedup = time_standard / time_simd;
            println!("    性能提升: {:.2}x", speedup);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            println!("  SIMD优化: 当前架构不支持");
        }

        println!();
    }

    println!("=== 测试完成 ===");
}
