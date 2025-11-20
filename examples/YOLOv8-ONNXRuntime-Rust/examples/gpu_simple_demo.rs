/// GPU加速仿射变换简单示例
/// 演示如何使用wgpu进行高性能图像变换
use yolov8_rs::utils::affine_transform::*;

#[cfg(feature = "gpu")]
use yolov8_rs::utils::affine_transform_wgpu::WgpuAffineTransform;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU加速仿射变换示例 ===\n");

    // 1. 初始化GPU (同步调用,内部使用pollster)
    println!("⏳ 正在初始化GPU...");
    let gpu = WgpuAffineTransform::new()?;
    println!("✅ GPU初始化成功!\n");

    // 2. 创建测试图像 (640x480 RGB)
    let width = 640u32;
    let height = 480u32;
    let mut image = vec![0u8; (width * height * 3) as usize];

    println!("📷 创建测试图像 ({}x{})...", width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            image[idx] = ((x * 255) / width) as u8; // R渐变
            image[idx + 1] = ((y * 255) / height) as u8; // G渐变
            image[idx + 2] = 128; // B固定
        }
    }

    // 在中心绘制矩形
    for y in height / 4..3 * height / 4 {
        for x in width / 4..3 * width / 4 {
            let idx = ((y * width + x) * 3) as usize;
            image[idx] = 255;
            image[idx + 1] = 255;
            image[idx + 2] = 0; // 黄色矩形
        }
    }
    println!("✅ 测试图像创建完成\n");

    // 3. 执行各种变换
    println!("🚀 开始GPU加速变换...\n");

    // 示例1: 旋转45度
    println!("1️⃣  旋转变换 (45度)");
    let matrix_rotate =
        AffineMatrix::rotation_around_center((width / 2) as f32, (height / 2) as f32, 45.0);
    let start = std::time::Instant::now();
    let result_rotate = gpu.warp_affine_rgb(
        &image,
        width,
        height,
        &matrix_rotate,
        (width, height),
        InterpolationMethod::Bilinear,
        BorderMode::Constant(0),
    );
    println!(
        "   ⏱️  用时: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!("   📊 结果大小: {} bytes\n", result_rotate.len());

    // 示例2: 缩放1.5倍
    println!("2️⃣  缩放变换 (1.5x)");
    let matrix_scale = AffineMatrix::scale(1.5, 1.5);
    let new_width = (width as f32 * 1.5) as u32;
    let new_height = (height as f32 * 1.5) as u32;
    let start = std::time::Instant::now();
    let result_scale = gpu.warp_affine_rgb(
        &image,
        width,
        height,
        &matrix_scale,
        (new_width, new_height),
        InterpolationMethod::Bilinear,
        BorderMode::Replicate,
    );
    println!(
        "   ⏱️  用时: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!("   📊 输出尺寸: {}x{}\n", new_width, new_height);

    // 示例3: 平移
    println!("3️⃣  平移变换 (+100, +50)");
    let matrix_translate = AffineMatrix::translation(100.0, 50.0);
    let start = std::time::Instant::now();
    let result_translate = gpu.warp_affine_rgb(
        &image,
        width,
        height,
        &matrix_translate,
        (width, height),
        InterpolationMethod::Nearest,
        BorderMode::Constant(0),
    );
    println!(
        "   ⏱️  用时: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!("   📊 插值方法: Nearest\n",);

    // 示例4: 组合变换 (旋转+缩放)
    println!("4️⃣  组合变换 (旋转30度 + 缩放0.8x)");
    let matrix_combo =
        AffineMatrix::rotation_around_center((width / 2) as f32, (height / 2) as f32, 30.0)
            .compose(&AffineMatrix::scale(0.8, 0.8));

    let start = std::time::Instant::now();
    let result_combo = gpu.warp_affine_rgb(
        &image,
        width,
        height,
        &matrix_combo,
        (width, height),
        InterpolationMethod::Bilinear,
        BorderMode::Constant(128),
    );
    println!(
        "   ⏱️  用时: {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    println!("   📊 边界模式: Constant(128)\n");

    // 5. 性能测试
    println!("⚡ 性能测试 (100次迭代)...");
    let iterations = 100;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let _ = gpu.warp_affine_rgb(
            &image,
            width,
            height,
            &matrix_rotate,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(0),
        );
    }

    let total_time = start.elapsed().as_secs_f64();
    let avg_time = total_time / iterations as f64;
    let fps = 1.0 / avg_time;

    println!("   📊 总时间: {:.3}s", total_time);
    println!("   ⚡ 平均每帧: {:.2}ms", avg_time * 1000.0);
    println!("   🚀 处理速度: {:.1} FPS\n", fps);

    println!("=== 演示完成 ===");

    // 提示: 如何保存结果
    println!("\n💡 提示:");
    println!("   可以使用image crate保存结果:");
    println!("   use image::{{RgbImage, ImageBuffer}};");
    println!("   let img = ImageBuffer::from_raw(width, height, result_rotate).unwrap();");
    println!("   img.save(\"output.png\").unwrap();");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("❌ 错误: GPU功能未启用!");
    eprintln!("\n请使用以下命令运行:");
    eprintln!("  cargo run --example gpu_simple_demo --features gpu --release");
    eprintln!("\n或添加到Cargo.toml:");
    eprintln!("  [dependencies]");
    eprintln!("  yolov8-rs = {{ version = \"0.1.0\", features = [\"gpu\"] }}");
}
