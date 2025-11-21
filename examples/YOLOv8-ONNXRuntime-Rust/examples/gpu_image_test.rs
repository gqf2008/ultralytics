use image::{open, RgbImage};
use std::path::Path;
/// GPU加速仿射变换实际图片测试
/// 读取assets/images中的图片,应用各种变换并保存结果
use yolov8_rs::utils::affine_transform::*;

#[cfg(feature = "gpu")]
use yolov8_rs::utils::affine_transform_wgpu::WgpuAffineTransform;

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU加速图片变换测试 ===\n");

    // 初始化GPU
    println!("⏳ 初始化GPU...");
    let gpu = WgpuAffineTransform::new()?;
    println!("✅ GPU初始化成功!\n");

    // 创建输出目录
    let output_dir = "assets/images/transformed";
    std::fs::create_dir_all(output_dir)?;
    println!("📁 输出目录: {}\n", output_dir);

    // 要处理的图片
    let images = vec![("bus.jpg", "公交车"), ("background.jpg", "背景")];

    for (filename, desc) in images {
        let input_path = format!("assets/images/{}", filename);

        if !Path::new(&input_path).exists() {
            println!("⚠️  跳过 {}: 文件不存在", filename);
            continue;
        }

        println!("{}", "=".repeat(60));
        println!("📷 处理图片: {} ({})", filename, desc);
        println!("{}", "-".repeat(60));

        // 加载图片
        let img = open(&input_path)?;
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let image_data = rgb_img.into_raw();

        println!("   原始尺寸: {}x{}", width, height);
        println!("   数据大小: {} bytes", image_data.len());

        let base_name = filename.replace(".jpg", "");

        // 1. 旋转45度
        println!("\n1️⃣  旋转45度...");
        let matrix_rotate =
            AffineMatrix::rotation_around_center(width as f32 / 2.0, height as f32 / 2.0, 45.0);
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
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

        let output_path = format!("{}/{}_rotate45.jpg", output_dir, base_name);
        save_image(&result, width, height, &output_path)?;
        println!("   💾 保存: {}", output_path);

        // 2. 缩放1.5倍
        println!("\n2️⃣  放大1.5倍...");
        let new_width = (width as f32 * 1.5) as u32;
        let new_height = (height as f32 * 1.5) as u32;
        let matrix_scale = AffineMatrix::scale(1.5, 1.5);
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
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

        let output_path = format!("{}/{}_scale1.5x.jpg", output_dir, base_name);
        save_image(&result, new_width, new_height, &output_path)?;
        println!("   💾 保存: {} ({}x{})", output_path, new_width, new_height);

        // 3. 缩小0.5倍
        println!("\n3️⃣  缩小0.5倍...");
        let new_width = (width as f32 * 0.5) as u32;
        let new_height = (height as f32 * 0.5) as u32;
        let matrix_scale = AffineMatrix::scale(0.5, 0.5);
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
            width,
            height,
            &matrix_scale,
            (new_width, new_height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(128),
        );
        println!(
            "   ⏱️  用时: {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        let output_path = format!("{}/{}_scale0.5x.jpg", output_dir, base_name);
        save_image(&result, new_width, new_height, &output_path)?;
        println!("   💾 保存: {} ({}x{})", output_path, new_width, new_height);

        // 4. 旋转30度 + 缩放0.8倍
        println!("\n4️⃣  旋转30度 + 缩放0.8倍...");
        let matrix_combo =
            AffineMatrix::rotation_around_center(width as f32 / 2.0, height as f32 / 2.0, 30.0)
                .compose(&AffineMatrix::scale(0.8, 0.8));
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
            width,
            height,
            &matrix_combo,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Constant(50),
        );
        println!(
            "   ⏱️  用时: {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        let output_path = format!("{}/{}_rotate30_scale0.8x.jpg", output_dir, base_name);
        save_image(&result, width, height, &output_path)?;
        println!("   💾 保存: {}", output_path);

        // 5. 水平翻转
        println!("\n5️⃣  水平翻转...");
        let matrix_flip_h = AffineMatrix::from_array([[-1.0, 0.0, width as f32], [0.0, 1.0, 0.0]]);
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
            width,
            height,
            &matrix_flip_h,
            (width, height),
            InterpolationMethod::Nearest,
            BorderMode::Constant(0),
        );
        println!(
            "   ⏱️  用时: {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        let output_path = format!("{}/{}_flip_horizontal.jpg", output_dir, base_name);
        save_image(&result, width, height, &output_path)?;
        println!("   💾 保存: {}", output_path);

        // 6. 垂直翻转
        println!("\n6️⃣  垂直翻转...");
        let matrix_flip_v = AffineMatrix::from_array([[1.0, 0.0, 0.0], [0.0, -1.0, height as f32]]);
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
            width,
            height,
            &matrix_flip_v,
            (width, height),
            InterpolationMethod::Nearest,
            BorderMode::Constant(0),
        );
        println!(
            "   ⏱️  用时: {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        let output_path = format!("{}/{}_flip_vertical.jpg", output_dir, base_name);
        save_image(&result, width, height, &output_path)?;
        println!("   💾 保存: {}", output_path);

        // 7. 倾斜变换
        println!("\n7️⃣  倾斜变换...");
        let matrix_shear = AffineMatrix::from_array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0]]);
        let start = std::time::Instant::now();
        let result = gpu.warp_affine_rgb(
            &image_data,
            width,
            height,
            &matrix_shear,
            (width, height),
            InterpolationMethod::Bilinear,
            BorderMode::Replicate,
        );
        println!(
            "   ⏱️  用时: {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        let output_path = format!("{}/{}_shear.jpg", output_dir, base_name);
        save_image(&result, width, height, &output_path)?;
        println!("   💾 保存: {}", output_path);

        println!();
    }

    println!("{}", "=".repeat(60));
    println!("✅ 全部完成!");
    println!("\n查看结果:");
    println!("   cd {}", output_dir);
    println!("   explorer .");

    Ok(())
}

/// 保存RGB图像为JPEG
fn save_image(
    data: &[u8],
    width: u32,
    height: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let img = RgbImage::from_raw(width, height, data.to_vec()).ok_or("无法创建图像")?;
    img.save(path)?;
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("❌ 错误: GPU功能未启用!");
    eprintln!("请使用以下命令运行:");
    eprintln!("  cargo run --example gpu_image_test --features gpu --release");
}
