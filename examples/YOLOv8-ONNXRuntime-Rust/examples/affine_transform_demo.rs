use ndarray::Array2;
/// 仿射变换使用示例
/// 演示如何使用 warp_affine 进行图像变换
use yolov8_rs::utils::affine_transform::*;

fn main() {
    println!("🎨 仿射变换示例\n");

    // 示例1: 平移变换
    println!("📍 示例1: 平移变换");
    demo_translation();

    // 示例2: 缩放变换
    println!("\n📐 示例2: 缩放变换");
    demo_scale();

    // 示例3: 旋转变换
    println!("\n🔄 示例3: 旋转变换");
    demo_rotation();

    // 示例4: 组合变换
    println!("\n🔗 示例4: 组合变换 (平移+旋转+缩放)");
    demo_combined();

    // 示例5: RGB图像变换
    println!("\n🖼️  示例5: RGB图像变换");
    demo_rgb();

    // 示例6: 从对应点计算变换矩阵
    println!("\n📊 示例6: 从对应点计算变换矩阵");
    demo_from_points();
}

fn demo_translation() {
    // 创建简单的测试图像 (10x10)
    let mut src = Array2::<u8>::zeros((10, 10));
    // 在中心画一个小方块
    for i in 3..7 {
        for j in 3..7 {
            src[[i, j]] = 255;
        }
    }

    // 向右下平移 (2, 3)
    let matrix = AffineMatrix::translation(2.0, 3.0);
    let dst = warp_affine_gray(
        &src,
        &matrix,
        (10, 10),
        InterpolationMethod::Nearest,
        BorderMode::Constant(0),
    );

    println!("原始图像中心方块: (3,3) -> (6,6)");
    println!("变换后方块位置: (5,6) -> (8,9)");
    println!("✅ 平移矩阵: {:?}", matrix.to_array());
}

fn demo_scale() {
    let mut src = Array2::<u8>::zeros((10, 10));
    for i in 4..6 {
        for j in 4..6 {
            src[[i, j]] = 255;
        }
    }

    // 缩放2倍
    let matrix = AffineMatrix::scale(2.0, 2.0);
    let dst = warp_affine_gray(
        &src,
        &matrix,
        (20, 20),
        InterpolationMethod::Bilinear,
        BorderMode::Constant(0),
    );

    println!("原始图像大小: 10x10");
    println!("缩放后图像大小: 20x20 (2倍)");
    println!("✅ 缩放矩阵: {:?}", matrix.to_array());
}

fn demo_rotation() {
    let mut src = Array2::<u8>::zeros((100, 100));
    // 画一条水平线
    for j in 20..80 {
        src[[50, j]] = 255;
    }

    // 绕中心旋转45度
    let matrix = AffineMatrix::rotation_around_center(50.0, 50.0, 45.0);
    let dst = warp_affine_gray(
        &src,
        &matrix,
        (100, 100),
        InterpolationMethod::Bilinear,
        BorderMode::Constant(0),
    );

    println!("原始: 水平线");
    println!("旋转: 45度斜线");
    println!("✅ 旋转矩阵: {:?}", matrix.to_array());
}

fn demo_combined() {
    let mut src = Array2::<u8>::zeros((100, 100));
    for i in 40..60 {
        for j in 40..60 {
            src[[i, j]] = 255;
        }
    }

    // 组合变换: 先缩放0.5倍，再旋转30度，最后平移
    let scale = AffineMatrix::scale(0.5, 0.5);
    let rotate = AffineMatrix::rotation_around_center(50.0, 50.0, 30.0);
    let translate = AffineMatrix::translation(20.0, 10.0);

    // 组合顺序: translate * rotate * scale
    let combined = translate.compose(&rotate.compose(&scale));

    let dst = warp_affine_gray(
        &src,
        &combined,
        (100, 100),
        InterpolationMethod::Bilinear,
        BorderMode::Constant(0),
    );

    println!("变换顺序: 缩放 -> 旋转 -> 平移");
    println!("✅ 组合矩阵: {:?}", combined.to_array());
}

fn demo_rgb() {
    // 创建RGB测试图像 (50x50)
    let width = 50;
    let height = 50;
    let mut src = vec![0u8; width * height * 3];

    // 绘制红色方块
    for y in 15..35 {
        for x in 15..35 {
            let idx = (y * width + x) * 3;
            src[idx] = 255; // R
            src[idx + 1] = 0; // G
            src[idx + 2] = 0; // B
        }
    }

    // 旋转45度
    let matrix = AffineMatrix::rotation_around_center(25.0, 25.0, 45.0);
    let dst = warp_affine_rgb(
        &src,
        width,
        height,
        &matrix,
        (width, height),
        InterpolationMethod::Bilinear,
        BorderMode::Constant(0),
    );

    println!("RGB图像: 红色方块");
    println!("变换: 旋转45度");
    println!("输出大小: {}x{} x 3通道", width, height);
    println!("✅ RGB变换完成");
}

fn demo_from_points() {
    // 定义源图像中的3个点
    let src_pts = [
        (0.0, 0.0),   // 左上角
        (100.0, 0.0), // 右上角
        (0.0, 100.0), // 左下角
    ];

    // 定义目标图像中对应的3个点 (梯形变换)
    let dst_pts = [
        (20.0, 10.0),  // 左上角向右下移动
        (120.0, 5.0),  // 右上角向右上移动
        (10.0, 110.0), // 左下角向右下移动
    ];

    match get_affine_transform(src_pts, dst_pts) {
        Some(matrix) => {
            println!("源点: {:?}", src_pts);
            println!("目标点: {:?}", dst_pts);
            println!("✅ 计算得到的仿射矩阵:");
            println!("   {:?}", matrix.to_array());

            // 验证变换
            println!("\n验证变换:");
            for i in 0..3 {
                let (x, y) = matrix.transform_point(src_pts[i].0, src_pts[i].1);
                println!(
                    "  {:?} -> ({:.2}, {:.2}), 期望: {:?}",
                    src_pts[i], x, y, dst_pts[i]
                );
            }
        }
        None => {
            println!("❌ 无法计算仿射矩阵 (点共线或重复)");
        }
    }
}
