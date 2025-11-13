#!/usr/bin/env python3
"""
YOLOv8 INT8量化模型导出脚本
Export YOLOv8 models to INT8 quantized ONNX format

使用方法:
    python export_int8_models.py
    
导出的模型会自动保存到 models/ 目录
"""

from ultralytics import YOLO
import os

def export_int8_model(model_path, output_dir='models'):
    """导出INT8量化ONNX模型"""
    print(f"\n{'='*60}")
    print(f"📦 加载模型: {model_path}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print(f"💡 请先下载模型: https://github.com/ultralytics/assets/releases")
        return False
    
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 导出配置
        export_args = {
            'format': 'onnx',
            'imgsz': 320,        # 输入尺寸 (与RTSP程序匹配)
            'int8': True,        # 启用INT8量化
            'dynamic': False,    # 固定尺寸 (更快)
            'simplify': True,    # 简化模型图
            'opset': 12,         # ONNX opset版本
        }
        
        print(f"⚙️  导出配置: {export_args}")
        print(f"🔄 开始导出INT8量化ONNX...")
        
        # 导出模型
        exported_path = model.export(**export_args)
        
        print(f"✅ 导出成功: {exported_path}")
        
        # 移动到目标目录
        base_name = os.path.basename(model_path).replace('.pt', '')
        target_path = os.path.join(output_dir, f"{base_name}_int8.onnx")
        
        if os.path.exists(exported_path):
            os.makedirs(output_dir, exist_ok=True)
            
            # 重命名并移动
            import shutil
            shutil.move(exported_path, target_path)
            print(f"📁 已保存到: {target_path}")
            
            # 显示文件大小
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"📊 模型大小: {size_mb:.2f} MB")
            
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         YOLOv8 INT8 量化模型导出工具                   ║
    ║         INT8 Quantization Export Tool                  ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    # 要导出的模型列表
    models_to_export = [
        # 检测模型
        ('yolov8n.pt', '超轻量检测模型'),
        ('yolov8s.pt', '小型检测模型'),
        ('yolov8m.pt', '中型检测模型'),
        ('yolov8l.pt', '大型检测模型'),
        ('yolov8x.pt', '超大检测模型'),
        
        # 姿态估计模型
        ('yolov8n-pose.pt', '超轻量姿态估计'),
        ('yolov8s-pose.pt', '小型姿态估计'),
        ('yolov8m-pose.pt', '中型姿态估计'),
        ('yolov8l-pose.pt', '大型姿态估计'),
        ('yolov8x-pose.pt', '超大姿态估计'),
    ]
    
    success_count = 0
    total_count = 0
    
    for model_path, description in models_to_export:
        print(f"\n🎯 {description}")
        
        # 检查是否存在
        if os.path.exists(model_path):
            total_count += 1
            if export_int8_model(model_path):
                success_count += 1
        else:
            print(f"⏭️  跳过 (文件不存在): {model_path}")
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"📊 导出完成!")
    print(f"✅ 成功: {success_count}/{total_count}")
    
    if success_count > 0:
        print(f"\n💡 使用方法:")
        print(f"   # 使用INT8量化的M模型")
        print(f"   .\\target\\release\\yolov8-rtsp.exe --int8 -m m")
        print(f"   ")
        print(f"   # 使用INT8量化的N模型 (最快)")
        print(f"   .\\target\\release\\yolov8-rtsp.exe --int8 -m n")


if __name__ == '__main__':
    main()
