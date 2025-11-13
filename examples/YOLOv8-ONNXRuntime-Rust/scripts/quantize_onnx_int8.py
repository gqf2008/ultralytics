#!/usr/bin/env python3
"""
ONNX模型INT8量化工具
Quantize existing ONNX models to INT8 format

使用方法:
    python quantize_onnx_int8.py
    
从 models/ 目录读取ONNX模型,导出INT8量化版本
"""

import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_path, output_path):
    """量化ONNX模型到INT8"""
    print(f"\n{'='*60}")
    print(f"📦 输入模型: {input_path}")
    
    # 检查模型文件
    if not os.path.exists(input_path):
        print(f"❌ 模型文件不存在: {input_path}")
        return False
    
    # 显示原始模型大小
    size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"📊 原始大小: {size_mb:.2f} MB")
    
    try:
        print(f"🔄 开始INT8动态量化...")
        
        # 动态量化(无需校准数据)
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8  # 使用无符号INT8
        )
        
        # 验证量化后的模型
        quantized_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = size_mb / quantized_size_mb
        
        print(f"✅ 量化成功: {output_path}")
        print(f"📊 量化后大小: {quantized_size_mb:.2f} MB")
        print(f"🗜️  压缩比: {compression_ratio:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 量化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         ONNX模型 INT8 量化工具                         ║
    ║         ONNX INT8 Quantization Tool                    ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    models_dir = 'models'
    
    # 查找所有ONNX模型(排除已量化的)
    onnx_models = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.onnx') and '_int8' not in filename:
            onnx_models.append(filename)
    
    if not onnx_models:
        print("❌ 在 models/ 目录下未找到ONNX模型")
        return
    
    print(f"🔍 找到 {len(onnx_models)} 个模型:")
    for model in onnx_models:
        print(f"   - {model}")
    
    success_count = 0
    
    for model_name in onnx_models:
        input_path = os.path.join(models_dir, model_name)
        
        # 生成输出文件名
        base_name = model_name.replace('.onnx', '')
        output_name = f"{base_name}_int8.onnx"
        output_path = os.path.join(models_dir, output_name)
        
        # 检查是否已存在
        if os.path.exists(output_path):
            print(f"\n⏭️  跳过 (已存在): {output_name}")
            continue
        
        # 量化模型
        if quantize_onnx_model(input_path, output_path):
            success_count += 1
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"📊 量化完成!")
    print(f"✅ 成功: {success_count}/{len(onnx_models)}")
    
    if success_count > 0:
        print(f"\n💡 使用方法:")
        print(f"   # 使用INT8量化的M模型")
        print(f"   .\\target\\release\\yolov8-rtsp.exe --int8 -m m")
        print(f"   ")
        print(f"   # 使用INT8量化的N模型 (最快)")
        print(f"   .\\target\\release\\yolov8-rtsp.exe --int8 -m n")
        print(f"\n📁 量化后的模型保存在: models/")
        
        # 列出生成的文件
        print(f"\n📦 生成的INT8模型:")
        for model_name in onnx_models:
            base_name = model_name.replace('.onnx', '')
            output_name = f"{base_name}_int8.onnx"
            output_path = os.path.join(models_dir, output_name)
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   ✅ {output_name:30s} ({size_mb:6.2f} MB)")


if __name__ == '__main__':
    main()
