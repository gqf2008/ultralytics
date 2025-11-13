#!/usr/bin/env pwsh
# RTSP 实时检测流程脚本

param(
    [string]$RtspUrl = "rtsp://admin:Wosai2018@172.19.54.45/cam/realmonitor?channel=1subtype=0",
    [int]$Duration = 30,
    [int]$Width = 416,
    [int]$Height = 416,
    [string]$Model = "yolov8n.onnx"
)

Write-Host "🚀 YOLOv8 RTSP 实时检测流程" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# 步骤 1: 从 RTSP 录制视频 (带硬件缩放)
Write-Host "📡 步骤 1: 从 RTSP 录制视频..." -ForegroundColor Cyan
Write-Host "  URL: $RtspUrl"
Write-Host "  时长: $Duration 秒"
Write-Host "  分辨率: ${Width}x${Height}"
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$videoFile = "rtsp_${timestamp}.mp4"

# 使用 FFmpeg 录制 + 缩放
$ffmpegArgs = @(
    "-rtsp_transport", "tcp",
    "-i", $RtspUrl,
    "-vf", "scale=${Width}:${Height}:flags=fast_bilinear",
    "-t", $Duration,
    "-c:v", "libx264",
    "-preset", "fast",
    "-y",
    $videoFile
)

Write-Host "  执行: ffmpeg $($ffmpegArgs -join ' ')" -ForegroundColor Gray
ffmpeg @ffmpegArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ FFmpeg 录制失败" -ForegroundColor Red
    exit 1
}

Write-Host "✅ 视频录制完成: $videoFile" -ForegroundColor Green
Write-Host ""

# 步骤 2: YOLO 检测
Write-Host "🔍 步骤 2: YOLO 目标检测..." -ForegroundColor Cyan
Write-Host "  模型: $Model"
Write-Host ""

$outputFile = "detected_${timestamp}.jpg"

cargo run --release -- `
    --model $Model `
    --source $videoFile `
    --conf 0.3 `
    --iou 0.45 `
    --width $Width `
    --height $Height `
    --device cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ YOLO 检测失败" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ 完成!" -ForegroundColor Green
Write-Host "  输入视频: $videoFile"
Write-Host "  检测结果: 查看保存的结果图片"
Write-Host ""
Write-Host "💡 提示: 如需实时处理,考虑:" -ForegroundColor Yellow
Write-Host "  1. 减小录制时长 (如 5-10 秒)"
Write-Host "  2. 使用 GPU 加速 (--device cuda)"
Write-Host "  3. 使用循环脚本持续处理"
