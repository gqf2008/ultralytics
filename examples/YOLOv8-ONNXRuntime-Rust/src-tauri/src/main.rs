// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod capture;
mod detector_async;
mod multi_window_capture;
mod qsv_decoder;
mod rtsp_proxy;
mod shared_memory;

use detector_async::AsyncDetectorState;
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::panic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::window::Color;
use tauri::{AppHandle, Emitter, Manager, State};

// Global Proxy State
pub struct ProxyState {
    pub proxy: Arc<rtsp_proxy::RtspProxy>,
}

/// 视频帧共享状态
pub struct VideoFrameState {
    pub buffer: Arc<RwLock<Vec<u8>>>,
    pub width: Arc<RwLock<u32>>,
    pub height: Arc<RwLock<u32>>,
    pub updated: Arc<RwLock<bool>>,
}

impl Default for VideoFrameState {
    fn default() -> Self {
        Self {
            buffer: Arc::new(RwLock::new(Vec::new())),
            width: Arc::new(RwLock::new(0)),
            height: Arc::new(RwLock::new(0)),
            updated: Arc::new(RwLock::new(false)),
        }
    }
}

/// 全局解码器代数计数器
static DECODER_GENERATION: AtomicUsize = AtomicUsize::new(0);

/// 检测结果
#[derive(Clone, Serialize, Deserialize)]
pub struct DetectionBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub class_id: u32,
}

#[derive(Clone, Serialize)]
pub struct FrameData {
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}

/// 获取最新视频帧 (WebGL 优化)
#[tauri::command]
async fn get_latest_frame(
    state: State<'_, VideoFrameState>,
) -> Result<(Vec<u8>, u32, u32), String> {
    let buffer = state.buffer.read().clone();
    let width = *state.width.read();
    let height = *state.height.read();

    if buffer.is_empty() {
        return Err("No frame available".to_string());
    }

    Ok((buffer, width, height))
}

/// 检查是否有新帧
#[tauri::command]
async fn has_new_frame(state: State<'_, VideoFrameState>) -> Result<bool, String> {
    Ok(*state.updated.read())
}

/// 标记帧已处理
#[tauri::command]
async fn mark_frame_processed(state: State<'_, VideoFrameState>) -> Result<(), String> {
    *state.updated.write() = false;
    Ok(())
}

/// 前端日志透传
#[tauri::command]
async fn log_frontend(msg: String) {
    println!("[Frontend]: {}", msg);
}

/// 显示主窗口（前端加载完成后调用，避免白屏闪烁）
#[tauri::command]
async fn show_window(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        window.show().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 读取文本文件
#[tauri::command]
async fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))
}

/// 读取 RTSP 历史记录
#[tauri::command]
async fn get_rtsp_history() -> Result<Vec<String>, String> {
    let path = "rtsp_history.txt";
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let urls: Vec<String> = content
                .lines()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
            Ok(urls)
        }
        Err(_) => Ok(Vec::new()), // 文件不存在返回空数组
    }
}

/// 添加 RTSP 历史记录
#[tauri::command]
async fn add_rtsp_history(url: String) -> Result<(), String> {
    let path = "rtsp_history.txt";

    // 读取现有记录
    let mut urls = match std::fs::read_to_string(path) {
        Ok(content) => content
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
        Err(_) => Vec::new(),
    };

    // 如果 URL已存在,先移除(保证最新的在最前面)
    urls.retain(|u| u != &url);

    // 添加到最前面
    urls.insert(0, url);

    // 限制最多保留 20 条记录
    if urls.len() > 20 {
        urls.truncate(20);
    }

    // 写回文件
    let content = urls.join("\n");
    std::fs::write(path, content).map_err(|e| format!("Failed to write history: {}", e))
}

/// 清空 RTSP 历史记录
#[tauri::command]
async fn clear_rtsp_history() -> Result<(), String> {
    let path = "rtsp_history.txt";
    std::fs::write(path, "").map_err(|e| format!("Failed to clear history: {}", e))
}

/// 启动 RTSP 流
#[tauri::command]
async fn start_rtsp_stream(
    app: AppHandle,
    url: String,
    video_channel: tauri::ipc::Channel,
    audio_channel: tauri::ipc::Channel,
) -> Result<String, String> {
    println!("🚀 启动 RTSP 流: {}", url);

    // 增加解码器代数
    let generation = DECODER_GENERATION.fetch_add(1, Ordering::SeqCst);
    println!("📌 新解码器代数: {}", generation);

    // 启动 RTSP 代理 (用于前端显示)
    let proxy_state = app.state::<ProxyState>();
    proxy_state
        .proxy
        .start(url.clone(), Some(video_channel), Some(audio_channel));

    Ok(format!("RTSP 流已启动 (Gen: {})", generation))
}

// ==================== 设备和模型配置命令 ====================

/// 可用加速设备信息
#[derive(Clone, Serialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub available: bool,
}

/// 获取可用加速设备列表
#[tauri::command]
async fn get_available_devices() -> Result<Vec<DeviceInfo>, String> {
    let mut devices = Vec::new();

    // 检查 CUDA
    let cuda = CUDAExecutionProvider::default();
    let cuda_available = cuda.is_available().unwrap_or(false);
    devices.push(DeviceInfo {
        id: "cuda".to_string(),
        name: "CUDA (NVIDIA GPU)".to_string(),
        available: cuda_available,
    });

    // 检查 DirectML
    let dml = DirectMLExecutionProvider::default();
    let dml_available = dml.is_available().unwrap_or(false);
    devices.push(DeviceInfo {
        id: "directml".to_string(),
        name: "DirectML (Windows GPU)".to_string(),
        available: dml_available,
    });

    // CPU 始终可用
    devices.push(DeviceInfo {
        id: "cpu".to_string(),
        name: "CPU".to_string(),
        available: true,
    });

    Ok(devices)
}

/// 模型信息
#[derive(Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub size: String,
    pub exists: bool,
}

/// 获取可用模型列表
#[tauri::command]
async fn get_available_models() -> Result<Vec<ModelInfo>, String> {
    let models_dir = std::env::current_exe()
        .map_err(|e| format!("获取当前exe路径失败: {}", e))?
        .parent()
        .map(|p| p.join("models"))
        .unwrap_or_else(|| std::path::PathBuf::from("models"));

    let model_configs = vec![
        ("yolov8n", "YOLOv8 Nano", "~6MB"),
        ("yolov8s", "YOLOv8 Small", "~22MB"),
        ("yolov8m", "YOLOv8 Medium", "~52MB"),
        ("yolov8l", "YOLOv8 Large", "~87MB"),
        ("yolov8n-seg", "YOLOv8n Seg", "~7MB"),
        ("yolov8m-seg", "YOLOv8m Seg", "~53MB"),
        ("yolov10n", "YOLOv10 Nano", "~5MB"),
        ("yolov11n", "YOLOv11 Nano", "~5MB"),
    ];

    // 只返回存在的模型
    let models: Vec<ModelInfo> = model_configs
        .iter()
        .filter_map(|(id, name, size)| {
            let model_path = models_dir.join(format!("{}.onnx", id));
            if model_path.exists() {
                Some(ModelInfo {
                    id: id.to_string(),
                    name: name.to_string(),
                    path: model_path.to_string_lossy().to_string(),
                    size: size.to_string(),
                    exists: true,
                })
            } else {
                None
            }
        })
        .collect();

    Ok(models)
}

// ==================== 检测器命令 ====================

/// 启动检测器返回结果
#[derive(Clone, serde::Serialize)]
pub struct StartDetectorResult {
    pub message: String,
    pub input_width: u32,
    pub input_height: u32,
    pub device: String,
}

/// 启动检测器
#[tauri::command]
async fn start_detector(
    app: AppHandle,
    model: String,
    device: String,
    _tracker: String,
    result_channel: tauri::ipc::Channel<detector_async::DetectionResult>,
) -> Result<StartDetectorResult, String> {
    println!("🚀 启动检测器: model={}, device={}", model, device);

    let detector_state = app.state::<AsyncDetectorState>();

    // 获取模型路径
    // 如果 model 已经是完整路径（包含路径分隔符或 .onnx 后缀），直接使用
    // 否则在 models 目录下查找
    let model_path = if model.contains(std::path::MAIN_SEPARATOR)
        || model.contains('/')
        || model.ends_with(".onnx")
    {
        // 可能是完整路径
        if std::path::Path::new(&model).exists() {
            std::path::PathBuf::from(&model)
        } else {
            // 可能只是文件名，在 models 目录下查找
            std::env::current_exe()
                .map_err(|e| format!("获取当前exe路径失败: {}", e))?
                .parent()
                .map(|p| p.join("models").join(&model))
                .unwrap_or_else(|| std::path::PathBuf::from("models").join(&model))
        }
    } else {
        // 简单模型名，添加 .onnx 后缀
        let model_filename = format!("{}.onnx", model);
        std::env::current_exe()
            .map_err(|e| format!("获取当前exe路径失败: {}", e))?
            .parent()
            .map(|p| p.join("models").join(&model_filename))
            .unwrap_or_else(|| std::path::PathBuf::from("models").join(&model_filename))
    };

    let model_path_str = model_path.to_string_lossy().to_string();
    println!("📁 模型路径: {}", model_path_str);

    // 启动异步检测器 - 传入结果 Channel 和设备选择
    let (input_width, input_height, actual_device) =
        detector_state.start_with_device(&model_path_str, &device, result_channel)?;

    Ok(StartDetectorResult {
        message: format!(
            "检测器已启动: {} @ {} (输入: {}x{})",
            model, actual_device, input_width, input_height
        ),
        input_width,
        input_height,
        device: actual_device,
    })
}

/// 停止检测器
#[tauri::command]
async fn stop_detector(app: AppHandle) -> Result<String, String> {
    let detector_state = app.state::<AsyncDetectorState>();
    detector_state.stop();
    Ok("检测器已停止".to_string())
}

/// 发送帧到检测线程 (非阻塞，使用 Raw Request 避免 JSON 序列化)
#[tauri::command]
async fn detect_frame(app: AppHandle, request: tauri::ipc::Request<'_>) -> Result<(), String> {
    // 从 Raw Body 获取帧数据 (避免 JSON 序列化 400KB)
    let rgba_data = match request.body() {
        tauri::ipc::InvokeBody::Raw(data) => data.clone(),
        tauri::ipc::InvokeBody::Json(_) => {
            return Err("期望 Raw body，收到 JSON".to_string());
        }
    };

    // 从 headers 获取尺寸 (header 名可能被转为小写)
    let headers = request.headers();
    let width: u32 = headers
        .get("x-width")
        .or_else(|| headers.get("X-Width"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(640);
    let height: u32 = headers
        .get("x-height")
        .or_else(|| headers.get("X-Height"))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(640);

    // 调试日志
    if width != 640 || height != 640 {
        println!("⚠️ detect_frame: 收到 {}x{} (期望 640x640)", width, height);
    }

    let detector_state = app.state::<AsyncDetectorState>();
    detector_state.send_frame(rgba_data, width, height)
}

// ==================== 零拷贝共享内存渲染 ====================

use shared_memory::{FrameInfo, SharedMemoryInfo, ZeroCopyRendererState};

/// 启动零拷贝解码 (共享内存)
#[tauri::command]
async fn start_zerocopy_stream(
    zerocopy_state: State<'_, ZeroCopyRendererState>,
    url: String,
    hardware: String,
    width: u32,
    height: u32,
) -> Result<SharedMemoryInfo, String> {
    let hw_accel = match hardware.as_str() {
        "cuda" | "nvidia" => qsv_decoder::HardwareAccel::Cuda,
        "qsv" | "intel" => qsv_decoder::HardwareAccel::Qsv,
        _ => qsv_decoder::HardwareAccel::Qsv,
    };

    println!(
        "🚀 启动零拷贝解码: {} ({}) @ {}x{}",
        url, hardware, width, height
    );

    zerocopy_state.start(url, hw_accel, width, height)
}

/// 停止零拷贝解码
#[tauri::command]
async fn stop_zerocopy_stream(
    zerocopy_state: State<'_, ZeroCopyRendererState>,
) -> Result<String, String> {
    zerocopy_state.stop();
    println!("🛑 零拷贝解码已停止");
    Ok("零拷贝解码已停止".to_string())
}

/// 获取当前帧信息 (用于 JS 端轮询)
#[tauri::command]
async fn get_zerocopy_frame_info(
    zerocopy_state: State<'_, ZeroCopyRendererState>,
) -> Result<Option<FrameInfo>, String> {
    Ok(zerocopy_state.get_frame_info())
}

/// 读取当前帧 NV12 数据 (当无法使用共享内存时的回退方案)
/// 使用 Raw Response 避免 Base64 编码
#[tauri::command]
async fn read_zerocopy_frame(
    zerocopy_state: State<'_, ZeroCopyRendererState>,
) -> Result<tauri::ipc::Response, String> {
    match zerocopy_state.read_current_frame() {
        Some((y_data, uv_data, info)) => {
            // 打包: [header(20)] [y_data] [uv_data]
            // header: frame_id(8) width(4) height(4) y_stride(4) uv_stride(4) = 24 bytes
            let y_len = y_data.len();
            let uv_len = uv_data.len();
            let total_len = 24 + y_len + uv_len;

            let mut buf = Vec::with_capacity(total_len);

            // Header
            buf.extend_from_slice(&info.frame_id.to_le_bytes());
            buf.extend_from_slice(&info.width.to_le_bytes());
            buf.extend_from_slice(&info.height.to_le_bytes());
            buf.extend_from_slice(&info.y_stride.to_le_bytes());
            buf.extend_from_slice(&info.uv_stride.to_le_bytes());

            // Data
            buf.extend_from_slice(&y_data);
            buf.extend_from_slice(&uv_data);

            Ok(tauri::ipc::Response::new(buf))
        }
        None => Err("没有可用的帧".to_string()),
    }
}

// ==================== 摄像头/桌面采集 ====================

use capture::{CaptureDevice, CaptureDeviceType, CaptureState};
use shared_memory::{RgbaFrameInfo, RgbaSharedMemoryInfo};

/// 列出所有可用的采集设备 (摄像头 + 屏幕)
#[tauri::command]
async fn list_capture_devices(
    capture_state: State<'_, CaptureState>,
) -> Result<Vec<CaptureDevice>, String> {
    Ok(capture_state.list_devices())
}

/// 采集区域
#[derive(Debug, Clone, serde::Deserialize)]
pub struct CaptureRegionParam {
    pub x: i32, // 支持负数（多显示器）
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// 启动采集 (摄像头或桌面) - 返回共享内存信息
#[tauri::command]
async fn start_capture(
    app: AppHandle,
    capture_state: State<'_, CaptureState>,
    device_id: String,
    device_type: String,
    width: u32,
    height: u32,
    fps: u32,
    region: Option<CaptureRegionParam>,
) -> Result<RgbaSharedMemoryInfo, String> {
    let dtype = match device_type.as_str() {
        "camera" => CaptureDeviceType::Camera,
        "screen" => CaptureDeviceType::Screen,
        "window" => CaptureDeviceType::Window,
        _ => return Err(format!("未知设备类型: {}", device_type)),
    };

    // 转换 region 参数
    let capture_region = region.map(|r| capture::CaptureRegion {
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
    });

    println!(
        "🎥 启动采集: {} ({:?}) @ {}x{} {}fps, region: {:?}",
        device_id, dtype, width, height, fps, capture_region
    );

    let result = capture_state.start(&device_id, dtype, width, height, fps, capture_region);

    // 只有桌面采集才需要录制指示器，窗口采集不需要
    if result.is_ok() && dtype == CaptureDeviceType::Screen {
        if let Some(indicator) = app.get_webview_window("recording-indicator") {
            let _ = indicator.emit(
                "capture-state-changed",
                serde_json::json!({ "recording": true }),
            );
        }
    }

    result
}

/// 停止采集
#[tauri::command]
async fn stop_capture(
    app: AppHandle,
    capture_state: State<'_, CaptureState>,
) -> Result<String, String> {
    capture_state.stop();

    // 通知录制指示器切换到待机状态
    if let Some(indicator) = app.get_webview_window("recording-indicator") {
        let _ = indicator.emit(
            "capture-state-changed",
            serde_json::json!({ "recording": false }),
        );
    }

    println!("🛑 采集已停止");
    Ok("采集已停止".to_string())
}

/// 获取采集帧信息 (用于 JS 轮询)
#[tauri::command]
async fn get_capture_frame_info(
    capture_state: State<'_, CaptureState>,
) -> Result<Option<RgbaFrameInfo>, String> {
    Ok(capture_state.get_frame_info())
}

/// 读取采集帧 (回退方案，当共享内存不可用时)
#[tauri::command]
async fn read_capture_frame(
    capture_state: State<'_, CaptureState>,
) -> Result<tauri::ipc::Response, String> {
    match capture_state.read_current_frame() {
        Some((rgba_data, info)) => {
            // 打包: [header(24)] [rgba_data]
            // header: frame_id(8) width(4) height(4) timestamp(8) = 24 bytes
            let total_len = 24 + rgba_data.len();
            let mut buf = Vec::with_capacity(total_len);

            buf.extend_from_slice(&info.frame_id.to_le_bytes());
            buf.extend_from_slice(&info.width.to_le_bytes());
            buf.extend_from_slice(&info.height.to_le_bytes());
            buf.extend_from_slice(&info.timestamp.to_le_bytes());
            buf.extend_from_slice(&rgba_data);

            Ok(tauri::ipc::Response::new(buf))
        }
        None => Err("没有可用的帧".to_string()),
    }
}

/// 动态更新采集区域 (录制指示器拖动/调整大小时调用)
#[tauri::command]
async fn update_capture_region(
    capture_state: State<'_, CaptureState>,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    println!(
        "🔄 [update_capture_region] 收到更新请求: ({}, {}) {}x{}",
        x, y, width, height
    );
    capture_state.update_region_full(x, y, width, height);
    println!("✅ [update_capture_region] 区域已更新");
    Ok(())
}

/// 打开全屏区域选择窗口
#[tauri::command]
async fn open_region_selector(app: AppHandle) -> Result<(), String> {
    use tauri::WebviewUrl;
    use tauri::WebviewWindowBuilder;

    // 先最小化主窗口
    if let Some(main_window) = app.get_webview_window("main") {
        main_window.minimize().map_err(|e| e.to_string())?;
    }

    // 等待一小段时间让窗口最小化完成
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // 获取主显示器尺寸
    let monitors = app.available_monitors().map_err(|e| e.to_string())?;
    let primary = monitors.into_iter().next().ok_or("No monitor found")?;

    let size = primary.size();
    let position = primary.position();

    // 创建全屏半透明窗口
    let _selector_window = WebviewWindowBuilder::new(
        &app,
        "region-selector",
        WebviewUrl::App("region-selector.html".into()),
    )
    .title("选择区域")
    .position(position.x as f64, position.y as f64)
    .inner_size(size.width as f64, size.height as f64)
    .decorations(false)
    .transparent(true) // 启用透明背景
    .always_on_top(true)
    .skip_taskbar(true)
    .resizable(false)
    .focused(true)
    .visible(false)
    .build()
    .map_err(|e| e.to_string())?;

    Ok(())
}

/// 显示区域选择窗口 (页面加载完成后调用)
#[tauri::command]
async fn show_region_selector(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("region-selector") {
        window.show().map_err(|e| e.to_string())?;
        window.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 关闭区域选择窗口
#[tauri::command]
async fn close_region_selector(app: AppHandle) -> Result<(), String> {
    // 关闭选择器窗口
    if let Some(window) = app.get_webview_window("region-selector") {
        window.close().map_err(|e| e.to_string())?;
    }

    // 恢复主窗口
    if let Some(main_window) = app.get_webview_window("main") {
        main_window.unminimize().map_err(|e| e.to_string())?;
        main_window.set_focus().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 录制区域参数
#[derive(Clone, Serialize, Deserialize)]
pub struct RecordingRegionParam {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// 打开录制指示器窗口 (在屏幕采集区域显示闪烁边框)
#[tauri::command]
async fn open_recording_indicator(
    app: AppHandle,
    region: Option<RecordingRegionParam>,
) -> Result<(), String> {
    use tauri::WebviewUrl;
    use tauri::WebviewWindowBuilder;

    // 如果已存在，先关闭
    if let Some(window) = app.get_webview_window("recording-indicator") {
        let _ = window.close();
        // 等待窗口关闭
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    // 默认区域：屏幕中央 640x480
    let (x, y, width, height) = match region {
        Some(r) => (r.x, r.y, r.width, r.height),
        None => {
            // 获取屏幕尺寸
            let monitors = app.available_monitors().map_err(|e| e.to_string())?;
            let primary = monitors.into_iter().next().ok_or("No monitor found")?;
            let size = primary.size();
            let pos = primary.position();

            // 默认 640x480，居中
            let default_w = 640;
            let default_h = 480;
            let center_x = pos.x + (size.width as i32 - default_w) / 2;
            let center_y = pos.y + (size.height as i32 - default_h) / 2;
            (center_x, center_y, default_w as u32, default_h as u32)
        }
    };

    // 创建透明窗口覆盖在采集区域上
    let _indicator_window = WebviewWindowBuilder::new(
        &app,
        "recording-indicator",
        WebviewUrl::App("recording-indicator.html".into()),
    )
    .title("录制中")
    .position(x as f64, y as f64)
    .inner_size(width as f64, height as f64)
    .decorations(false)
    .transparent(true)
    .always_on_top(true)
    .skip_taskbar(true)
    .resizable(true) // 允许调整大小，通过 JS 手柄控制
    .focused(false)
    .visible(false)
    // 窗口不响应鼠标事件，让事件穿透到下层
    .build()
    .map_err(|e| e.to_string())?;

    Ok(())
}

/// 显示录制指示器窗口 (页面加载完成后调用)
#[tauri::command]
async fn show_recording_indicator(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("recording-indicator") {
        window.show().map_err(|e| e.to_string())?;
        // 注意：不再设置 WS_EX_TRANSPARENT，这样 REC 标签可以接收鼠标事件进行拖动
        // 但 HTML/CSS 中的 pointer-events: none 仍会让边框区域穿透
    }
    Ok(())
}

/// 关闭录制指示器窗口
#[tauri::command]
async fn close_recording_indicator(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("recording-indicator") {
        window.close().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 移动录制指示器窗口位置
#[tauri::command]
async fn move_recording_indicator(
    app: AppHandle,
    capture_state: State<'_, CaptureState>,
    x: i32,
    y: i32,
) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("recording-indicator") {
        window
            .set_position(tauri::Position::Physical(tauri::PhysicalPosition { x, y }))
            .map_err(|e| e.to_string())?;

        // 同时更新采集区域位置
        capture_state.update_region(x, y);
    }
    Ok(())
}

/// 调整录制指示器窗口大小和位置
#[tauri::command]
async fn resize_recording_indicator(
    app: AppHandle,
    capture_state: State<'_, CaptureState>,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("recording-indicator") {
        // 更新窗口位置和大小
        window
            .set_position(tauri::Position::Physical(tauri::PhysicalPosition { x, y }))
            .map_err(|e| e.to_string())?;
        window
            .set_size(tauri::Size::Physical(tauri::PhysicalSize { width, height }))
            .map_err(|e| e.to_string())?;

        // 同时更新采集区域
        capture_state.update_region_full(x, y, width, height);
    }
    Ok(())
}

/// 获取录制指示器窗口的当前位置和大小
#[tauri::command]
async fn get_recording_indicator_region(
    app: AppHandle,
) -> Result<Option<RecordingRegionParam>, String> {
    if let Some(window) = app.get_webview_window("recording-indicator") {
        let pos = window.outer_position().map_err(|e| e.to_string())?;
        let size = window.inner_size().map_err(|e| e.to_string())?;
        Ok(Some(RecordingRegionParam {
            x: pos.x,
            y: pos.y,
            width: size.width,
            height: size.height,
        }))
    } else {
        Ok(None)
    }
}

// ==================== 多窗口捕获命令 ====================

use multi_window_capture::{MultiWindowCaptureState, WindowCaptureInfo};

/// 开始捕获一个窗口 (支持同时捕获多个)
#[tauri::command]
async fn start_multi_window_capture(
    state: State<'_, MultiWindowCaptureState>,
    window_id: u32,
    fps: Option<u32>,
) -> Result<WindowCaptureInfo, String> {
    let fps = fps.unwrap_or(30);
    state.start_capture(window_id, fps)
}

/// 停止捕获一个窗口
#[tauri::command]
async fn stop_multi_window_capture(
    state: State<'_, MultiWindowCaptureState>,
    window_id: u32,
) -> Result<(), String> {
    state.stop_capture(window_id)
}

/// 停止所有窗口捕获
#[tauri::command]
async fn stop_all_multi_window_capture(
    state: State<'_, MultiWindowCaptureState>,
) -> Result<(), String> {
    state.stop_all();
    Ok(())
}

/// 获取窗口的帧信息
#[tauri::command]
async fn get_multi_window_frame_info(
    state: State<'_, MultiWindowCaptureState>,
    window_id: u32,
) -> Result<Option<RgbaFrameInfo>, String> {
    Ok(state.get_frame_info(window_id))
}

/// 读取窗口的当前帧
#[tauri::command]
async fn read_multi_window_frame(
    state: State<'_, MultiWindowCaptureState>,
    window_id: u32,
) -> Result<tauri::ipc::Response, String> {
    match state.read_frame(window_id) {
        Some((rgba_data, info)) => {
            // 打包: [header(24)] [rgba_data]
            let total_len = 24 + rgba_data.len();
            let mut buf = Vec::with_capacity(total_len);

            buf.extend_from_slice(&info.frame_id.to_le_bytes());
            buf.extend_from_slice(&info.width.to_le_bytes());
            buf.extend_from_slice(&info.height.to_le_bytes());
            buf.extend_from_slice(&info.timestamp.to_le_bytes());
            buf.extend_from_slice(&rgba_data);

            Ok(tauri::ipc::Response::new(buf))
        }
        None => Err(format!("窗口 {} 没有可用的帧", window_id)),
    }
}

/// 获取所有正在捕获的窗口 ID
#[tauri::command]
async fn get_active_window_captures(
    state: State<'_, MultiWindowCaptureState>,
) -> Result<Vec<u32>, String> {
    Ok(state.get_active_captures())
}

fn main() {
    // 添加 panic hook 以捕获 Rust 层面的崩溃
    panic::set_hook(Box::new(|info| {
        println!("🔥 程序发生严重错误 (Panic): {:?}", info);
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            println!("错误信息: {}", s);
        }
    }));

    // 初始化 RTSP 代理
    let proxy = Arc::new(rtsp_proxy::RtspProxy::new());
    // let proxy_clone = proxy.clone();

    // 在后台启动 WebSocket 服务器 (已切换为 IPC)
    // tauri::async_runtime::spawn(async move {
    //     proxy_clone.run_server(9001).await;
    // });

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // 设置窗口背景色为黑色
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_background_color(Some(Color(10, 10, 15, 255)));

                // 🔧 Windows: 禁用 WebView2 的可见性节流
                // 这样即使窗口被遮挡，WebView2 也会继续渲染
                #[cfg(windows)]
                {
                    use tauri::WebviewWindow;
                    // Tauri 2.x: 设置 WebView 不暂停渲染
                    // WebView2 默认在窗口不可见时会降低渲染频率
                    println!("🔧 配置 WebView2 在遮挡时继续渲染...");
                }

                // 监听主窗口关闭事件，关闭所有子窗口
                let app_handle = app.handle().clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { .. } = event {
                        // 关闭录制指示器窗口
                        if let Some(indicator) =
                            app_handle.get_webview_window("recording-indicator")
                        {
                            let _ = indicator.close();
                        }
                        // 关闭区域选择窗口
                        if let Some(selector) = app_handle.get_webview_window("region-selector") {
                            let _ = selector.close();
                        }
                    }
                });
            }
            Ok(())
        })
        .manage(VideoFrameState::default())
        .manage(ProxyState { proxy })
        .manage(AsyncDetectorState::default())
        .manage(ZeroCopyRendererState::default())
        .manage(CaptureState::default())
        .manage(multi_window_capture::MultiWindowCaptureState::default())
        .invoke_handler(tauri::generate_handler![
            get_latest_frame,
            has_new_frame,
            mark_frame_processed,
            start_rtsp_stream,
            log_frontend,
            show_window,
            read_text_file,
            get_rtsp_history,
            add_rtsp_history,
            clear_rtsp_history,
            // 设备和模型配置
            get_available_devices,
            get_available_models,
            // 检测器命令
            start_detector,
            stop_detector,
            detect_frame,
            // 零拷贝共享内存命令
            start_zerocopy_stream,
            stop_zerocopy_stream,
            get_zerocopy_frame_info,
            read_zerocopy_frame,
            // 摄像头/桌面采集命令
            list_capture_devices,
            start_capture,
            stop_capture,
            get_capture_frame_info,
            read_capture_frame,
            update_capture_region,
            // 区域选择命令
            open_region_selector,
            show_region_selector,
            close_region_selector,
            // 录制指示器命令
            open_recording_indicator,
            show_recording_indicator,
            close_recording_indicator,
            move_recording_indicator,
            resize_recording_indicator,
            get_recording_indicator_region,
            // 多窗口捕获命令
            start_multi_window_capture,
            stop_multi_window_capture,
            stop_all_multi_window_capture,
            get_multi_window_frame_info,
            read_multi_window_frame,
            get_active_window_captures,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
