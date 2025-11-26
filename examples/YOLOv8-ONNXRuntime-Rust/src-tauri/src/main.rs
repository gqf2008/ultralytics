// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod detector;
mod detector_async;
mod rtsp_proxy;

use detector_async::AsyncDetectorState;
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::panic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};

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
    println!("🖥️ [Frontend]: {}", msg);
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
        .manage(VideoFrameState::default())
        .manage(ProxyState { proxy })
        .manage(AsyncDetectorState::default())
        .invoke_handler(tauri::generate_handler![
            get_latest_frame,
            has_new_frame,
            mark_frame_processed,
            start_rtsp_stream,
            log_frontend,
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
