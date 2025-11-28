// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod demuxer;
mod detector_async;
mod llm_inference;
mod qsv_decoder;
mod shared_memory;
mod video_events;
mod xbus;

use demuxer::{DemuxerConfig, DemuxerHandle, FrameCallback};
use detector_async::AsyncDetectorState;
use llm_inference::{LlmConfig, LlmInferenceState, StreamChunk};
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use shared_memory::{RgbaSharedBuffer, RgbaFrameHeader};
use std::panic;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tauri::window::Color;
use tauri::{AppHandle, Manager, State};

/// 解复用器状态
pub struct DemuxerState {
    pub handle: RwLock<DemuxerHandle>,
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

/// 共享内存状态 - 用于零拷贝帧传输
pub struct SharedMemoryState {
    pub buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
    pub name: Arc<RwLock<String>>,
}

impl Default for SharedMemoryState {
    fn default() -> Self {
        Self {
            buffer: Arc::new(RwLock::new(None)),
            name: Arc::new(RwLock::new(String::new())),
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

/// 获取共享内存信息 (用于前端零拷贝渲染)
#[tauri::command]
async fn get_shared_memory_info(
    state: State<'_, SharedMemoryState>,
) -> Result<SharedMemInfo, String> {
    let name = state.name.read().clone();
    let buffer = state.buffer.read();
    
    if let Some(ref shm) = *buffer {
        let header = shm.header();
        Ok(SharedMemInfo {
            name,
            frame_id: header.write_frame_id.load(Ordering::Acquire),
            width: header.width,
            height: header.height,
            max_width: header.max_width,
            max_height: header.max_height,
            timestamp: header.timestamp.load(Ordering::Acquire),
        })
    } else {
        Err("共享内存未初始化".to_string())
    }
}

/// 共享内存信息结构
#[derive(Serialize)]
pub struct SharedMemInfo {
    pub name: String,
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub max_width: u32,
    pub max_height: u32,
    pub timestamp: u64,
}

/// 从共享内存读取当前帧 (IPC 回退方案)
#[tauri::command]
async fn get_shared_memory_frame(
    state: State<'_, SharedMemoryState>,
) -> Result<(Vec<u8>, u32, u32, u64), String> {
    let buffer = state.buffer.read();
    
    if let Some(ref shm) = *buffer {
        let header = shm.header();
        let frame_id = header.write_frame_id.load(Ordering::Acquire);
        let width = header.width;
        let height = header.height;
        
        // 读取当前帧数据
        let frame_data = shm.current_read_buffer().to_vec();
        
        Ok((frame_data, width, height, frame_id))
    } else {
        Err("共享内存未初始化".to_string())
    }
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
        Err(_) => Ok(Vec::new()),
    }
}

/// 添加 RTSP 历史记录
#[tauri::command]
async fn add_rtsp_history(url: String) -> Result<(), String> {
    let path = "rtsp_history.txt";

    let mut urls = match std::fs::read_to_string(path) {
        Ok(content) => content
            .lines()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
        Err(_) => Vec::new(),
    };

    urls.retain(|u| u != &url);
    urls.insert(0, url);

    if urls.len() > 20 {
        urls.truncate(20);
    }

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
///
/// # 参数
/// * `url` - RTSP URL
/// * `hardware_decode` - 是否使用硬件解码 (false 则使用 WebCodecs)
/// * `video_channel` - 视频帧输出通道
#[tauri::command]
async fn start_rtsp_stream(
    app: AppHandle,
    url: String,
    hardware_decode: Option<bool>,
) -> Result<String, String> {
    let hw = hardware_decode.unwrap_or(true);
    println!("🚀 启动 RTSP 流: {} (硬件解码: {})", url, hw);

    let generation = DECODER_GENERATION.fetch_add(1, Ordering::SeqCst);
    println!("📌 新解码器代数: {}", generation);

    let demuxer_state = app.state::<DemuxerState>();
    let shm_state = app.state::<SharedMemoryState>();
    let video_state = app.state::<VideoFrameState>();

    // 创建共享内存 (4K 尺寸预分配)
    let shm_name = format!("yolo_rgba_{}", std::process::id());
    let shm_buffer = RgbaSharedBuffer::create(&shm_name, 3840, 2160)
        .map_err(|e| format!("创建共享内存失败: {}", e))?;
    
    *shm_state.name.write() = shm_name.clone();
    *shm_state.buffer.write() = Some(shm_buffer);
    
    println!("✅ 创建共享内存: {}", shm_name);

    // 创建帧回调，将帧数据写入共享内存
    let shm_buffer_arc = shm_state.buffer.clone();
    // 同时保留 IPC 回退 (用于兼容)
    let buffer = video_state.buffer.clone();
    let width_state = video_state.width.clone();
    let height_state = video_state.height.clone();
    let updated = video_state.updated.clone();

    let frame_callback: FrameCallback = Arc::new(move |frame| {
        // 写入共享内存 (零拷贝)
        if let Some(ref mut shm) = *shm_buffer_arc.write() {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            shm.write_frame(&frame.data, frame.width, frame.height, timestamp);
        }
        
        // IPC 回退 (保留兼容性)
        *buffer.write() = frame.data;
        *width_state.write() = frame.width;
        *height_state.write() = frame.height;
        *updated.write() = true;
    });

    let config = DemuxerConfig {
        url: url.clone(),
        use_hw_decode: hw,
        hw_type: "qsv".to_string(),
        output_rgba: true,
        shm_name_prefix: "yolo_frame".to_string(),
        connect_timeout: 10,
        read_timeout: 5,
        frame_callback: Some(frame_callback),
    };

    demuxer_state.handle.write().start(config)?;

    Ok(format!("RTSP 流已启动: {}", url))
}

/// 停止 RTSP 流
#[tauri::command]
async fn stop_rtsp_stream(app: AppHandle) -> Result<(), String> {
    println!("⏹ 停止 RTSP 流");
    
    // 停止解码器
    let demuxer_state = app.state::<DemuxerState>();
    demuxer_state.handle.write().stop();
    
    // 清理共享内存
    let shm_state = app.state::<SharedMemoryState>();
    *shm_state.buffer.write() = None;
    *shm_state.name.write() = String::new();
    println!("🗑️ 共享内存已释放");
    
    Ok(())
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

    let cuda = CUDAExecutionProvider::default();
    let cuda_available = cuda.is_available().unwrap_or(false);
    devices.push(DeviceInfo {
        id: "cuda".to_string(),
        name: "CUDA (NVIDIA GPU)".to_string(),
        available: cuda_available,
    });

    let dml = DirectMLExecutionProvider::default();
    let dml_available = dml.is_available().unwrap_or(false);
    devices.push(DeviceInfo {
        id: "directml".to_string(),
        name: "DirectML (Windows GPU)".to_string(),
        available: dml_available,
    });

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

    let model_path = if model.contains(std::path::MAIN_SEPARATOR)
        || model.contains('/')
        || model.ends_with(".onnx")
    {
        if std::path::Path::new(&model).exists() {
            std::path::PathBuf::from(&model)
        } else {
            std::env::current_exe()
                .map_err(|e| format!("获取当前exe路径失败: {}", e))?
                .parent()
                .map(|p| p.join("models").join(&model))
                .unwrap_or_else(|| std::path::PathBuf::from("models").join(&model))
        }
    } else {
        let model_filename = format!("{}.onnx", model);
        std::env::current_exe()
            .map_err(|e| format!("获取当前exe路径失败: {}", e))?
            .parent()
            .map(|p| p.join("models").join(&model_filename))
            .unwrap_or_else(|| std::path::PathBuf::from("models").join(&model_filename))
    };

    let model_path_str = model_path.to_string_lossy().to_string();
    println!("📁 模型路径: {}", model_path_str);

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
    let rgba_data = match request.body() {
        tauri::ipc::InvokeBody::Raw(data) => data.clone(),
        tauri::ipc::InvokeBody::Json(_) => {
            return Err("期望 Raw body，收到 JSON".to_string());
        }
    };

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

    if width != 640 || height != 640 {
        println!("⚠️ detect_frame: 收到 {}x{} (期望 640x640)", width, height);
    }

    let detector_state = app.state::<AsyncDetectorState>();
    detector_state.send_frame(rgba_data, width, height)
}

// ==================== LLM 推理命令 ====================

/// 启动 LLM 视频推理
#[tauri::command]
async fn start_llm_inference(
    app: AppHandle,
    config: LlmConfig,
    result_channel: tauri::ipc::Channel<StreamChunk>,
) -> Result<String, String> {
    println!("🤖 启动 LLM 推理: model={}", config.model);

    let llm_state = app.state::<LlmInferenceState>();
    // LLM 推理订阅 xbus 事件，不需要 RTSP URL
    llm_state.start(config, result_channel)
}

/// 停止 LLM 推理
#[tauri::command]
async fn stop_llm_inference(app: AppHandle) -> Result<String, String> {
    let llm_state = app.state::<LlmInferenceState>();
    llm_state.stop();
    Ok("LLM 推理已停止".to_string())
}

/// 获取 LLM 推理状态
#[tauri::command]
async fn get_llm_status(app: AppHandle) -> Result<(bool, u64), String> {
    let llm_state = app.state::<LlmInferenceState>();
    Ok(llm_state.get_status())
}

/// 更新 LLM 配置
#[tauri::command]
async fn update_llm_config(app: AppHandle, config: LlmConfig) -> Result<(), String> {
    let llm_state = app.state::<LlmInferenceState>();
    llm_state.update_config(config);
    Ok(())
}

fn main() {
    panic::set_hook(Box::new(|info| {
        println!("🔥 程序发生严重错误 (Panic): {:?}", info);
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            println!("错误信息: {}", s);
        }
    }));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_background_color(Some(Color(10, 10, 15, 255)));
            }
            Ok(())
        })
        .manage(VideoFrameState::default())
        .manage(SharedMemoryState::default())
        .manage(DemuxerState {
            handle: RwLock::new(DemuxerHandle::new()),
        })
        .manage(AsyncDetectorState::default())
        .manage(LlmInferenceState::default())
        .invoke_handler(tauri::generate_handler![
            get_latest_frame,
            has_new_frame,
            mark_frame_processed,
            get_shared_memory_info,
            get_shared_memory_frame,
            start_rtsp_stream,
            stop_rtsp_stream,
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
            // LLM 推理命令
            start_llm_inference,
            stop_llm_inference,
            get_llm_status,
            update_llm_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
