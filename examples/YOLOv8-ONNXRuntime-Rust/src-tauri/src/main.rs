// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod rtsp_proxy;
mod video_bridge;

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

/// 启动 RTSP 流
#[tauri::command]
async fn start_rtsp_stream(
    app: AppHandle,
    url: String,
    channel: tauri::ipc::Channel,
) -> Result<String, String> {
    println!("🚀 启动 RTSP 流: {}", url);

    // 增加解码器代数
    let generation = DECODER_GENERATION.fetch_add(1, Ordering::SeqCst);
    println!("📌 新解码器代数: {}", generation);

    // 更新主项目的全局代数计数器
    yolov8_rs::input::decoder_manager::ACTIVE_DECODER_GENERATION
        .store(generation, Ordering::Relaxed);

    // 启动 RTSP 代理 (用于前端显示)
    let proxy_state = app.state::<ProxyState>();
    proxy_state.proxy.start(url.clone(), Some(channel));

    Ok(format!("RTSP 流已启动 (Gen: {})", generation))
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
        .invoke_handler(tauri::generate_handler![
            get_latest_frame,
            has_new_frame,
            mark_frame_processed,
            start_rtsp_stream,
            log_frontend,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
