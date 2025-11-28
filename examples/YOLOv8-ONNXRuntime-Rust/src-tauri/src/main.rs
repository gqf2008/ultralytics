// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod demuxer;
mod llm_inference;
mod xbus;

use demuxer::{DemuxerConfig, DemuxerHandle, EncodedPacket, StreamMetadata};
use llm_inference::{LlmConfig, LlmInferenceState, StreamChunk};
use parking_lot::RwLock;
use serde::Serialize;
use std::panic;
use std::sync::Arc;
use tauri::ipc::Channel;
use tauri::window::Color;
use tauri::{AppHandle, Manager};

/// 解复用器状态
pub struct DemuxerState {
    pub handle: RwLock<DemuxerHandle>,
}

// ==================== 数据包消息 (发送给前端 WebCodecs) ====================

/// 视频配置消息
#[derive(Clone, Serialize)]
pub struct VideoConfigMessage {
    pub r#type: String,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub extradata: Vec<u8>,
}

/// 视频数据包消息
#[derive(Clone, Serialize)]
pub struct VideoDataMessage {
    pub r#type: String,
    pub data: Vec<u8>,
    pub is_keyframe: bool,
    pub pts: i64,
    pub dts: i64,
    pub packet_id: u64,
}

/// 通用流消息
#[derive(Clone, Serialize)]
#[serde(untagged)]
pub enum StreamMessage {
    Config(VideoConfigMessage),
    Data(VideoDataMessage),
}

// ==================== RTSP 流命令 (WebCodecs 模式) ====================

/// 启动 RTSP 流 (WebCodecs 模式)
///
/// 后端只做解复用，发送编码数据给前端，由前端 WebCodecs 解码
#[tauri::command]
async fn start_rtsp_stream(
    app: AppHandle,
    url: String,
    on_data: Channel<StreamMessage>,
) -> Result<String, String> {
    println!("🚀 启动 RTSP 流 (WebCodecs 模式): {}", url);

    let demuxer_state = app.state::<DemuxerState>();

    // 元数据回调：发送视频配置
    let on_data_meta = on_data.clone();
    let metadata_callback = Arc::new(move |meta: StreamMetadata| {
        println!(
            "📹 视频流元数据: {} {}x{} @ {:.2} fps",
            meta.codec, meta.width, meta.height, meta.fps
        );
        let _ = on_data_meta.send(StreamMessage::Config(VideoConfigMessage {
            r#type: "video_config".to_string(),
            codec: meta.codec,
            width: meta.width,
            height: meta.height,
            extradata: meta.extradata,
        }));
    });

    // 数据包回调：发送编码数据
    let packet_callback = Arc::new(move |packet: EncodedPacket| {
        let _ = on_data.send(StreamMessage::Data(VideoDataMessage {
            r#type: "video".to_string(),
            data: packet.data,
            is_keyframe: packet.is_keyframe,
            pts: packet.pts,
            dts: packet.dts,
            packet_id: packet.packet_id,
        }));
    });

    let config = DemuxerConfig {
        url: url.clone(),
        connect_timeout: 10,
        read_timeout: 5,
        metadata_callback: Some(metadata_callback),
        packet_callback: Some(packet_callback),
    };

    demuxer_state.handle.write().start(config)?;

    Ok(format!("RTSP 流已启动 (WebCodecs): {}", url))
}

/// 停止 RTSP 流
#[tauri::command]
async fn stop_rtsp_stream(app: AppHandle) -> Result<(), String> {
    println!("⏹ 停止 RTSP 流");

    let demuxer_state = app.state::<DemuxerState>();
    demuxer_state.handle.write().stop();

    Ok(())
}

/// 获取 RTSP 流状态
#[tauri::command]
async fn get_rtsp_status(app: AppHandle) -> Result<(bool, u64), String> {
    let demuxer_state = app.state::<DemuxerState>();
    let handle = demuxer_state.handle.read();
    Ok((handle.is_running(), handle.packet_count()))
}

// ==================== 历史记录命令 ====================

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

// ==================== 工具命令 ====================

/// 前端日志透传
#[tauri::command]
async fn log_frontend(msg: String) {
    println!("[Frontend]: {}", msg);
}

/// 显示主窗口
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

// ==================== LLM 推理命令 ====================

/// 启动 LLM 视频推理
#[tauri::command]
async fn start_llm_inference(
    app: AppHandle,
    config: LlmConfig,
    result_channel: Channel<StreamChunk>,
) -> Result<String, String> {
    println!("🤖 启动 LLM 推理: model={}", config.model);

    let llm_state = app.state::<LlmInferenceState>();
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

// ==================== 主函数 ====================

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
        .manage(DemuxerState {
            handle: RwLock::new(DemuxerHandle::new()),
        })
        .manage(LlmInferenceState::default())
        .invoke_handler(tauri::generate_handler![
            // RTSP 流命令 (WebCodecs 模式)
            start_rtsp_stream,
            stop_rtsp_stream,
            get_rtsp_status,
            // 历史记录
            get_rtsp_history,
            add_rtsp_history,
            clear_rtsp_history,
            // 工具命令
            log_frontend,
            show_window,
            read_text_file,
            // LLM 推理命令
            start_llm_inference,
            stop_llm_inference,
            get_llm_status,
            update_llm_config,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
