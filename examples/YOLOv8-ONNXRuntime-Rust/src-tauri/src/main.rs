// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod demuxer;
mod llm_inference;
mod xbus;

use demuxer::{DemuxerConfig, DemuxerHandle, EncodedPacket, StreamMetadata};
use llm_inference::{LlmConfig, LlmInferenceState};
use parking_lot::RwLock;
use serde::Serialize;
use std::panic;
use std::sync::Arc;
use tauri::ipc::{Channel, InvokeBody, InvokeResponseBody, Request};
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
/// 使用 InvokeResponseBody 支持高效传输：
/// - Json: 视频配置等结构化数据
/// - Raw: 编码视频数据包 (避免 JSON 序列化开销)
#[tauri::command]
async fn start_rtsp_stream(
    app: AppHandle,
    url: String,
    on_data: Channel<InvokeResponseBody>,
) -> Result<String, String> {
    println!("🚀 启动 RTSP 流 (WebCodecs 模式): {}", url);

    let demuxer_state = app.state::<DemuxerState>();

    // 元数据回调：发送视频配置 (JSON)
    let on_data_meta = on_data.clone();
    let metadata_callback = Arc::new(move |meta: StreamMetadata| {
        println!(
            "📹 视频流元数据: {} {}x{} @ {:.2} fps",
            meta.codec, meta.width, meta.height, meta.fps
        );
        let config_msg = VideoConfigMessage {
            r#type: "video_config".to_string(),
            codec: meta.codec,
            width: meta.width,
            height: meta.height,
            extradata: meta.extradata,
        };
        // 配置信息用 JSON 发送
        let _ = on_data_meta.send(InvokeResponseBody::Json(
            serde_json::to_string(&config_msg).unwrap_or_default(),
        ));
    });

    // 数据包回调：发送编码数据 (Raw 二进制)
    let packet_callback = Arc::new(move |packet: EncodedPacket| {
        // 构建高效的二进制包格式:
        // [0]     : type (1=video, 2=audio)
        // [1]     : is_keyframe (0/1)
        // [2..10] : pts (i64, little-endian)
        // [10..18]: dts (i64, little-endian)
        // [18..26]: packet_id (u64, little-endian)
        // [26..30]: data_len (u32, little-endian)
        // [30..]  : data

        let header_size = 30;
        let total_size = header_size + packet.data.len();
        let mut buffer = vec![0u8; total_size];

        buffer[0] = 1; // type: video
        buffer[1] = if packet.is_keyframe { 1 } else { 0 };
        buffer[2..10].copy_from_slice(&packet.pts.to_le_bytes());
        buffer[10..18].copy_from_slice(&packet.dts.to_le_bytes());
        buffer[18..26].copy_from_slice(&packet.packet_id.to_le_bytes());
        buffer[26..30].copy_from_slice(&(packet.data.len() as u32).to_le_bytes());
        buffer[30..].copy_from_slice(&packet.data);

        // 首包打印日志
        if packet.packet_id == 1 {
            println!(
                "📦 发送首个视频包: keyframe={}, size={} bytes",
                packet.is_keyframe, total_size
            );
        }

        // 使用 Raw 发送二进制数据，避免 JSON 序列化开销
        let result = on_data.send(InvokeResponseBody::Raw(buffer));
        if packet.packet_id == 1 {
            println!("📤 Channel 发送结果: {:?}", result);
        }
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
///
/// result_channel 使用 InvokeResponseBody 支持高效传输：
/// - Json: 结构化数据 (StreamChunk)
/// - Raw: 二进制数据 (如需要返回处理后的图像)
#[tauri::command]
async fn start_llm_inference(
    app: AppHandle,
    config: LlmConfig,
    result_channel: Channel<InvokeResponseBody>,
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
/// 接收前端传来的 RGBA 视频帧进行 LLM 推理
///
/// 前端通过 Raw body 传递 RGBA 数据，格式：
/// - 前 16 字节: header (width: u32, height: u32, frame_id: u64)
/// - 剩余字节: RGBA 像素数据 (width * height * 4)
#[tauri::command]
async fn submit_frame_for_llm(app: AppHandle, request: Request<'_>) -> Result<(), String> {
    match request.body() {
        InvokeBody::Raw(data) => {
            if data.len() < 16 {
                return Err("数据太短，缺少 header".to_string());
            }

            // 解析 header
            let width = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let height = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let frame_id = u64::from_le_bytes([
                data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            ]);

            let expected_size = 16 + (width * height * 4) as usize;
            if data.len() < expected_size {
                return Err(format!(
                    "数据不完整: 期望 {} 字节，实际 {} 字节",
                    expected_size,
                    data.len()
                ));
            }

            let rgba_data = &data[16..expected_size];

            println!(
                "📸 收到 RGBA 帧: {}x{}, frame_id={}, 数据大小={} 字节",
                width,
                height,
                frame_id,
                rgba_data.len()
            );

            // 通知 LLM 推理模块有新帧
            let llm_state = app.state::<LlmInferenceState>();
            llm_state.submit_frame(width, height, frame_id, rgba_data.to_vec());

            Ok(())
        }
        InvokeBody::Json(_) => Err("请使用 Raw body 传递 RGBA 数据".to_string()),
    }
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
            submit_frame_for_llm,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
