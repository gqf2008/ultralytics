//! LLM 视频推理模块
//!
//! 通过 xbus 订阅解码帧事件，按配置的间隔抽帧发送给 LLM 进行分析。
//!
//! ## 架构
//!
//! ```text
//! Demuxer ──xbus──→ DecodedFrameEvent ──订阅──→ LlmInference
//!                                                    │
//!                                                    ├─→ 抽帧
//!                                                    ├─→ JPEG/Base64 编码
//!                                                    ├─→ LLM API 调用
//!                                                    └─→ 流式输出 (Tauri Channel)
//! ```

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::ipc::Channel;

use crate::video_events::{DecodedFrameEvent, RgbaFrameReadyEvent};
use crate::xbus::{self, Subscription};

/// LLM 推理配置
#[derive(Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API 端点 (如 OpenAI, Ollama, 本地服务)
    pub api_url: String,
    /// API Key (可选)
    pub api_key: Option<String>,
    /// 模型名称
    pub model: String,
    /// 系统提示词
    pub system_prompt: String,
    /// 用户提示词模板
    pub user_prompt: String,
    /// 抽帧间隔 (秒)
    pub frame_interval: f32,
    /// 最大 tokens
    pub max_tokens: u32,
    /// 温度
    pub temperature: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:11434/api/chat".to_string(),
            api_key: None,
            model: "llava".to_string(),
            system_prompt: "你是一个视频监控分析助手。请分析图像中的内容，描述你看到的场景、人物、物体和活动。".to_string(),
            user_prompt: "请分析这张监控画面".to_string(),
            frame_interval: 2.0,
            max_tokens: 500,
            temperature: 0.7,
        }
    }
}

/// 流式推理结果 (发送给前端)
#[derive(Clone, Serialize)]
pub struct StreamChunk {
    /// 类型: "start" | "chunk" | "end" | "error" | "frame_info"
    pub r#type: String,
    /// 文本内容
    pub content: String,
    /// 帧序号
    pub frame_id: u64,
    /// 时间戳 (毫秒)
    pub timestamp: u64,
}

/// 帧缓冲区 (用于保存最新帧供 LLM 分析)
struct FrameBuffer {
    /// RGBA 数据
    rgba_data: Vec<u8>,
    /// 宽度
    width: u32,
    /// 高度
    height: u32,
    /// 帧 ID
    frame_id: u64,
    /// 时间戳
    timestamp: u64,
    /// 是否有新帧
    has_new_frame: bool,
}

impl Default for FrameBuffer {
    fn default() -> Self {
        Self {
            rgba_data: Vec::new(),
            width: 0,
            height: 0,
            frame_id: 0,
            timestamp: 0,
            has_new_frame: false,
        }
    }
}

/// LLM 推理状态
pub struct LlmInferenceState {
    /// 是否正在运行
    running: Arc<AtomicBool>,
    /// 已处理帧数
    frame_count: Arc<AtomicU64>,
    /// 当前配置
    config: Arc<RwLock<LlmConfig>>,
    /// 结果输出 Channel
    result_channel: Arc<RwLock<Option<Channel<StreamChunk>>>>,
    /// 帧缓冲区
    frame_buffer: Arc<RwLock<FrameBuffer>>,
    /// xbus 订阅凭证 (保持存活)
    _subscriptions: Arc<RwLock<Vec<Subscription>>>,
    /// 推理线程句柄
    thread_handle: Arc<RwLock<Option<std::thread::JoinHandle<()>>>>,
}

impl Default for LlmInferenceState {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            frame_count: Arc::new(AtomicU64::new(0)),
            config: Arc::new(RwLock::new(LlmConfig::default())),
            result_channel: Arc::new(RwLock::new(None)),
            frame_buffer: Arc::new(RwLock::new(FrameBuffer::default())),
            _subscriptions: Arc::new(RwLock::new(Vec::new())),
            thread_handle: Arc::new(RwLock::new(None)),
        }
    }
}

impl LlmInferenceState {
    /// 启动 LLM 推理
    ///
    /// 订阅 xbus 的 RgbaFrameReadyEvent，按配置间隔分析帧
    pub fn start(
        &self,
        config: LlmConfig,
        result_channel: Channel<StreamChunk>,
    ) -> Result<String, String> {
        // 停止之前的任务
        self.stop();

        // 更新配置和 channel
        *self.config.write() = config.clone();
        *self.result_channel.write() = Some(result_channel);

        self.running.store(true, Ordering::SeqCst);
        self.frame_count.store(0, Ordering::SeqCst);

        // 订阅 RGBA 帧就绪事件
        let frame_buffer = self.frame_buffer.clone();
        let sub = xbus::subscribe::<RgbaFrameReadyEvent, _>(move |event| {
            // 这里只保存帧信息，实际数据在共享内存中
            // 由于共享内存访问需要额外实现，这里先记录元信息
            let mut buf = frame_buffer.write();
            buf.frame_id = event.frame_id;
            buf.width = event.width;
            buf.height = event.height;
            buf.timestamp = event.timestamp;
            buf.has_new_frame = true;
            // 实际的 RGBA 数据需要从共享内存读取
            // TODO: 集成 shared_memory 模块
        });

        // 也订阅 DecodedFrameEvent (NV12 帧)
        let frame_buffer2 = self.frame_buffer.clone();
        let sub2 = xbus::subscribe::<DecodedFrameEvent, _>(move |event| {
            let mut buf = frame_buffer2.write();
            buf.frame_id = event.frame_id;
            buf.width = event.width;
            buf.height = event.height;
            buf.timestamp = event.timestamp;
            buf.has_new_frame = true;
        });

        // 保存订阅凭证
        *self._subscriptions.write() = vec![sub, sub2];

        // 启动推理线程
        let running = self.running.clone();
        let frame_count = self.frame_count.clone();
        let config_arc = self.config.clone();
        let result_channel_arc = self.result_channel.clone();
        let frame_buffer_arc = self.frame_buffer.clone();
        let frame_interval = config.frame_interval;

        let handle = std::thread::spawn(move || {
            println!("🚀 LLM 推理线程启动");

            let client = ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .build();

            let interval = Duration::from_secs_f32(frame_interval);
            let mut last_inference_time = Instant::now() - Duration::from_secs(10);

            while running.load(Ordering::Relaxed) {
                // 检查是否有新帧且达到分析间隔
                let should_analyze = {
                    let buf = frame_buffer_arc.read();
                    buf.has_new_frame && last_inference_time.elapsed() >= interval
                };

                if should_analyze {
                    // 获取帧信息并标记为已处理
                    let (frame_id, width, height, timestamp) = {
                        let mut buf = frame_buffer_arc.write();
                        buf.has_new_frame = false;
                        (buf.frame_id, buf.width, buf.height, buf.timestamp)
                    };

                    if width == 0 || height == 0 {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }

                    frame_count.fetch_add(1, Ordering::Relaxed);
                    last_inference_time = Instant::now();

                    // 发送帧信息
                    if let Some(ref channel) = *result_channel_arc.read() {
                        let _ = channel.send(StreamChunk {
                            r#type: "frame_info".to_string(),
                            content: format!("分析帧 #{} ({}x{})", frame_id, width, height),
                            frame_id,
                            timestamp,
                        });

                        // TODO: 从共享内存读取 RGBA 数据，转换为 JPEG
                        // 目前先发送一个占位消息
                        let _ = channel.send(StreamChunk {
                            r#type: "start".to_string(),
                            content: String::new(),
                            frame_id,
                            timestamp,
                        });

                        // 调用 LLM (占位实现)
                        let config = config_arc.read().clone();
                        
                        // TODO: 实现实际的 LLM 调用
                        // 需要从共享内存读取帧数据 → JPEG → Base64 → LLM API
                        let _ = channel.send(StreamChunk {
                            r#type: "chunk".to_string(),
                            content: format!("[等待共享内存集成] 帧 #{} 已就绪，模型: {}", frame_id, config.model),
                            frame_id,
                            timestamp,
                        });

                        let _ = channel.send(StreamChunk {
                            r#type: "end".to_string(),
                            content: String::new(),
                            frame_id,
                            timestamp,
                        });
                    }
                }

                std::thread::sleep(Duration::from_millis(50));
            }

            println!("🛑 LLM 推理线程退出");
        });

        *self.thread_handle.write() = Some(handle);

        Ok(format!("LLM 推理已启动，每 {:.1}s 分析一帧", frame_interval))
    }

    /// 停止推理
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        // 清空订阅 (自动取消订阅)
        self._subscriptions.write().clear();

        // 等待线程退出
        let mut handle = self.thread_handle.write();
        if let Some(h) = handle.take() {
            let _ = h.join();
        }
    }

    /// 获取状态
    pub fn get_status(&self) -> (bool, u64) {
        (
            self.running.load(Ordering::Relaxed),
            self.frame_count.load(Ordering::Relaxed),
        )
    }

    /// 更新配置
    pub fn update_config(&self, config: LlmConfig) {
        *self.config.write() = config;
    }
}

/// NV12 转 RGBA
pub fn nv12_to_rgba(
    y_data: &[u8],
    uv_data: &[u8],
    width: u32,
    height: u32,
    y_stride: usize,
    uv_stride: usize,
) -> Vec<u8> {
    let mut rgba = vec![0u8; (width * height * 4) as usize];

    for y in 0..height {
        for x in 0..width {
            let y_idx = (y as usize) * y_stride + (x as usize);
            let uv_idx = ((y / 2) as usize) * uv_stride + ((x / 2) as usize) * 2;

            let y_val = y_data.get(y_idx).copied().unwrap_or(128) as f32;
            let u_val = uv_data.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;
            let v_val = uv_data.get(uv_idx + 1).copied().unwrap_or(128) as f32 - 128.0;

            // YUV → RGB (BT.601)
            let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
            let g = (y_val - 0.344 * u_val - 0.714 * v_val).clamp(0.0, 255.0) as u8;
            let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = 255;
        }
    }

    rgba
}

/// RGBA 转 JPEG
pub fn rgba_to_jpeg(rgba_data: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
    use image::{ImageBuffer, Rgba};

    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_data.to_vec())
        .ok_or("Invalid RGBA data dimensions")?;

    // 缩放到合适大小 (LLM 不需要超高分辨率)
    let max_dim = 1024u32;
    let (new_width, new_height) = if width > max_dim || height > max_dim {
        let scale = (max_dim as f32) / (width.max(height) as f32);
        ((width as f32 * scale) as u32, (height as f32 * scale) as u32)
    } else {
        (width, height)
    };

    let resized = image::imageops::resize(
        &img,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    );

    // 转换为 RGB 并编码 JPEG
    let rgb_img = image::DynamicImage::ImageRgba8(resized).to_rgb8();
    
    let mut jpeg_data = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_data, quality);
    encoder
        .encode_image(&rgb_img)
        .map_err(|e| format!("JPEG encode failed: {}", e))?;

    Ok(jpeg_data)
}

/// 调用 LLM API (流式输出)
pub fn call_llm_stream(
    client: &ureq::Agent,
    config: &LlmConfig,
    base64_image: &str,
    frame_id: u64,
    timestamp: u64,
    channel: &Channel<StreamChunk>,
) -> Result<(), String> {
    // 构建请求体 (Ollama 格式)
    let request_body = serde_json::json!({
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": config.system_prompt
            },
            {
                "role": "user",
                "content": config.user_prompt,
                "images": [base64_image]
            }
        ],
        "stream": true,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens
        }
    });

    // 发送请求
    let mut request = client.post(&config.api_url);
    
    if let Some(ref api_key) = config.api_key {
        request = request.set("Authorization", &format!("Bearer {}", api_key));
    }

    let response = request
        .set("Content-Type", "application/json")
        .send_json(&request_body)
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    // 读取流式响应
    let reader = std::io::BufReader::new(response.into_reader());
    use std::io::BufRead;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read response failed: {}", e))?;
        
        if line.is_empty() {
            continue;
        }

        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
            // Ollama 格式
            if let Some(message) = json.get("message") {
                if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                    if !content.is_empty() {
                        let _ = channel.send(StreamChunk {
                            r#type: "chunk".to_string(),
                            content: content.to_string(),
                            frame_id,
                            timestamp,
                        });
                    }
                }
            }
            
            // OpenAI 格式
            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                if let Some(delta) = choices.first().and_then(|c| c.get("delta")) {
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            let _ = channel.send(StreamChunk {
                                r#type: "chunk".to_string(),
                                content: content.to_string(),
                                frame_id,
                                timestamp,
                            });
                        }
                    }
                }
            }

            if json.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                break;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LlmConfig::default();
        assert_eq!(config.frame_interval, 2.0);
        assert!(config.api_url.contains("11434"));
    }
}
