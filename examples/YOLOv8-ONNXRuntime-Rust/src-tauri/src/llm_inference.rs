//! LLM 视频推理模块
//!
//! 按配置的间隔抽帧发送给 LLM 进行分析。

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::ipc::{Channel, InvokeResponseBody};

/// LLM 推理配置
#[derive(Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// API 端点
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
            system_prompt: "你是一个视频监控分析助手。请分析图像中的内容。".to_string(),
            user_prompt: "请分析这张监控画面".to_string(),
            frame_interval: 2.0,
            max_tokens: 500,
            temperature: 0.7,
        }
    }
}

/// 流式推理结果
#[derive(Clone, Serialize)]
pub struct StreamChunk {
    /// 类型: "start" | "chunk" | "end" | "error" | "frame_info"
    pub r#type: String,
    /// 文本内容
    pub content: String,
    /// 帧序号
    pub frame_id: u64,
    /// 时间戳
    pub timestamp: u64,
}

/// 帧缓冲区
struct FrameBuffer {
    width: u32,
    height: u32,
    frame_id: u64,
    timestamp: u64,
    rgba_data: Vec<u8>,
    has_new_frame: bool,
}

impl Default for FrameBuffer {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            frame_id: 0,
            timestamp: 0,
            rgba_data: Vec::new(),
            has_new_frame: false,
        }
    }
}

/// LLM 推理状态
pub struct LlmInferenceState {
    running: Arc<AtomicBool>,
    frame_count: Arc<AtomicU64>,
    config: Arc<RwLock<LlmConfig>>,
    result_channel: Arc<RwLock<Option<Channel<InvokeResponseBody>>>>,
    frame_buffer: Arc<RwLock<FrameBuffer>>,
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
            thread_handle: Arc::new(RwLock::new(None)),
        }
    }
}

impl LlmInferenceState {
    /// 启动 LLM 推理
    pub fn start(
        &self,
        config: LlmConfig,
        result_channel: Channel<InvokeResponseBody>,
    ) -> Result<String, String> {
        self.stop();

        *self.config.write() = config.clone();
        *self.result_channel.write() = Some(result_channel);

        self.running.store(true, Ordering::SeqCst);
        self.frame_count.store(0, Ordering::SeqCst);

        // 启动推理线程
        let running = self.running.clone();
        let frame_count = self.frame_count.clone();
        let config_arc = self.config.clone();
        let result_channel_arc = self.result_channel.clone();
        let frame_buffer_arc = self.frame_buffer.clone();
        let frame_interval = config.frame_interval;

        let handle = std::thread::spawn(move || {
            println!("🚀 LLM 推理线程启动");

            let interval = Duration::from_secs_f32(frame_interval);
            let mut last_inference_time = Instant::now() - Duration::from_secs(10);

            while running.load(Ordering::Relaxed) {
                let should_analyze = {
                    let buf = frame_buffer_arc.read();
                    buf.has_new_frame && last_inference_time.elapsed() >= interval
                };

                if should_analyze {
                    let (frame_id, width, height, timestamp, rgba_data) = {
                        let mut buf = frame_buffer_arc.write();
                        buf.has_new_frame = false;
                        (
                            buf.frame_id,
                            buf.width,
                            buf.height,
                            buf.timestamp,
                            buf.rgba_data.clone(),
                        )
                    };

                    if width == 0 || height == 0 || rgba_data.is_empty() {
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }

                    frame_count.fetch_add(1, Ordering::Relaxed);
                    last_inference_time = Instant::now();

                    if let Some(ref channel) = *result_channel_arc.read() {
                        let config = config_arc.read().clone();

                        // 发送帧信息 (JSON)
                        let frame_info = StreamChunk {
                            r#type: "frame_info".to_string(),
                            content: format!("分析帧 #{} ({}x{})", frame_id, width, height),
                            frame_id,
                            timestamp,
                        };
                        let _ = channel.send(InvokeResponseBody::Json(
                            serde_json::to_string(&frame_info).unwrap(),
                        ));

                        // 发送开始信号 (JSON)
                        let start_chunk = StreamChunk {
                            r#type: "start".to_string(),
                            content: String::new(),
                            frame_id,
                            timestamp,
                        };
                        let _ = channel.send(InvokeResponseBody::Json(
                            serde_json::to_string(&start_chunk).unwrap(),
                        ));

                        // TODO: 实际 LLM 调用，使用 rgba_data
                        println!(
                            "📸 正在分析帧 #{}, 大小: {}x{}, RGBA 数据: {} 字节",
                            frame_id,
                            width,
                            height,
                            rgba_data.len()
                        );

                        let chunk = StreamChunk {
                            r#type: "chunk".to_string(),
                            content: format!(
                                "[待实现 LLM 调用] 帧 #{}, 模型: {}, 图像尺寸: {}x{}",
                                frame_id, config.model, width, height
                            ),
                            frame_id,
                            timestamp,
                        };
                        let _ = channel.send(InvokeResponseBody::Json(
                            serde_json::to_string(&chunk).unwrap(),
                        ));

                        // 发送结束信号 (JSON)
                        let end_chunk = StreamChunk {
                            r#type: "end".to_string(),
                            content: String::new(),
                            frame_id,
                            timestamp,
                        };
                        let _ = channel.send(InvokeResponseBody::Json(
                            serde_json::to_string(&end_chunk).unwrap(),
                        ));
                    }
                }

                std::thread::sleep(Duration::from_millis(50));
            }

            println!("🛑 LLM 推理线程退出");
        });

        *self.thread_handle.write() = Some(handle);

        Ok(format!(
            "LLM 推理已启动，每 {:.1}s 分析一帧",
            frame_interval
        ))
    }

    /// 提交帧进行 LLM 推理
    pub fn submit_frame(&self, width: u32, height: u32, frame_id: u64, rgba_data: Vec<u8>) {
        let mut buf = self.frame_buffer.write();
        buf.width = width;
        buf.height = height;
        buf.frame_id = frame_id;
        buf.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        buf.rgba_data = rgba_data;
        buf.has_new_frame = true;
    }

    /// 停止推理
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

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
