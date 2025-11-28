//! LLM 视频推理模块
//!
//! 按配置的间隔抽帧发送给 LLM 进行分析。

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::ipc::Channel;

use crate::xbus::{self, Subscription};

/// RGBA 帧就绪事件 (用于 xbus 通信)
#[derive(Clone, Debug)]
pub struct RgbaFrameReadyEvent {
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}

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
    has_new_frame: bool,
}

impl Default for FrameBuffer {
    fn default() -> Self {
        Self {
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
    running: Arc<AtomicBool>,
    frame_count: Arc<AtomicU64>,
    config: Arc<RwLock<LlmConfig>>,
    result_channel: Arc<RwLock<Option<Channel<StreamChunk>>>>,
    frame_buffer: Arc<RwLock<FrameBuffer>>,
    _subscriptions: Arc<RwLock<Vec<Subscription>>>,
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
    pub fn start(
        &self,
        config: LlmConfig,
        result_channel: Channel<StreamChunk>,
    ) -> Result<String, String> {
        self.stop();

        *self.config.write() = config.clone();
        *self.result_channel.write() = Some(result_channel);

        self.running.store(true, Ordering::SeqCst);
        self.frame_count.store(0, Ordering::SeqCst);

        // 订阅 RGBA 帧就绪事件
        let frame_buffer = self.frame_buffer.clone();
        let sub = xbus::subscribe::<RgbaFrameReadyEvent, _>(move |event| {
            let mut buf = frame_buffer.write();
            buf.frame_id = event.frame_id;
            buf.width = event.width;
            buf.height = event.height;
            buf.timestamp = event.timestamp;
            buf.has_new_frame = true;
        });

        *self._subscriptions.write() = vec![sub];

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

                    if let Some(ref channel) = *result_channel_arc.read() {
                        let config = config_arc.read().clone();

                        let _ = channel.send(StreamChunk {
                            r#type: "frame_info".to_string(),
                            content: format!("分析帧 #{} ({}x{})", frame_id, width, height),
                            frame_id,
                            timestamp,
                        });

                        let _ = channel.send(StreamChunk {
                            r#type: "start".to_string(),
                            content: String::new(),
                            frame_id,
                            timestamp,
                        });

                        // TODO: 实际 LLM 调用
                        let _ = channel.send(StreamChunk {
                            r#type: "chunk".to_string(),
                            content: format!("[待实现] 帧 #{}, 模型: {}", frame_id, config.model),
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

        Ok(format!(
            "LLM 推理已启动，每 {:.1}s 分析一帧",
            frame_interval
        ))
    }

    /// 停止推理
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self._subscriptions.write().clear();

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
