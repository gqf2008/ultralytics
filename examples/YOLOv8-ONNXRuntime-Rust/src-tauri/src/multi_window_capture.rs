//! 多窗口捕获模块 - 支持同时捕获多个窗口
//!
//! 特性:
//! - 同时捕获多个窗口
//! - 每个窗口独立的共享内存缓冲区
//! - 自动为浏览器窗口使用 WGC 捕获（避免视频暂停）

use crate::shared_memory::{RgbaFrameInfo, RgbaSharedBuffer};
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use xcap::Window;
#[cfg(windows)]
use yolov8_sentinel_lib::wgc_capture::{self, WgcCapture};

/// 捕获方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CaptureMethod {
    /// 使用 xcap (BitBlt)
    Xcap,
    /// 使用 Windows Graphics Capture
    Wgc,
}

/// 单个窗口的捕获会话
struct WindowCaptureSession {
    running: Arc<AtomicBool>,
    shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
    capture_thread: Option<JoinHandle<()>>,
    capture_method: CaptureMethod,
}

/// 窗口捕获信息
#[derive(Debug, Clone, Serialize)]
pub struct WindowCaptureInfo {
    pub window_id: u32,
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub shm_name: String,
    pub shm_size: usize,
    pub capture_method: CaptureMethod,
}

/// 多窗口捕获管理器
pub struct MultiWindowCaptureState {
    sessions: RwLock<HashMap<u32, WindowCaptureSession>>,
}

impl Default for MultiWindowCaptureState {
    fn default() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
        }
    }
}

impl MultiWindowCaptureState {
    /// 开始捕获一个窗口
    pub fn start_capture(&self, window_id: u32, fps: u32) -> Result<WindowCaptureInfo, String> {
        self.start_capture_with_method(window_id, fps, None)
    }

    /// 开始捕获一个窗口，可指定捕获方法
    /// 如果 method 为 None，会自动选择（浏览器使用 WGC）
    pub fn start_capture_with_method(
        &self,
        window_id: u32,
        fps: u32,
        method: Option<CaptureMethod>,
    ) -> Result<WindowCaptureInfo, String> {
        // 检查是否已在捕获
        if self.sessions.read().contains_key(&window_id) {
            return Err(format!("窗口 {} 已在捕获中", window_id));
        }

        // 查找窗口
        let windows = Window::all().map_err(|e| format!("获取窗口列表失败: {}", e))?;
        let window = windows
            .into_iter()
            .find(|w| w.id().unwrap_or(0) == window_id)
            .ok_or_else(|| format!("窗口 {} 不存在", window_id))?;

        let title = window.title().unwrap_or_else(|_| "Unknown".to_string());
        let width = window.width().unwrap_or(640);
        let height = window.height().unwrap_or(480);

        // 自动选择捕获方法
        let capture_method = method.unwrap_or_else(|| select_capture_method(&title));

        println!(
            "🪟 开始捕获窗口: {} (ID: {}, {}x{}, 方法: {:?})",
            title, window_id, width, height, capture_method
        );

        // 创建共享内存
        let shm_name = format!("window_capture_{}_{}", std::process::id(), window_id);
        let shared_buffer = RgbaSharedBuffer::create(&shm_name, width, height)?;
        let shm_info = shared_buffer.get_info();

        let running = Arc::new(AtomicBool::new(true));
        let shared_buffer = Arc::new(RwLock::new(Some(shared_buffer)));

        // 启动捕获线程
        let capture_thread = {
            let running = running.clone();
            let shared_buffer = shared_buffer.clone();

            match capture_method {
                CaptureMethod::Wgc => {
                    #[cfg(windows)]
                    {
                        thread::spawn(move || {
                            if let Err(e) =
                                run_wgc_window_capture(window_id, running, shared_buffer, fps)
                            {
                                eprintln!("❌ WGC 窗口 {} 捕获错误: {}", window_id, e);
                            }
                        })
                    }
                    #[cfg(not(windows))]
                    {
                        return Err("WGC 只支持 Windows".to_string());
                    }
                }
                CaptureMethod::Xcap => thread::spawn(move || {
                    if let Err(e) = run_xcap_window_capture(window_id, running, shared_buffer, fps)
                    {
                        eprintln!("❌ xcap 窗口 {} 捕获错误: {}", window_id, e);
                    }
                }),
            }
        };

        // 保存会话
        let session = WindowCaptureSession {
            running,
            shared_buffer,
            capture_thread: Some(capture_thread),
            capture_method,
        };

        self.sessions.write().insert(window_id, session);

        Ok(WindowCaptureInfo {
            window_id,
            title,
            width,
            height,
            shm_name: shm_info.name,
            shm_size: shm_info.total_size,
            capture_method,
        })
    }

    /// 停止捕获一个窗口
    pub fn stop_capture(&self, window_id: u32) -> Result<(), String> {
        let mut sessions = self.sessions.write();
        if let Some(mut session) = sessions.remove(&window_id) {
            println!("🛑 停止捕获窗口: {}", window_id);

            // 停止运行
            session.running.store(false, Ordering::SeqCst);

            // 等待线程结束
            if let Some(thread) = session.capture_thread.take() {
                let _ = thread.join();
            }

            // 清理共享内存
            *session.shared_buffer.write() = None;

            Ok(())
        } else {
            Err(format!("窗口 {} 未在捕获中", window_id))
        }
    }

    /// 停止所有捕获
    pub fn stop_all(&self) {
        let window_ids: Vec<u32> = self.sessions.read().keys().cloned().collect();
        for window_id in window_ids {
            let _ = self.stop_capture(window_id);
        }
    }

    /// 获取所有正在捕获的窗口 ID
    pub fn get_active_captures(&self) -> Vec<u32> {
        self.sessions.read().keys().cloned().collect()
    }

    /// 获取窗口的帧信息
    pub fn get_frame_info(&self, window_id: u32) -> Option<RgbaFrameInfo> {
        let sessions = self.sessions.read();
        let session = sessions.get(&window_id)?;
        let buffer = session.shared_buffer.read();
        let shm = buffer.as_ref()?;
        let header = shm.header();

        Some(RgbaFrameInfo {
            frame_id: header.write_frame_id.load(Ordering::Acquire),
            width: header.width,
            height: header.height,
            timestamp: header.timestamp.load(Ordering::Acquire),
        })
    }

    /// 读取窗口的当前帧
    pub fn read_frame(&self, window_id: u32) -> Option<(Vec<u8>, RgbaFrameInfo)> {
        let sessions = self.sessions.read();
        let session = sessions.get(&window_id)?;
        let buffer = session.shared_buffer.read();
        let shm = buffer.as_ref()?;
        let header = shm.header();

        let info = RgbaFrameInfo {
            frame_id: header.write_frame_id.load(Ordering::Acquire),
            width: header.width,
            height: header.height,
            timestamp: header.timestamp.load(Ordering::Acquire),
        };

        let rgba_data = shm.current_read_buffer().to_vec();
        Some((rgba_data, info))
    }
}

impl Drop for MultiWindowCaptureState {
    fn drop(&mut self) {
        self.stop_all();
    }
}

/// 根据窗口标题自动选择捕获方法
/// 浏览器窗口使用 WGC (避免视频暂停问题)
fn select_capture_method(title: &str) -> CaptureMethod {
    let title_lower = title.to_lowercase();

    // 检测常见浏览器
    let is_browser = title_lower.contains("chrome")
        || title_lower.contains("edge")
        || title_lower.contains("firefox")
        || title_lower.contains("opera")
        || title_lower.contains("brave")
        || title_lower.contains("vivaldi")
        || title_lower.contains("chromium")
        || title_lower.ends_with(" - 个人")  // Edge 中文
        || title_lower.ends_with(" - personal"); // Edge 英文

    #[cfg(windows)]
    if is_browser && wgc_capture::is_wgc_supported() {
        return CaptureMethod::Wgc;
    }

    CaptureMethod::Xcap
}

/// 使用 xcap 的窗口捕获线程
fn run_xcap_window_capture(
    window_id: u32,
    running: Arc<AtomicBool>,
    shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
    fps: u32,
) -> Result<(), String> {
    // 查找窗口
    let windows = Window::all().map_err(|e| format!("获取窗口列表失败: {}", e))?;
    let window = windows
        .into_iter()
        .find(|w| w.id().unwrap_or(0) == window_id)
        .ok_or_else(|| format!("窗口 {} 不存在", window_id))?;

    let window_title = window.title().unwrap_or_else(|_| "Unknown".to_string());
    println!("📺 xcap 捕获线程启动: {} (ID: {})", window_title, window_id);

    let start_time = Instant::now();
    let mut frame_count: u64 = 0;
    let mut last_stats_update = Instant::now();
    let frame_interval = Duration::from_millis(1000 / fps.max(1) as u64);
    let mut last_frame_time = Instant::now();

    while running.load(Ordering::SeqCst) {
        // 帧率控制
        let elapsed = last_frame_time.elapsed();
        if elapsed < frame_interval {
            thread::sleep(frame_interval - elapsed);
        }
        last_frame_time = Instant::now();

        // 捕获窗口截图
        let image = match window.capture_image() {
            Ok(img) => img,
            Err(e) => {
                // 窗口可能已关闭或最小化
                println!("⚠️ xcap 窗口 {} 截图失败: {}", window_id, e);
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };

        let frame_width = image.width();
        let frame_height = image.height();
        let rgba_data = image.into_raw();

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // 写入共享内存
        if let Some(ref mut shm) = *shared_buffer.write() {
            shm.write_frame(&rgba_data, frame_width, frame_height, timestamp);
        }

        frame_count += 1;

        if last_stats_update.elapsed() >= Duration::from_secs(5) {
            let elapsed = start_time.elapsed().as_secs_f32();
            let current_fps = frame_count as f32 / elapsed;
            last_stats_update = Instant::now();
            println!(
                "🪟 xcap 窗口 {} 帧数: {}, FPS: {:.1}",
                window_id, frame_count, current_fps
            );
        }
    }

    println!("🛑 xcap 窗口 {} 捕获线程已停止", window_id);
    Ok(())
}

/// 使用 WGC 的窗口捕获线程
#[cfg(windows)]
fn run_wgc_window_capture(
    window_id: u32,
    running: Arc<AtomicBool>,
    shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
    fps: u32,
) -> Result<(), String> {
    println!("📺 WGC 捕获线程启动: 窗口 ID {}", window_id);

    // 创建 WGC 捕获器
    let capture =
        WgcCapture::new(window_id as isize).map_err(|e| format!("创建 WGC 捕获器失败: {:?}", e))?;

    capture
        .start()
        .map_err(|e| format!("启动 WGC 捕获失败: {:?}", e))?;

    let start_time = Instant::now();
    let mut frame_count: u64 = 0;
    let mut last_stats_update = Instant::now();
    let frame_interval = Duration::from_millis(1000 / fps.max(1) as u64);
    let mut last_frame_time = Instant::now();

    while running.load(Ordering::SeqCst) {
        // 帧率控制
        let elapsed = last_frame_time.elapsed();
        if elapsed < frame_interval {
            thread::sleep(frame_interval - elapsed);
        }
        last_frame_time = Instant::now();

        // 获取 WGC 帧
        let Some(frame) = capture.try_get_frame() else {
            // 没有新帧，继续等待
            thread::sleep(Duration::from_millis(1));
            continue;
        };

        // 写入共享内存
        if let Some(ref mut shm) = *shared_buffer.write() {
            shm.write_frame(&frame.data, frame.width, frame.height, frame.timestamp);
        }

        frame_count += 1;

        if last_stats_update.elapsed() >= Duration::from_secs(5) {
            let elapsed = start_time.elapsed().as_secs_f32();
            let current_fps = frame_count as f32 / elapsed;
            last_stats_update = Instant::now();
            println!(
                "🪟 WGC 窗口 {} 帧数: {}, FPS: {:.1}",
                window_id, frame_count, current_fps
            );
        }
    }

    capture.stop();
    println!("🛑 WGC 窗口 {} 捕获线程已停止", window_id);
    Ok(())
}
