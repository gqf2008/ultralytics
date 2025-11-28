//! 跨平台摄像头捕获和桌面采集模块 - 共享内存版本
//!
//! - 摄像头: nokhwa (Windows/macOS/Linux)
//! - 桌面: xcap (跨平台屏幕采集)
//! - 窗口: Windows Graphics Capture API (支持被遮挡窗口，浏览器视频不暂停)
//! - 共享内存: 零拷贝传输 RGBA 帧给前端
//!
//! 架构:
//! ```text
//! ┌─────────────┐     ┌──────────────────┐    ┌───────────┐
//! │  nokhwa/    │     │   共享内存       │    │  Canvas   │
//! │  xcap/WGC   │────▶│  RGBA 双缓冲    │───▶│  渲染     │
//! │  采集       │写入  │                  │读取 │           │
//! └─────────────┘     └──────────────────┘    └───────────┘
//! ```

use crate::shared_memory::{RgbaFrameInfo, RgbaSharedBuffer, RgbaSharedMemoryInfo};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
#[cfg(windows)]
use yolov8_sentinel_lib::wgc_capture::WgcCapture;

// ==================== 数据结构 ====================

/// 捕获设备信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureDevice {
    pub id: String,
    pub name: String,
    pub device_type: CaptureDeviceType,
}

/// 设备类型
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CaptureDeviceType {
    Camera,
    Screen,
    Window,
}

/// 捕获统计信息
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CaptureStats {
    pub fps: f32,
    pub frame_count: u64,
    pub dropped_frames: u64,
    pub latency_ms: f32,
}

/// 采集区域
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CaptureRegion {
    pub x: i32, // 支持负数（多显示器）
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

// ==================== nokhwa 摄像头采集 ====================

mod camera {
    use super::*;
    use nokhwa::pixel_format::RgbAFormat;
    use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    /// 列出所有摄像头
    pub fn list_cameras() -> Vec<CaptureDevice> {
        match nokhwa::query(ApiBackend::Auto) {
            Ok(cameras) => {
                let devices: Vec<_> = cameras
                    .iter()
                    .map(|info| {
                        let index = match info.index() {
                            CameraIndex::Index(i) => i.to_string(),
                            CameraIndex::String(s) => s.clone(),
                        };
                        println!("✅ 发现摄像头: {} (index: {})", info.human_name(), index);
                        CaptureDevice {
                            id: index,
                            name: info.human_name(),
                            device_type: CaptureDeviceType::Camera,
                        }
                    })
                    .collect();
                println!("📷 共找到 {} 个摄像头", devices.len());
                devices
            }
            Err(e) => {
                println!("⚠️ 枚举摄像头失败: {}", e);
                vec![]
            }
        }
    }

    /// 摄像头采集线程
    #[allow(clippy::too_many_arguments)]
    pub fn run_camera_capture(
        device_id: String,
        width: u32,
        height: u32,
        _fps: u32,
        running: Arc<AtomicBool>,
        shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
        stats: Arc<RwLock<CaptureStats>>,
    ) -> Result<(), String> {
        let index = device_id
            .parse::<u32>()
            .map(CameraIndex::Index)
            .unwrap_or(CameraIndex::Index(0));

        println!(
            "🎥 打开摄像头: index={:?}, 目标尺寸: {}x{}",
            index, width, height
        );

        let requested =
            RequestedFormat::new::<RgbAFormat>(RequestedFormatType::AbsoluteHighestResolution);

        let mut camera =
            Camera::new(index, requested).map_err(|e| format!("打开摄像头失败: {}", e))?;

        camera
            .open_stream()
            .map_err(|e| format!("启动摄像头流失败: {}", e))?;

        println!("✅ 摄像头已打开");

        let mut frame_count: u64 = 0;
        let start_time = Instant::now();
        let mut last_stats_update = Instant::now();

        while running.load(Ordering::SeqCst) {
            match camera.frame() {
                Ok(frame) => {
                    match frame.decode_image::<RgbAFormat>() {
                        Ok(decoded) => {
                            let w = frame.resolution().width();
                            let h = frame.resolution().height();
                            let timestamp = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64;

                            // 写入共享内存
                            if let Some(ref mut buffer) = *shared_buffer.write() {
                                buffer.write_frame(&decoded.into_raw(), w, h, timestamp);
                            }

                            frame_count += 1;

                            if last_stats_update.elapsed() >= Duration::from_secs(1) {
                                let elapsed = start_time.elapsed().as_secs_f32();
                                let mut s = stats.write();
                                s.fps = frame_count as f32 / elapsed;
                                s.frame_count = frame_count;
                                last_stats_update = Instant::now();
                                println!("📷 摄像头帧数: {}, FPS: {:.1}", frame_count, s.fps);
                            }
                        }
                        Err(e) => {
                            eprintln!("解码帧失败: {}", e);
                        }
                    }
                }
                Err(e) => {
                    if !e.to_string().contains("Would block") {
                        eprintln!("获取帧失败: {}", e);
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }

        let _ = camera.stop_stream();
        println!("🛑 摄像头已停止");
        Ok(())
    }
}

// ==================== 桌面采集 (跨平台: xcap) ====================

mod screen {
    use super::*;
    use xcap::{Monitor, Window};

    /// 列出所有屏幕
    pub fn list_screens() -> Vec<CaptureDevice> {
        let mut devices = Vec::new();

        match Monitor::all() {
            Ok(monitors) => {
                for (i, monitor) in monitors.iter().enumerate() {
                    let width = monitor.width().unwrap_or(1920);
                    let height = monitor.height().unwrap_or(1080);
                    let is_primary = monitor.is_primary().unwrap_or(false);
                    let name = monitor.name().unwrap_or_else(|_| format!("Monitor {}", i));

                    if is_primary {
                        println!("✅ 发现主显示器: {} ({}x{})", name, width, height);
                        devices.insert(
                            0,
                            CaptureDevice {
                                id: format!("primary:{}:{}x{}", i, width, height),
                                name: format!("主显示器 ({}x{})", width, height),
                                device_type: CaptureDeviceType::Screen,
                            },
                        );
                    } else {
                        println!("✅ 发现显示器 {}: {} ({}x{})", i, name, width, height);
                        devices.push(CaptureDevice {
                            id: format!("display:{}:{}x{}", i, width, height),
                            name: format!("显示器 {} ({}x{})", i + 1, width, height),
                            device_type: CaptureDeviceType::Screen,
                        });
                    }
                }
            }
            Err(e) => {
                println!("❌ 获取显示器列表失败: {}", e);
            }
        }

        println!("🖥️ 共找到 {} 个显示器", devices.len());
        devices
    }

    /// 列出所有可见窗口
    pub fn list_windows() -> Vec<CaptureDevice> {
        let mut devices = Vec::new();

        match Window::all() {
            Ok(windows) => {
                println!("🔍 xcap 发现 {} 个原始窗口", windows.len());

                for window in windows.iter() {
                    let title = window.title().unwrap_or_else(|_| String::from(""));
                    let is_minimized = window.is_minimized().unwrap_or(true);
                    let width = window.width().unwrap_or(0);
                    let height = window.height().unwrap_or(0);

                    // 调试输出所有窗口
                    if !title.is_empty() {
                        println!(
                            "   - 窗口: {} | {}x{} | 最小化: {}",
                            if title.len() > 50 {
                                format!("{}...", &title[..47])
                            } else {
                                title.clone()
                            },
                            width,
                            height,
                            is_minimized
                        );
                    }

                    // 跳过最小化的窗口
                    if is_minimized {
                        continue;
                    }

                    // 跳过没有标题的窗口
                    if title.is_empty() {
                        continue;
                    }

                    // 跳过太小的窗口 (降低阈值以包含更多窗口)
                    if width < 50 || height < 50 {
                        continue;
                    }

                    let window_id = match window.id() {
                        Ok(id) => id,
                        Err(_) => continue,
                    };

                    // 截断过长的标题
                    let display_title = if title.len() > 50 {
                        format!("{}...", &title[..47])
                    } else {
                        title.clone()
                    };

                    devices.push(CaptureDevice {
                        id: format!("window:{}:{}x{}", window_id, width, height),
                        name: format!("🪟 {} ({}x{})", display_title, width, height),
                        device_type: CaptureDeviceType::Window,
                    });
                }
            }
            Err(e) => {
                println!("❌ 获取窗口列表失败: {}", e);
            }
        }

        println!("🪟 共找到 {} 个可捕获窗口", devices.len());
        devices
    }

    /// 从设备 ID 解析分辨率
    pub fn parse_screen_resolution(device_id: &str) -> (u32, u32) {
        // device_id 格式: "primary:0:1920x1080" 或 "display:1:1920x1080"
        if let Some(res_part) = device_id.rsplit(':').next() {
            if let Some((w, h)) = res_part.split_once('x') {
                if let (Ok(width), Ok(height)) = (w.parse(), h.parse()) {
                    return (width, height);
                }
            }
        }
        (1920, 1080) // 默认
    }

    /// 从设备 ID 解析显示器索引
    fn parse_monitor_index(device_id: &str) -> usize {
        // device_id 格式: "primary:0:1920x1080" 或 "display:1:1920x1080"
        let parts: Vec<&str> = device_id.split(':').collect();
        if parts.len() >= 2 {
            parts[1].parse().unwrap_or(0)
        } else {
            0
        }
    }

    /// 从窗口设备 ID 解析窗口 ID
    pub fn parse_window_id(device_id: &str) -> u32 {
        // device_id 格式: "window:12345:1920x1080"
        let parts: Vec<&str> = device_id.split(':').collect();
        if parts.len() >= 2 {
            parts[1].parse().unwrap_or(0)
        } else {
            0
        }
    }

    /// 桌面采集线程 (使用 xcap - 无边框)
    pub fn run_screen_capture(
        device_id: String,
        running: Arc<AtomicBool>,
        shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
        stats: Arc<RwLock<CaptureStats>>,
        region: Arc<RwLock<Option<CaptureRegion>>>,
    ) -> Result<(), String> {
        println!("🖥️ 启动桌面采集 (xcap - 无边框模式)...");

        if let Some(r) = region.read().as_ref() {
            println!("📐 采集区域: ({}, {}) {}x{}", r.x, r.y, r.width, r.height);
        } else {
            println!("📐 全屏采集");
        }

        // 获取指定显示器
        let monitor_index = parse_monitor_index(&device_id);
        let monitors = Monitor::all().map_err(|e| format!("获取显示器列表失败: {}", e))?;

        let monitor = monitors
            .into_iter()
            .nth(monitor_index)
            .ok_or_else(|| format!("显示器 {} 不存在", monitor_index))?;

        let monitor_width = monitor
            .width()
            .map_err(|e| format!("获取显示器宽度失败: {}", e))?;
        let monitor_height = monitor
            .height()
            .map_err(|e| format!("获取显示器高度失败: {}", e))?;

        println!("📺 显示器分辨率: {}x{}", monitor_width, monitor_height);

        // 使用 video_recorder 进行连续采集
        let (video_recorder, frame_rx) = monitor
            .video_recorder()
            .map_err(|e| format!("创建视频录制器失败: {}", e))?;

        // 启动录制
        video_recorder
            .start()
            .map_err(|e| format!("启动录制失败: {}", e))?;
        println!("✅ 桌面采集已启动 (xcap video_recorder)");

        let start_time = Instant::now();
        let mut frame_count: u64 = 0;
        let mut last_stats_update = Instant::now();

        // 帧处理循环
        while running.load(Ordering::SeqCst) {
            match frame_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(frame) => {
                    let frame_width = frame.width;
                    let frame_height = frame.height;

                    // 获取 RGBA 数据 (xcap 使用 raw 字段)
                    let rgba_data = frame.raw;

                    // 每帧读取最新的区域配置（支持动态更新）
                    let current_region = region.read().clone();

                    let (final_data, out_width, out_height) = if let Some(r) = current_region {
                        // 有区域选择，进行裁剪
                        let start_x = r.x.max(0) as u32;
                        let start_y = r.y.max(0) as u32;
                        let start_x = start_x.min(frame_width.saturating_sub(1));
                        let start_y = start_y.min(frame_height.saturating_sub(1));
                        let end_x = (start_x + r.width).min(frame_width);
                        let end_y = (start_y + r.height).min(frame_height);

                        let rw = end_x - start_x;
                        let rh = end_y - start_y;

                        if rw < 10 || rh < 10 {
                            continue; // 区域太小，跳过
                        }

                        // 手动裁剪
                        let mut cropped = Vec::with_capacity((rw * rh * 4) as usize);
                        for y in start_y..end_y {
                            let row_start = (y * frame_width * 4 + start_x * 4) as usize;
                            let row_end = row_start + (rw * 4) as usize;
                            if row_end <= rgba_data.len() {
                                cropped.extend_from_slice(&rgba_data[row_start..row_end]);
                            }
                        }
                        (cropped, rw, rh)
                    } else {
                        // 全屏
                        (rgba_data, frame_width, frame_height)
                    };

                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64;

                    // 写入共享内存
                    if let Some(ref mut shm) = *shared_buffer.write() {
                        shm.write_frame(&final_data, out_width, out_height, timestamp);
                    }

                    frame_count += 1;

                    if last_stats_update.elapsed() >= Duration::from_secs(1) {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let mut s = stats.write();
                        s.fps = frame_count as f32 / elapsed;
                        s.frame_count = frame_count;
                        last_stats_update = Instant::now();
                        println!("🖥️ 桌面帧数: {}, FPS: {:.1}", frame_count, s.fps);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // 超时，继续等待
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    println!("⚠️ 帧接收通道已断开");
                    break;
                }
            }
        }

        // 停止录制
        if let Err(e) = video_recorder.stop() {
            println!("⚠️ 停止录制失败: {}", e);
        }

        println!("🛑 桌面采集已停止");
        Ok(())
    }

    /// 窗口采集线程 - Windows 使用 WGC (浏览器视频不暂停)，其他平台使用 xcap
    pub fn run_window_capture(
        device_id: String,
        running: Arc<AtomicBool>,
        shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
        stats: Arc<RwLock<CaptureStats>>,
        region: Arc<RwLock<Option<CaptureRegion>>>,
        fps: u32,
    ) -> Result<(), String> {
        // Windows 上使用 WGC API
        #[cfg(windows)]
        {
            run_window_capture_wgc(device_id, running, shared_buffer, stats, region, fps)
        }

        // 非 Windows 平台使用 xcap
        #[cfg(not(windows))]
        {
            run_window_capture_xcap(device_id, running, shared_buffer, stats, region, fps)
        }
    }

    /// Windows: 使用 WGC 捕获窗口 (浏览器视频不暂停)
    #[cfg(windows)]
    fn run_window_capture_wgc(
        device_id: String,
        running: Arc<AtomicBool>,
        shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
        stats: Arc<RwLock<CaptureStats>>,
        region: Arc<RwLock<Option<CaptureRegion>>>,
        fps: u32,
    ) -> Result<(), String> {
        println!("🪟 启动窗口采集 (WGC - 浏览器视频不暂停)...");

        if let Some(r) = region.read().as_ref() {
            println!("📐 采集区域: ({}, {}) {}x{}", r.x, r.y, r.width, r.height);
        } else {
            println!("📐 全窗口采集");
        }

        // 获取窗口 ID
        let window_id = parse_window_id(&device_id);
        println!("📺 目标窗口 ID: {}", window_id);

        // 创建 WGC 捕获器
        let capture = WgcCapture::new(window_id as isize)
            .map_err(|e| format!("创建 WGC 捕获器失败: {:?}", e))?;

        capture
            .start()
            .map_err(|e| format!("启动 WGC 捕获失败: {:?}", e))?;

        let start_time = Instant::now();
        let mut frame_count: u64 = 0;
        let mut last_stats_update = Instant::now();
        let frame_interval = Duration::from_millis(1000 / fps.max(1) as u64);
        let mut last_frame_time = Instant::now();

        // 帧处理循环
        while running.load(Ordering::SeqCst) {
            // 帧率控制
            let elapsed = last_frame_time.elapsed();
            if elapsed < frame_interval {
                thread::sleep(frame_interval - elapsed);
            }
            last_frame_time = Instant::now();

            // 获取 WGC 帧
            let Some(frame) = capture.try_get_frame() else {
                thread::sleep(Duration::from_millis(1));
                continue;
            };

            let frame_width = frame.width;
            let frame_height = frame.height;
            let rgba_data = frame.data;

            // 每帧读取最新的区域配置（支持动态更新）
            let current_region = region.read().clone();

            let (final_data, out_width, out_height) = if let Some(r) = current_region {
                // 有区域选择，进行裁剪
                let start_x = r.x.max(0) as u32;
                let start_y = r.y.max(0) as u32;
                let start_x = start_x.min(frame_width.saturating_sub(1));
                let start_y = start_y.min(frame_height.saturating_sub(1));
                let end_x = (start_x + r.width).min(frame_width);
                let end_y = (start_y + r.height).min(frame_height);

                let rw = end_x - start_x;
                let rh = end_y - start_y;

                if rw < 10 || rh < 10 {
                    continue; // 区域太小，跳过
                }

                // 手动裁剪
                let mut cropped = Vec::with_capacity((rw * rh * 4) as usize);
                for y in start_y..end_y {
                    let row_start = (y * frame_width * 4 + start_x * 4) as usize;
                    let row_end = row_start + (rw * 4) as usize;
                    if row_end <= rgba_data.len() {
                        cropped.extend_from_slice(&rgba_data[row_start..row_end]);
                    }
                }
                (cropped, rw, rh)
            } else {
                // 全窗口
                (rgba_data, frame_width, frame_height)
            };

            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // 写入共享内存
            if let Some(ref mut shm) = *shared_buffer.write() {
                shm.write_frame(&final_data, out_width, out_height, timestamp);
            }

            frame_count += 1;

            if last_stats_update.elapsed() >= Duration::from_secs(5) {
                let elapsed = start_time.elapsed().as_secs_f32();
                let mut s = stats.write();
                s.fps = frame_count as f32 / elapsed;
                s.frame_count = frame_count;
                last_stats_update = Instant::now();
                println!("🪟 WGC 窗口帧数: {}, FPS: {:.1}", frame_count, s.fps);
            }
        }

        capture.stop();
        println!("🛑 WGC 窗口采集已停止");
        Ok(())
    }

    /// 非 Windows: 使用 xcap 轮询模式
    #[cfg(not(windows))]
    fn run_window_capture_xcap(
        device_id: String,
        running: Arc<AtomicBool>,
        shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
        stats: Arc<RwLock<CaptureStats>>,
        region: Arc<RwLock<Option<CaptureRegion>>>,
        fps: u32,
    ) -> Result<(), String> {
        println!("🪟 启动窗口采集 (xcap - 轮询模式)...");

        if let Some(r) = region.read().as_ref() {
            println!("📐 采集区域: ({}, {}) {}x{}", r.x, r.y, r.width, r.height);
        } else {
            println!("📐 全窗口采集");
        }

        // 获取指定窗口
        let window_id = parse_window_id(&device_id);
        let windows = Window::all().map_err(|e| format!("获取窗口列表失败: {}", e))?;

        let window = windows
            .into_iter()
            .find(|w| w.id().unwrap_or(0) == window_id)
            .ok_or_else(|| format!("窗口 ID {} 不存在或已关闭", window_id))?;

        let window_title = window.title().unwrap_or_else(|_| "Unknown".to_string());
        println!("📺 目标窗口: {} (ID: {})", window_title, window_id);

        let start_time = Instant::now();
        let mut frame_count: u64 = 0;
        let mut last_stats_update = Instant::now();
        let frame_interval = Duration::from_millis(1000 / fps.max(1) as u64);
        let mut last_frame_time = Instant::now();

        // 帧处理循环 (轮询模式)
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
                    println!("⚠️ 窗口截图失败: {} (窗口可能已关闭或最小化)", e);
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };

            let frame_width = image.width();
            let frame_height = image.height();

            // 获取 RGBA 数据
            let rgba_data = image.into_raw();

            // 每帧读取最新的区域配置（支持动态更新）
            let current_region = region.read().clone();

            let (final_data, out_width, out_height) = if let Some(r) = current_region {
                // 有区域选择，进行裁剪
                let start_x = r.x.max(0) as u32;
                let start_y = r.y.max(0) as u32;
                let start_x = start_x.min(frame_width.saturating_sub(1));
                let start_y = start_y.min(frame_height.saturating_sub(1));
                let end_x = (start_x + r.width).min(frame_width);
                let end_y = (start_y + r.height).min(frame_height);

                let rw = end_x - start_x;
                let rh = end_y - start_y;

                if rw < 10 || rh < 10 {
                    continue; // 区域太小，跳过
                }

                // 手动裁剪
                let mut cropped = Vec::with_capacity((rw * rh * 4) as usize);
                for y in start_y..end_y {
                    let row_start = (y * frame_width * 4 + start_x * 4) as usize;
                    let row_end = row_start + (rw * 4) as usize;
                    if row_end <= rgba_data.len() {
                        cropped.extend_from_slice(&rgba_data[row_start..row_end]);
                    }
                }
                (cropped, rw, rh)
            } else {
                // 全窗口
                (rgba_data, frame_width, frame_height)
            };

            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // 写入共享内存
            if let Some(ref mut shm) = *shared_buffer.write() {
                shm.write_frame(&final_data, out_width, out_height, timestamp);
            }

            frame_count += 1;

            if last_stats_update.elapsed() >= Duration::from_secs(1) {
                let elapsed = start_time.elapsed().as_secs_f32();
                let mut s = stats.write();
                s.fps = frame_count as f32 / elapsed;
                s.frame_count = frame_count;
                last_stats_update = Instant::now();
                println!("🪟 窗口帧数: {}, FPS: {:.1}", frame_count, s.fps);
            }
        }

        println!("🛑 窗口采集已停止");
        Ok(())
    }
}

// ==================== 统一采集状态 (共享内存版) ====================

/// 捕获状态管理 - 使用共享内存
pub struct CaptureState {
    running: Arc<AtomicBool>,
    generation: Arc<AtomicUsize>,
    shared_buffer: Arc<RwLock<Option<RgbaSharedBuffer>>>,
    capture_thread: Arc<RwLock<Option<thread::JoinHandle<()>>>>,
    stats: Arc<RwLock<CaptureStats>>,
    current_device: RwLock<Option<CaptureDevice>>,
    capture_region: Arc<RwLock<Option<CaptureRegion>>>, // 动态可更新的采集区域
}

impl Default for CaptureState {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicUsize::new(0)),
            shared_buffer: Arc::new(RwLock::new(None)),
            capture_thread: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(CaptureStats::default())),
            current_device: RwLock::new(None),
            capture_region: Arc::new(RwLock::new(None)),
        }
    }
}

impl CaptureState {
    /// 列出所有可用设备 (摄像头 + 屏幕 + 窗口)
    pub fn list_devices(&self) -> Vec<CaptureDevice> {
        let mut devices = Vec::new();
        devices.extend(camera::list_cameras());
        devices.extend(screen::list_screens());
        devices.extend(screen::list_windows());
        devices
    }

    /// 开始捕获 - 返回共享内存信息
    pub fn start(
        &self,
        device_id: &str,
        device_type: CaptureDeviceType,
        width: u32,
        height: u32,
        fps: u32,
        region: Option<CaptureRegion>,
    ) -> Result<RgbaSharedMemoryInfo, String> {
        // 停止现有捕获
        self.stop();

        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;

        // 等待旧线程结束
        {
            let mut handle = self.capture_thread.write();
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }

        // 确定实际缓冲区尺寸
        // 对于桌面采集，始终使用全屏尺寸以支持动态调整区域大小
        let (buf_width, buf_height) = match device_type {
            CaptureDeviceType::Screen => {
                // 屏幕采集使用设备报告的分辨率
                screen::parse_screen_resolution(device_id)
            }
            CaptureDeviceType::Window => {
                // 窗口采集: WGC 返回的尺寸包含窗口边框，比 xcap 大
                // 增加 32 像素边距以容纳 WGC 的额外边框
                let (w, h) = screen::parse_screen_resolution(device_id);
                (w + 32, h + 32)
            }
            CaptureDeviceType::Camera => {
                // USB 摄像头使用请求的尺寸
                (width.max(1920), height.max(1080))
            }
        };

        println!(
            "📐 创建缓冲区: {}x{} (请求: {}x{})",
            buf_width, buf_height, width, height
        );

        // 创建共享内存
        let shm_name = format!("rgba_capture_{}", std::process::id());
        let shared_buffer = RgbaSharedBuffer::create(&shm_name, buf_width, buf_height)?;
        let shm_info = shared_buffer.get_info();

        *self.shared_buffer.write() = Some(shared_buffer);

        // 设置初始采集区域
        *self.capture_region.write() = region.clone();

        // 重置统计
        *self.stats.write() = CaptureStats::default();

        self.running.store(true, Ordering::SeqCst);

        let running = self.running.clone();
        let generation = self.generation.clone();
        let shared_buffer = self.shared_buffer.clone();
        let stats = self.stats.clone();
        let device_id_owned = device_id.to_string();
        let capture_region = self.capture_region.clone(); // 使用共享的 capture_region

        let thread = match device_type {
            CaptureDeviceType::Camera => thread::spawn(move || {
                if generation.load(Ordering::SeqCst) != gen {
                    return;
                }
                if let Err(e) = camera::run_camera_capture(
                    device_id_owned,
                    width,
                    height,
                    fps,
                    running.clone(),
                    shared_buffer,
                    stats,
                ) {
                    eprintln!("❌ 摄像头采集错误: {}", e);
                }
                running.store(false, Ordering::SeqCst);
            }),
            CaptureDeviceType::Screen => thread::spawn(move || {
                if generation.load(Ordering::SeqCst) != gen {
                    return;
                }
                if let Err(e) = screen::run_screen_capture(
                    device_id_owned,
                    running.clone(),
                    shared_buffer,
                    stats,
                    capture_region, // 传递共享的 capture_region
                ) {
                    eprintln!("❌ 桌面采集错误: {}", e);
                }
                running.store(false, Ordering::SeqCst);
            }),
            CaptureDeviceType::Window => thread::spawn(move || {
                if generation.load(Ordering::SeqCst) != gen {
                    return;
                }
                if let Err(e) = screen::run_window_capture(
                    device_id_owned,
                    running.clone(),
                    shared_buffer,
                    stats,
                    capture_region,
                    fps,
                ) {
                    eprintln!("❌ 窗口采集错误: {}", e);
                }
                running.store(false, Ordering::SeqCst);
            }),
        };

        *self.capture_thread.write() = Some(thread);
        *self.current_device.write() = Some(CaptureDevice {
            id: device_id.to_string(),
            name: device_id.to_string(),
            device_type,
        });

        Ok(shm_info)
    }

    /// 停止捕获
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        // 等待线程结束
        if let Some(thread) = self.capture_thread.write().take() {
            let _ = thread.join();
        }

        // 清理共享内存
        *self.shared_buffer.write() = None;
        *self.current_device.write() = None;
    }

    /// 获取帧信息 (用于 JS 轮询)
    pub fn get_frame_info(&self) -> Option<RgbaFrameInfo> {
        let buffer = self.shared_buffer.read();
        let shm = buffer.as_ref()?;
        let header = shm.header();

        Some(RgbaFrameInfo {
            frame_id: header.write_frame_id.load(Ordering::Acquire),
            width: header.width,
            height: header.height,
            timestamp: header.timestamp.load(Ordering::Acquire),
        })
    }

    /// 读取当前帧 RGBA 数据 (回退方案)
    pub fn read_current_frame(&self) -> Option<(Vec<u8>, RgbaFrameInfo)> {
        let buffer = self.shared_buffer.read();
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

    /// 动态更新采集区域 (仅更新位置，保持宽高不变)
    pub fn update_region(&self, x: i32, y: i32) {
        let mut region = self.capture_region.write();
        if let Some(ref mut r) = *region {
            r.x = x;
            r.y = y;
            println!("📐 更新采集区域位置: ({}, {})", x, y);
        }
    }

    /// 动态更新采集区域 (同时更新位置和大小)
    pub fn update_region_full(&self, x: i32, y: i32, width: u32, height: u32) {
        let mut region = self.capture_region.write();
        if let Some(ref mut r) = *region {
            r.x = x;
            r.y = y;
            r.width = width;
            r.height = height;
            println!("📐 更新采集区域: ({}, {}) {}x{}", x, y, width, height);
        } else {
            // 如果还没有区域，创建一个
            *region = Some(CaptureRegion {
                x,
                y,
                width,
                height,
            });
            println!("📐 创建采集区域: ({}, {}) {}x{}", x, y, width, height);
        }
    }
}
