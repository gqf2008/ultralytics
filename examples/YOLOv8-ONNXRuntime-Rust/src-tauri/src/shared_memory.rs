//! 跨平台共享内存模块 - NV12 零拷贝渲染
//!
//! 实现 FFmpeg 解码 → 共享内存 → WebGL 的零拷贝管道
//! 支持 Windows (CreateFileMappingW) / macOS (shm_open) / Linux (memfd_create)
//!
//! ## 架构
//! ```text
//! ┌─────────────┐     ┌──────────────────┐    ┌───────────┐
//! │  FFmpeg     │     │   共享内存       │    │  WebGL    │
//! │  QSV/CUDA   │────▶│  (mmap/shmem)   │───▶│  纹理上传  │
//! │  硬解码     │写入  │  双缓冲环形     │读取 │  零拷贝   │
//! └─────────────┘     └──────────────────┘    └───────────┘
//! ```

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// 共享内存帧缓冲区头部 (64 字节对齐)
/// 这个结构体会被映射到共享内存的开头
#[repr(C, align(64))]
pub struct SharedFrameHeader {
    /// 魔数，用于验证共享内存有效性 (0xDEADBEEF)
    pub magic: u32,
    /// 版本号
    pub version: u32,
    /// 写入帧计数 (生产者递增)
    pub write_frame_id: AtomicU64,
    /// 读取帧计数 (消费者递增)
    pub read_frame_id: AtomicU64,
    /// 视频宽度
    pub width: u32,
    /// 视频高度
    pub height: u32,
    /// Y 平面步幅
    pub y_stride: u32,
    /// UV 平面步幅
    pub uv_stride: u32,
    /// 当前写入的缓冲区索引 (0 或 1，双缓冲)
    pub write_buffer_idx: AtomicU64,
    /// Y 数据偏移 (相对于 header 末尾)
    pub y_offset: u32,
    /// UV 数据偏移
    pub uv_offset: u32,
    /// 单帧数据大小 (Y + UV)
    pub frame_size: u32,
    /// 保留字段，对齐到 64 字节
    pub _reserved: [u8; 8],
}

impl SharedFrameHeader {
    pub const MAGIC: u32 = 0xDEADBEEF;
    pub const VERSION: u32 = 1;
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// 共享内存帧缓冲区
/// 布局: [Header (64B)] [Buffer0: Y + UV] [Buffer1: Y + UV]
pub struct SharedFrameBuffer {
    /// 共享内存名称
    pub name: String,
    /// 总大小
    pub total_size: usize,
    /// 原始指针
    ptr: *mut u8,
    /// 平台特定句柄
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    fd: std::os::unix::io::RawFd,
}

// 允许跨线程共享
unsafe impl Send for SharedFrameBuffer {}
unsafe impl Sync for SharedFrameBuffer {}

impl SharedFrameBuffer {
    /// 计算所需的共享内存大小
    /// 4K NV12: Y = 3840 * 2160 = 8,294,400 bytes
    ///          UV = 3840 * 1080 = 4,147,200 bytes
    ///          单帧 ≈ 12.4 MB，双缓冲 ≈ 25 MB + header
    pub fn calculate_size(width: u32, height: u32) -> usize {
        let y_size = (width * height) as usize;
        let uv_size = (width * height / 2) as usize;
        let frame_size = y_size + uv_size;
        // 双缓冲 + header，对齐到 4KB 页
        let total = SharedFrameHeader::SIZE + frame_size * 2;
        (total + 4095) & !4095
    }

    /// 获取共享内存名称前缀
    pub fn name_prefix() -> &'static str {
        "nv12_frame_"
    }

    /// 获取 header 指针
    pub fn header(&self) -> &SharedFrameHeader {
        unsafe { &*(self.ptr as *const SharedFrameHeader) }
    }

    /// 获取可变 header 指针
    pub fn header_mut(&mut self) -> &mut SharedFrameHeader {
        unsafe { &mut *(self.ptr as *mut SharedFrameHeader) }
    }

    /// 获取指定缓冲区的 Y 数据指针 (用于读取)
    pub fn y_buffer(&self, buffer_idx: usize) -> *const u8 {
        let header = self.header();
        let frame_offset = SharedFrameHeader::SIZE + buffer_idx * header.frame_size as usize;
        unsafe { self.ptr.add(frame_offset) as *const u8 }
    }

    /// 获取指定缓冲区的 UV 数据指针 (用于读取)
    pub fn uv_buffer(&self, buffer_idx: usize) -> *const u8 {
        let header = self.header();
        let frame_offset = SharedFrameHeader::SIZE + buffer_idx * header.frame_size as usize;
        let uv_offset = header.uv_offset as usize;
        unsafe { self.ptr.add(frame_offset + uv_offset) as *const u8 }
    }

    /// 写入一帧 NV12 数据
    pub fn write_frame(&mut self, y_data: &[u8], uv_data: &[u8]) {
        let header = self.header();
        let y_size = (header.y_stride * header.height) as usize;
        let uv_size = (header.uv_stride * header.height / 2) as usize;

        // 切换到下一个缓冲区
        let next_idx = 1 - header.write_buffer_idx.load(Ordering::Acquire);

        // 获取缓冲区指针
        let frame_offset = SharedFrameHeader::SIZE + next_idx as usize * header.frame_size as usize;
        unsafe {
            let y_ptr = self.ptr.add(frame_offset);
            let uv_ptr = self.ptr.add(frame_offset + header.uv_offset as usize);

            // 拷贝数据 (从 FFmpeg 到共享内存，这是唯一的一次拷贝)
            std::ptr::copy_nonoverlapping(y_data.as_ptr(), y_ptr, y_size.min(y_data.len()));
            std::ptr::copy_nonoverlapping(uv_data.as_ptr(), uv_ptr, uv_size.min(uv_data.len()));
        }

        // 更新写入索引和帧计数 (原子操作)
        let header_mut = unsafe { &mut *(self.ptr as *mut SharedFrameHeader) };
        header_mut
            .write_buffer_idx
            .store(next_idx, Ordering::Release);
        header_mut.write_frame_id.fetch_add(1, Ordering::Release);
    }

    /// 更新视频尺寸 (当检测到尺寸变化时)
    pub fn update_dimensions(&mut self, width: u32, height: u32, y_stride: u32, uv_stride: u32) {
        let header = self.header_mut();
        header.width = width;
        header.height = height;
        header.y_stride = y_stride;
        header.uv_stride = uv_stride;

        let y_size = y_stride * height;
        let uv_size = uv_stride * height / 2;
        header.y_offset = 0;
        header.uv_offset = y_size;
        header.frame_size = y_size + uv_size;
    }

    /// 获取共享内存的原始指针和大小 (用于 JS 端映射)
    pub fn get_memory_info(&self) -> SharedMemoryInfo {
        SharedMemoryInfo {
            name: self.name.clone(),
            total_size: self.total_size,
            header_size: SharedFrameHeader::SIZE,
            #[cfg(windows)]
            handle: self.handle as usize,
            #[cfg(unix)]
            fd: self.fd,
        }
    }
}

// ============ Windows 实现 ============
#[cfg(windows)]
impl SharedFrameBuffer {
    /// 创建共享内存 (生产者端调用)
    pub fn create(name: &str, width: u32, height: u32) -> Result<Self, String> {
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::Memory::*;

        let total_size = Self::calculate_size(width, height);

        // 将名称转换为宽字符
        let wide_name: Vec<u16> = format!("Local\\{}", name)
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();

        unsafe {
            // 创建文件映射
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                (total_size >> 32) as u32,
                total_size as u32,
                wide_name.as_ptr(),
            );

            if handle == 0 {
                return Err(format!("CreateFileMappingW failed: {}", GetLastError()));
            }

            // 映射视图 - windows-sys 0.52 返回 MEMORY_MAPPED_VIEW_ADDRESS
            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size);

            let ptr = view.Value as *mut u8;

            if ptr.is_null() {
                CloseHandle(handle);
                return Err(format!("MapViewOfFile failed: {}", GetLastError()));
            }

            // 初始化 header
            let header = ptr as *mut SharedFrameHeader;
            (*header).magic = SharedFrameHeader::MAGIC;
            (*header).version = SharedFrameHeader::VERSION;
            (*header).write_frame_id = AtomicU64::new(0);
            (*header).read_frame_id = AtomicU64::new(0);
            (*header).width = width;
            (*header).height = height;
            (*header).y_stride = width; // 假设无 padding
            (*header).uv_stride = width;
            (*header).write_buffer_idx = AtomicU64::new(0);

            let y_size = (width * height) as u32;
            let uv_size = (width * height / 2) as u32;
            let frame_size = y_size + uv_size;

            (*header).y_offset = 0;
            (*header).uv_offset = y_size;
            (*header).frame_size = frame_size;

            println!(
                "✅ 创建共享内存: {} ({} MB)",
                name,
                total_size / 1024 / 1024
            );

            Ok(Self {
                name: name.to_string(),
                total_size,
                ptr,
                handle,
            })
        }
    }

    /// 打开已存在的共享内存 (消费者端调用)
    #[allow(dead_code)]
    pub fn open(name: &str) -> Result<Self, String> {
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::Memory::*;

        let wide_name: Vec<u16> = format!("Local\\{}", name)
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();

        unsafe {
            let handle = OpenFileMappingW(
                FILE_MAP_ALL_ACCESS,
                0, // FALSE
                wide_name.as_ptr(),
            );

            if handle == 0 {
                return Err(format!("OpenFileMappingW failed: {}", GetLastError()));
            }

            // 先映射 header 获取大小
            let header_view =
                MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, SharedFrameHeader::SIZE);

            let header_ptr = header_view.Value as *mut SharedFrameHeader;

            if header_ptr.is_null() {
                CloseHandle(handle);
                return Err(format!("MapViewOfFile (header) failed: {}", GetLastError()));
            }

            if (*header_ptr).magic != SharedFrameHeader::MAGIC {
                UnmapViewOfFile(header_view);
                CloseHandle(handle);
                return Err("Invalid shared memory magic".to_string());
            }

            let width = (*header_ptr).width;
            let height = (*header_ptr).height;
            let total_size = Self::calculate_size(width, height);

            UnmapViewOfFile(header_view);

            // 重新映射完整大小
            let full_view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size);

            let ptr = full_view.Value as *mut u8;

            if ptr.is_null() {
                CloseHandle(handle);
                return Err(format!("MapViewOfFile (full) failed: {}", GetLastError()));
            }

            println!(
                "✅ 打开共享内存: {} ({} MB)",
                name,
                total_size / 1024 / 1024
            );

            Ok(Self {
                name: name.to_string(),
                total_size,
                ptr,
                handle,
            })
        }
    }

    /// 兼容旧接口
    #[allow(dead_code)]
    pub fn new(name: &str, width: u32, height: u32) -> Result<Self, String> {
        Self::create(name, width, height)
    }
}

#[cfg(windows)]
impl Drop for SharedFrameBuffer {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};

        unsafe {
            if !self.ptr.is_null() {
                let view = MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.ptr as *mut _,
                };
                UnmapViewOfFile(view);
            }
            if self.handle != 0 {
                CloseHandle(self.handle);
            }
        }
        println!("🗑️ 关闭共享内存: {}", self.name);
    }
}

// ============ macOS/Linux 实现 ============
#[cfg(unix)]
impl SharedFrameBuffer {
    pub fn create(name: &str, width: u32, height: u32) -> Result<Self, String> {
        use std::ffi::CString;

        let total_size = Self::calculate_size(width, height);
        let shm_name = CString::new(format!("/{}", name)).unwrap();

        unsafe {
            // 创建共享内存
            let fd = libc::shm_open(shm_name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o666);

            if fd < 0 {
                return Err(format!(
                    "shm_open failed: {}",
                    std::io::Error::last_os_error()
                ));
            }

            // 设置大小
            if libc::ftruncate(fd, total_size as libc::off_t) != 0 {
                libc::close(fd);
                libc::shm_unlink(shm_name.as_ptr());
                return Err(format!(
                    "ftruncate failed: {}",
                    std::io::Error::last_os_error()
                ));
            }

            // 映射内存
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            ) as *mut u8;

            if ptr == libc::MAP_FAILED as *mut u8 {
                libc::close(fd);
                libc::shm_unlink(shm_name.as_ptr());
                return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
            }

            // 初始化 header
            let header = ptr as *mut SharedFrameHeader;
            (*header).magic = SharedFrameHeader::MAGIC;
            (*header).version = SharedFrameHeader::VERSION;
            (*header).write_frame_id = AtomicU64::new(0);
            (*header).read_frame_id = AtomicU64::new(0);
            (*header).width = width;
            (*header).height = height;
            (*header).y_stride = width;
            (*header).uv_stride = width;
            (*header).write_buffer_idx = AtomicU64::new(0);

            let y_size = (width * height) as u32;
            let frame_size = y_size + y_size / 2;

            (*header).y_offset = 0;
            (*header).uv_offset = y_size;
            (*header).frame_size = frame_size;

            println!(
                "✅ 创建共享内存: {} ({} MB)",
                name,
                total_size / 1024 / 1024
            );

            Ok(Self {
                name: name.to_string(),
                total_size,
                ptr,
                fd,
            })
        }
    }

    #[allow(dead_code)]
    pub fn open(name: &str) -> Result<Self, String> {
        use std::ffi::CString;

        let shm_name = CString::new(format!("/{}", name)).unwrap();

        unsafe {
            let fd = libc::shm_open(shm_name.as_ptr(), libc::O_RDWR, 0o666);

            if fd < 0 {
                return Err(format!(
                    "shm_open failed: {}",
                    std::io::Error::last_os_error()
                ));
            }

            // 获取大小
            let mut stat: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut stat) != 0 {
                libc::close(fd);
                return Err(format!("fstat failed: {}", std::io::Error::last_os_error()));
            }

            let total_size = stat.st_size as usize;

            let ptr = libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            ) as *mut u8;

            if ptr == libc::MAP_FAILED as *mut u8 {
                libc::close(fd);
                return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
            }

            // 验证 magic
            let header = ptr as *const SharedFrameHeader;
            if (*header).magic != SharedFrameHeader::MAGIC {
                libc::munmap(ptr as *mut _, total_size);
                libc::close(fd);
                return Err("Invalid shared memory magic".to_string());
            }

            println!(
                "✅ 打开共享内存: {} ({} MB)",
                name,
                total_size / 1024 / 1024
            );

            Ok(Self {
                name: name.to_string(),
                total_size,
                ptr,
                fd,
            })
        }
    }

    #[allow(dead_code)]
    pub fn new(name: &str, width: u32, height: u32) -> Result<Self, String> {
        Self::create(name, width, height)
    }
}

#[cfg(unix)]
impl Drop for SharedFrameBuffer {
    fn drop(&mut self) {
        use std::ffi::CString;

        unsafe {
            if !self.ptr.is_null() {
                libc::munmap(self.ptr as *mut _, self.total_size);
            }
            if self.fd >= 0 {
                libc::close(self.fd);
                // 注意：只有创建者应该 unlink
                let shm_name = CString::new(format!("/{}", self.name)).unwrap();
                libc::shm_unlink(shm_name.as_ptr());
            }
        }
        println!("🗑️ 关闭共享内存: {}", self.name);
    }
}

/// 共享内存信息 (用于传递给 JS 端)
#[derive(Clone, serde::Serialize)]
pub struct SharedMemoryInfo {
    pub name: String,
    pub total_size: usize,
    pub header_size: usize,
    #[cfg(windows)]
    #[serde(rename = "handle")]
    pub handle: usize,
    #[cfg(unix)]
    #[serde(rename = "fd")]
    pub fd: i32,
}

/// 零拷贝渲染器状态
pub struct ZeroCopyRendererState {
    pub running: Arc<AtomicBool>,
    pub generation: Arc<AtomicUsize>,
    pub shared_buffer: Arc<RwLock<Option<SharedFrameBuffer>>>,
    pub thread_handle: Arc<RwLock<Option<std::thread::JoinHandle<()>>>>,
}

impl Default for ZeroCopyRendererState {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicUsize::new(0)),
            shared_buffer: Arc::new(RwLock::new(None)),
            thread_handle: Arc::new(RwLock::new(None)),
        }
    }
}

impl ZeroCopyRendererState {
    /// 启动零拷贝解码
    pub fn start(
        &self,
        url: String,
        hardware: crate::qsv_decoder::HardwareAccel,
        width: u32,
        height: u32,
    ) -> Result<SharedMemoryInfo, String> {
        use crate::qsv_decoder::{DecoderConfig, RtspTransport};

        // 停止之前的解码
        self.running.store(false, Ordering::SeqCst);
        let gen = self.generation.fetch_add(1, Ordering::SeqCst) + 1;

        // 等待旧线程结束
        {
            let mut handle = self.thread_handle.write();
            if let Some(h) = handle.take() {
                let _ = h.join();
            }
        }

        // 创建共享内存
        let shm_name = format!(
            "{}_{}",
            SharedFrameBuffer::name_prefix(),
            std::process::id()
        );
        let shared_buffer = SharedFrameBuffer::create(&shm_name, width, height)?;
        let shm_info = shared_buffer.get_memory_info();

        *self.shared_buffer.write() = Some(shared_buffer);

        let running = self.running.clone();
        let generation = self.generation.clone();
        let shared_buffer_arc = self.shared_buffer.clone();
        running.store(true, Ordering::SeqCst);

        let handle = std::thread::spawn(move || {
            println!("🚀 零拷贝解码线程启动: {} (gen={})", url, gen);

            let config = DecoderConfig {
                url: url.clone(),
                hardware_accel: hardware,
                rtsp_transport: RtspTransport::Tcp,
                buffer_size: 5 * 1024 * 1024,
                max_delay: 0,
            };

            let mut decoder = match crate::qsv_decoder::QsvDecoder::new(config) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("❌ 创建解码器失败: {}", e);
                    return;
                }
            };

            println!("✅ 零拷贝解码器就绪");

            let mut frame_count: u64 = 0;
            let mut last_fps_time = std::time::Instant::now();

            while running.load(Ordering::SeqCst) && generation.load(Ordering::SeqCst) == gen {
                match decoder.decode_next_frame() {
                    Ok(Some(frame)) => {
                        // 写入共享内存
                        {
                            let mut buffer_guard = shared_buffer_arc.write();
                            if let Some(ref mut buffer) = *buffer_guard {
                                // 检查尺寸是否变化
                                let header = buffer.header();
                                if header.width != frame.width || header.height != frame.height {
                                    buffer.update_dimensions(
                                        frame.width,
                                        frame.height,
                                        frame.y_stride as u32,
                                        frame.uv_stride as u32,
                                    );
                                }

                                // 写入帧数据
                                buffer.write_frame(&frame.y_data, &frame.uv_data);
                            }
                        }

                        frame_count += 1;

                        // FPS 统计
                        let elapsed = last_fps_time.elapsed();
                        if elapsed >= std::time::Duration::from_secs(1) {
                            let fps = frame_count as f64 / elapsed.as_secs_f64();
                            println!("📊 零拷贝 FPS: {:.1}", fps);
                            frame_count = 0;
                            last_fps_time = std::time::Instant::now();
                        }
                    }
                    Ok(None) => {
                        println!("🔚 RTSP 流结束");
                        break;
                    }
                    Err(e) => {
                        eprintln!("⚠️ 解码错误: {}", e);
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            }

            println!("🛑 零拷贝解码线程退出");
        });

        *self.thread_handle.write() = Some(handle);

        Ok(shm_info)
    }

    /// 停止解码
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);

        let mut handle = self.thread_handle.write();
        if let Some(h) = handle.take() {
            let _ = h.join();
        }

        // 关闭共享内存
        *self.shared_buffer.write() = None;
    }

    /// 获取当前帧信息 (供 JS 端轮询)
    pub fn get_frame_info(&self) -> Option<FrameInfo> {
        let buffer_guard = self.shared_buffer.read();
        buffer_guard.as_ref().map(|buffer| {
            let header = buffer.header();
            FrameInfo {
                frame_id: header.write_frame_id.load(Ordering::Acquire),
                buffer_idx: header.write_buffer_idx.load(Ordering::Acquire) as usize,
                width: header.width,
                height: header.height,
                y_stride: header.y_stride,
                uv_stride: header.uv_stride,
            }
        })
    }

    /// 读取当前帧的 NV12 数据 (IPC 回退方案)
    pub fn read_current_frame(&self) -> Option<(Vec<u8>, Vec<u8>, FrameInfo)> {
        let buffer_guard = self.shared_buffer.read();
        buffer_guard.as_ref().map(|buffer| {
            let header = buffer.header();
            let buffer_idx = header.write_buffer_idx.load(Ordering::Acquire) as usize;
            let y_size = (header.y_stride * header.height) as usize;
            let uv_size = (header.uv_stride * header.height / 2) as usize;

            let y_ptr = buffer.y_buffer(buffer_idx);
            let uv_ptr = buffer.uv_buffer(buffer_idx);

            let y_data = unsafe { std::slice::from_raw_parts(y_ptr, y_size).to_vec() };
            let uv_data = unsafe { std::slice::from_raw_parts(uv_ptr, uv_size).to_vec() };

            let info = FrameInfo {
                frame_id: header.write_frame_id.load(Ordering::Acquire),
                buffer_idx,
                width: header.width,
                height: header.height,
                y_stride: header.y_stride,
                uv_stride: header.uv_stride,
            };

            (y_data, uv_data, info)
        })
    }
}

/// 帧信息 (用于 JS 端)
#[derive(Clone, serde::Serialize)]
pub struct FrameInfo {
    pub frame_id: u64,
    pub buffer_idx: usize,
    pub width: u32,
    pub height: u32,
    pub y_stride: u32,
    pub uv_stride: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(SharedFrameHeader::SIZE, 64);
    }

    #[test]
    fn test_calculate_size() {
        // 4K
        let size = SharedFrameBuffer::calculate_size(3840, 2160);
        println!("4K 共享内存大小: {} MB", size / 1024 / 1024);
        assert!(size > 24 * 1024 * 1024); // > 24 MB
    }
}

// ==================== RGBA 共享内存 (用于摄像头/桌面采集) ====================

/// RGBA 共享内存帧头部 (64 字节对齐)
#[repr(C, align(64))]
pub struct RgbaFrameHeader {
    /// 魔数 (0xRGBA0001)
    pub magic: u32,
    /// 版本号
    pub version: u32,
    /// 写入帧计数
    pub write_frame_id: AtomicU64,
    /// 读取帧计数
    pub read_frame_id: AtomicU64,
    /// 当前帧宽度
    pub width: u32,
    /// 当前帧高度
    pub height: u32,
    /// 最大缓冲区宽度 (用于计算双缓冲偏移)
    pub max_width: u32,
    /// 最大缓冲区高度
    pub max_height: u32,
    /// 当前写入缓冲区索引 (0 或 1)
    pub write_buffer_idx: AtomicU64,
    /// 时间戳 (毫秒)
    pub timestamp: AtomicU64,
}

impl RgbaFrameHeader {
    pub const MAGIC: u32 = 0x52474241; // "RGBA"
    pub const VERSION: u32 = 1;
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// RGBA 共享内存缓冲区
pub struct RgbaSharedBuffer {
    pub name: String,
    pub total_size: usize,
    ptr: *mut u8,
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(unix)]
    fd: std::os::unix::io::RawFd,
}

unsafe impl Send for RgbaSharedBuffer {}
unsafe impl Sync for RgbaSharedBuffer {}

impl RgbaSharedBuffer {
    /// 计算所需大小 (双缓冲 RGBA)
    pub fn calculate_size(width: u32, height: u32) -> usize {
        let frame_size = (width * height * 4) as usize; // RGBA
        let total = RgbaFrameHeader::SIZE + frame_size * 2;
        (total + 4095) & !4095 // 4KB 对齐
    }

    /// 获取 header
    pub fn header(&self) -> &RgbaFrameHeader {
        unsafe { &*(self.ptr as *const RgbaFrameHeader) }
    }

    /// 获取当前读取缓冲区 (读取上一个完整写入的缓冲区)
    pub fn current_read_buffer(&self) -> &[u8] {
        let header = self.header();
        // 注意：write_buffer_idx 在写入完成后才递增，所以当前 write_idx 指向的是"下一个"要写入的缓冲区
        // 我们读取的是 (write_idx - 1) % 2，即上一个完整写入的缓冲区
        let write_idx = header.write_buffer_idx.load(Ordering::Acquire) as usize;
        let read_idx = (write_idx + 1) % 2; // 读取上一个写入的缓冲区
                                            // 使用 max_width/max_height 计算缓冲区偏移 (固定大小)
        let max_frame_size = (header.max_width * header.max_height * 4) as usize;
        let current_frame_size = (header.width * header.height * 4) as usize;
        let offset = RgbaFrameHeader::SIZE + read_idx * max_frame_size;
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), current_frame_size) }
    }

    /// 写入帧数据 - 带安全检查
    pub fn write_frame(&mut self, data: &[u8], width: u32, height: u32, timestamp: u64) {
        // 安全检查：确保帧尺寸不超过缓冲区
        let header = self.header();
        let buf_max_width = header.max_width;
        let buf_max_height = header.max_height;

        // 如果帧尺寸超过缓冲区，只打印警告并跳过
        if width > buf_max_width || height > buf_max_height {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "⚠️ 帧尺寸 {}x{} 超过缓冲区 {}x{}, 跳过写入",
                    width, height, buf_max_width, buf_max_height
                );
            }
            return;
        }

        // 直接使用原始指针避免多次可变借用
        unsafe {
            let header = self.ptr as *mut RgbaFrameHeader;
            (*header).timestamp.store(timestamp, Ordering::Release);

            // 计算写入缓冲区 - 使用 max 尺寸计算偏移
            let write_idx = (*header).write_buffer_idx.load(Ordering::Acquire) as usize % 2;
            let max_frame_size = (buf_max_width * buf_max_height * 4) as usize;
            let frame_size = (width * height * 4) as usize;
            let offset = RgbaFrameHeader::SIZE + write_idx * max_frame_size;

            // 确保不越界
            if offset + frame_size > self.total_size {
                eprintln!(
                    "⚠️ 帧数据越界: offset={}, frame_size={}, total={}",
                    offset, frame_size, self.total_size
                );
                return;
            }

            let buffer = std::slice::from_raw_parts_mut(self.ptr.add(offset), frame_size);

            let copy_len = data.len().min(buffer.len());
            buffer[..copy_len].copy_from_slice(&data[..copy_len]);

            // 更新实际帧尺寸 (在写入完成后更新)
            (*header).width = width;
            (*header).height = height;

            // 切换缓冲区并更新帧号
            (*header).write_buffer_idx.fetch_add(1, Ordering::Release);
            (*header).write_frame_id.fetch_add(1, Ordering::Release);

            // 仅第一帧打印
            static FIRST_FRAME: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(true);
            if FIRST_FRAME.swap(false, Ordering::Relaxed) {
                println!("📝 首帧写入: {}x{}, data_len={}", width, height, copy_len);
            }
        }
    }

    /// 获取共享内存信息
    pub fn get_info(&self) -> RgbaSharedMemoryInfo {
        RgbaSharedMemoryInfo {
            name: self.name.clone(),
            total_size: self.total_size,
            header_size: RgbaFrameHeader::SIZE,
        }
    }

    #[cfg(windows)]
    pub fn create(name: &str, width: u32, height: u32) -> Result<Self, String> {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::Memory::*;

        let total_size = Self::calculate_size(width, height);
        let wide_name: Vec<u16> = OsStr::new(&format!("Local\\{}", name))
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        unsafe {
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE as _,
                std::ptr::null_mut(),
                PAGE_READWRITE,
                (total_size >> 32) as u32,
                total_size as u32,
                wide_name.as_ptr(),
            );

            if handle == 0 {
                return Err(format!("CreateFileMappingW 失败: {}", GetLastError()));
            }

            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
            let ptr = view.Value as *mut u8;

            if ptr.is_null() {
                CloseHandle(handle);
                return Err(format!("MapViewOfFile 失败: {}", GetLastError()));
            }

            // 初始化 header
            let header = ptr as *mut RgbaFrameHeader;
            (*header).magic = RgbaFrameHeader::MAGIC;
            (*header).version = RgbaFrameHeader::VERSION;
            (*header).width = width;
            (*header).height = height;
            (*header).max_width = width;
            (*header).max_height = height;
            (*header).write_frame_id = AtomicU64::new(0);
            (*header).read_frame_id = AtomicU64::new(0);
            (*header).write_buffer_idx = AtomicU64::new(0);
            (*header).timestamp = AtomicU64::new(0);

            println!(
                "✅ 创建 RGBA 共享内存: {} ({} MB)",
                name,
                total_size / 1024 / 1024
            );

            Ok(Self {
                name: name.to_string(),
                total_size,
                ptr,
                handle,
            })
        }
    }

    #[cfg(windows)]
    #[allow(dead_code)]
    pub fn open(name: &str) -> Result<Self, String> {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::*;
        use windows_sys::Win32::System::Memory::*;

        let wide_name: Vec<u16> = OsStr::new(&format!("Local\\{}", name))
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        unsafe {
            let handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide_name.as_ptr());

            if handle == 0 {
                return Err(format!("OpenFileMappingW 失败: {}", GetLastError()));
            }

            // 先映射 header 获取大小
            let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, RgbaFrameHeader::SIZE);
            let header_ptr = view.Value as *mut u8;
            if header_ptr.is_null() {
                CloseHandle(handle);
                return Err("MapViewOfFile 失败".to_string());
            }

            let header = header_ptr as *const RgbaFrameHeader;
            if (*header).magic != RgbaFrameHeader::MAGIC {
                UnmapViewOfFile(view);
                CloseHandle(handle);
                return Err("无效的共享内存".to_string());
            }

            let width = (*header).width;
            let height = (*header).height;
            UnmapViewOfFile(view);

            // 重新映射完整大小
            let total_size = Self::calculate_size(width, height);
            let view2 = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
            let ptr = view2.Value as *mut u8;

            if ptr.is_null() {
                CloseHandle(handle);
                return Err("MapViewOfFile 失败".to_string());
            }

            Ok(Self {
                name: name.to_string(),
                total_size,
                ptr,
                handle,
            })
        }
    }
}

#[cfg(windows)]
impl Drop for RgbaSharedBuffer {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{UnmapViewOfFile, MEMORY_MAPPED_VIEW_ADDRESS};

        unsafe {
            if !self.ptr.is_null() {
                let view = MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.ptr as _,
                };
                UnmapViewOfFile(view);
            }
            if self.handle != 0 {
                CloseHandle(self.handle);
            }
        }
        println!("🗑️ 关闭 RGBA 共享内存: {}", self.name);
    }
}

/// RGBA 共享内存信息
#[derive(Clone, serde::Serialize)]
pub struct RgbaSharedMemoryInfo {
    pub name: String,
    pub total_size: usize,
    pub header_size: usize,
}

/// RGBA 帧信息 (用于 JS 轮询)
#[derive(Clone, serde::Serialize)]
pub struct RgbaFrameInfo {
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}
