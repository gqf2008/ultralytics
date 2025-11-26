//! 共享内存帧缓冲 - 零拷贝数据传输
//!
//! 使用 Windows 命名内存映射实现前端和后端之间的零拷贝传输
//! 前端直接写入共享内存，后端直接读取，无需 IPC 序列化

use parking_lot::RwLock;
use std::sync::Arc;

#[cfg(windows)]
mod windows_impl {
    use std::ffi::CString;
    use std::ptr;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingA, MapViewOfFile, UnmapViewOfFile,
        FILE_MAP_ALL_ACCESS, PAGE_READWRITE,
    };

    pub struct SharedFrameBuffer {
        handle: HANDLE,
        ptr: *mut u8,
        size: usize,
        width: u32,
        height: u32,
    }

    // 手动实现 Send + Sync (指针在单线程写入，多线程读取是安全的)
    unsafe impl Send for SharedFrameBuffer {}
    unsafe impl Sync for SharedFrameBuffer {}

    impl SharedFrameBuffer {
        /// 创建共享内存区域
        pub fn new(name: &str, width: u32, height: u32) -> Result<Self, String> {
            let size = (width * height * 4) as usize; // RGBA
            let c_name = CString::new(name).map_err(|e| e.to_string())?;

            unsafe {
                // 创建或打开命名内存映射
                let handle = CreateFileMappingA(
                    INVALID_HANDLE_VALUE,
                    ptr::null(),
                    PAGE_READWRITE,
                    0,
                    size as u32,
                    c_name.as_ptr() as *const u8,
                );

                if handle == 0 {
                    return Err("CreateFileMapping 失败".to_string());
                }

                // 映射视图
                let ptr = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
                if ptr.is_null() {
                    CloseHandle(handle);
                    return Err("MapViewOfFile 失败".to_string());
                }

                Ok(Self {
                    handle,
                    ptr: ptr as *mut u8,
                    size,
                    width,
                    height,
                })
            }
        }

        /// 获取共享内存名称前缀
        pub fn name_prefix() -> &'static str {
            "Local\\YOLOv8Sentinel_Frame_"
        }

        /// 读取帧数据 (零拷贝引用)
        pub fn read(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
        }

        /// 写入帧数据
        pub fn write(&self, data: &[u8]) -> Result<(), String> {
            if data.len() != self.size {
                return Err(format!(
                    "数据大小不匹配: 期望 {}, 实际 {}",
                    self.size,
                    data.len()
                ));
            }
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, self.size);
            }
            Ok(())
        }

        pub fn width(&self) -> u32 {
            self.width
        }

        pub fn height(&self) -> u32 {
            self.height
        }

        pub fn size(&self) -> usize {
            self.size
        }
    }

    impl Drop for SharedFrameBuffer {
        fn drop(&mut self) {
            unsafe {
                if !self.ptr.is_null() {
                    UnmapViewOfFile(self.ptr as *const _);
                }
                if self.handle != 0 {
                    CloseHandle(self.handle);
                }
            }
        }
    }
}

#[cfg(windows)]
pub use windows_impl::SharedFrameBuffer;

/// 共享内存帧缓冲管理器
pub struct SharedFrameManager {
    buffer: Option<SharedFrameBuffer>,
    frame_ready: Arc<RwLock<bool>>,
}

impl Default for SharedFrameManager {
    fn default() -> Self {
        Self {
            buffer: None,
            frame_ready: Arc::new(RwLock::new(false)),
        }
    }
}

impl SharedFrameManager {
    /// 初始化共享内存
    pub fn init(&mut self, width: u32, height: u32) -> Result<String, String> {
        let name = format!("{}{}x{}", SharedFrameBuffer::name_prefix(), width, height);
        let buffer = SharedFrameBuffer::new(&name, width, height)?;
        
        println!("✅ 共享内存已创建: {} ({}x{}, {} bytes)", 
            name, width, height, buffer.size());
        
        self.buffer = Some(buffer);
        Ok(name)
    }

    /// 读取帧 (零拷贝)
    pub fn read_frame(&self) -> Option<&[u8]> {
        self.buffer.as_ref().map(|b| b.read())
    }

    /// 标记帧已准备好
    pub fn set_frame_ready(&self, ready: bool) {
        *self.frame_ready.write() = ready;
    }

    /// 检查帧是否准备好
    pub fn is_frame_ready(&self) -> bool {
        *self.frame_ready.read()
    }

    /// 获取缓冲区尺寸
    pub fn get_size(&self) -> Option<(u32, u32)> {
        self.buffer.as_ref().map(|b| (b.width(), b.height()))
    }
}
