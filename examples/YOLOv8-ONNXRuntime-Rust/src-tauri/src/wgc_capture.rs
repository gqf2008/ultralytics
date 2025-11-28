//! Windows Graphics Capture (WGC) 窗口捕获模块
//!
//! 使用 Windows.Graphics.Capture API 捕获窗口
//! 优势：
//! - 被遮挡的窗口仍可捕获
//! - 浏览器检测到被 WGC 捕获时会继续渲染视频
//! - 性能优于 BitBlt/PrintWindow
//!
//! 参考: OBS Studio 的 winrt-capture.cpp 实现

#![cfg(windows)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use windows::{
    core::{IInspectable, Interface, Result as WinResult},
    Graphics::{
        Capture::{Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession},
        DirectX::DirectXPixelFormat,
    },
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_HARDWARE,
            Direct3D11::{
                D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
                D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ,
                D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
            },
            Dxgi::IDXGIDevice,
        },
        System::WinRT::Direct3D11::{
            CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
        },
    },
};

/// WGC 捕获的帧数据
pub struct WgcFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub timestamp: u64,
}

/// WGC 窗口捕获器 (同步轮询模式)
pub struct WgcCapture {
    d3d_device: ID3D11Device,
    d3d_context: ID3D11DeviceContext,
    _winrt_device: windows::Graphics::DirectX::Direct3D11::IDirect3DDevice,
    frame_pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    running: Arc<AtomicBool>,
    frame_id: Arc<AtomicU64>,
    _item: GraphicsCaptureItem, // 保持引用
}

impl WgcCapture {
    /// 从窗口句柄创建捕获器
    pub fn new(hwnd: isize) -> WinResult<Self> {
        let hwnd = HWND(hwnd as *mut _);

        // 创建 D3D11 设备
        let (d3d_device, d3d_context) = create_d3d11_device()?;

        // 创建 WinRT 设备
        let winrt_device = create_winrt_device(&d3d_device)?;

        // 创建 GraphicsCaptureItem
        let item = create_capture_item_for_window(hwnd)?;

        // 获取窗口大小
        let size = item.Size()?;
        println!(
            "🪟 WGC 捕获窗口: {}x{} (HWND: {:?})",
            size.Width, size.Height, hwnd
        );

        // 创建帧池 (BGRA 格式，使用 FreeThreaded 模式)
        // FreeThreaded 模式允许从任意线程访问
        // OBS 使用 2 帧缓冲
        let frame_pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &winrt_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2, // OBS 使用 2 帧缓冲
            size,
        )?;

        println!("✅ WGC 使用 FreeThreaded 帧池模式 (2帧缓冲)");

        // 创建捕获会话
        let session = frame_pool.CreateCaptureSession(&item)?;

        // 尝试禁用黄色边框 (需要 Windows 10 2004+)
        // 这会告诉浏览器"你正在被录制"，使其在遮挡时也继续渲染视频
        // 参考 OBS: libobs-winrt/winrt-capture.cpp
        // 注意: SetIsBorderRequired 在 IGraphicsCaptureSession3 上 (不是 Session2!)
        match session.SetIsBorderRequired(false) {
            Ok(_) => println!("✅ WGC 已禁用黄色边框 (浏览器将保持视频渲染)"),
            Err(e) => println!(
                "⚠️ 禁用黄色边框失败 (需要 Windows 10 2004+ 或管理员权限): {:?}",
                e
            ),
        }

        // 启用光标捕获 (Session2+)
        match session.SetIsCursorCaptureEnabled(true) {
            Ok(_) => println!("✅ WGC 光标捕获已启用"),
            Err(e) => println!("ℹ️ 光标捕获设置失败: {:?}", e),
        }

        Ok(Self {
            d3d_device,
            d3d_context,
            _winrt_device: winrt_device,
            frame_pool,
            session,
            running: Arc::new(AtomicBool::new(false)),
            frame_id: Arc::new(AtomicU64::new(0)),
            _item: item,
        })
    }

    /// 开始捕获
    pub fn start(&self) -> WinResult<()> {
        self.running.store(true, Ordering::SeqCst);
        self.session.StartCapture()?;
        println!("🚀 WGC 捕获已启动");
        Ok(())
    }

    /// 停止捕获
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.session.Close();
        let _ = self.frame_pool.Close();
        println!("🛑 WGC 捕获已停止");
    }

    /// 同步获取帧 (轮询模式)
    /// 直接从帧池获取下一帧，如果没有新帧则返回 None
    pub fn try_get_frame(&self) -> Option<WgcFrame> {
        if !self.running.load(Ordering::SeqCst) {
            return None;
        }

        // 尝试获取下一帧
        let frame = self.frame_pool.TryGetNextFrame().ok()?;
        let surface = frame.Surface().ok()?;

        // 复制到 CPU
        let frame_data = copy_frame_to_cpu(&self.d3d_device, &self.d3d_context, &surface).ok()?;
        self.frame_id.fetch_add(1, Ordering::Release);

        Some(frame_data)
    }

    /// 获取帧 ID (用于检测是否有新帧)
    pub fn get_frame_id(&self) -> u64 {
        self.frame_id.load(Ordering::Acquire)
    }

    /// 是否正在运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Drop for WgcCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

// ==================== 辅助函数 ====================

/// 创建 D3D11 设备
fn create_d3d11_device() -> WinResult<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device = None;
    let mut context = None;

    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            windows::Win32::Foundation::HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )?;
    }

    Ok((device.unwrap(), context.unwrap()))
}

/// 创建 WinRT D3D 设备
fn create_winrt_device(
    d3d_device: &ID3D11Device,
) -> WinResult<windows::Graphics::DirectX::Direct3D11::IDirect3DDevice> {
    let dxgi_device: IDXGIDevice = d3d_device.cast()?;

    let inspectable: IInspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi_device)? };

    inspectable.cast()
}

/// 为窗口创建 GraphicsCaptureItem
fn create_capture_item_for_window(hwnd: HWND) -> WinResult<GraphicsCaptureItem> {
    use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;

    let interop: IGraphicsCaptureItemInterop =
        windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;

    unsafe { interop.CreateForWindow(hwnd) }
}

/// 将 GPU 纹理复制到 CPU 内存
fn copy_frame_to_cpu(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    surface: &windows::Graphics::DirectX::Direct3D11::IDirect3DSurface,
) -> WinResult<WgcFrame> {
    // 获取 D3D11 纹理
    let access: IDirect3DDxgiInterfaceAccess = surface.cast()?;
    let texture: ID3D11Texture2D = unsafe { access.GetInterface()? };

    // 获取纹理描述
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&mut desc) };

    let width = desc.Width;
    let height = desc.Height;

    // 创建 staging 纹理 (CPU 可读)
    let staging_desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: desc.Format,
        SampleDesc: desc.SampleDesc,
        Usage: D3D11_USAGE_STAGING,
        BindFlags: Default::default(),
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: Default::default(),
    };

    let mut staging_texture: Option<ID3D11Texture2D> = None;
    unsafe {
        device.CreateTexture2D(&staging_desc, None, Some(&mut staging_texture))?;
    }
    let staging_texture = staging_texture.unwrap();

    // 复制到 staging
    unsafe {
        context.CopyResource(&staging_texture, &texture);
    }

    // 映射并读取数据
    let mut mapped = windows::Win32::Graphics::Direct3D11::D3D11_MAPPED_SUBRESOURCE::default();
    unsafe {
        context.Map(&staging_texture, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
    }

    let row_pitch = mapped.RowPitch as usize;
    let expected_pitch = (width * 4) as usize;

    // 复制数据 (处理行对齐)
    let mut data = vec![0u8; (width * height * 4) as usize];
    unsafe {
        let src = mapped.pData as *const u8;
        if row_pitch == expected_pitch {
            // 无需处理对齐
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        } else {
            // 处理行对齐
            for y in 0..height as usize {
                let src_row = src.add(y * row_pitch);
                let dst_row = data.as_mut_ptr().add(y * expected_pitch);
                std::ptr::copy_nonoverlapping(src_row, dst_row, expected_pitch);
            }
        }
    }

    // 取消映射
    unsafe {
        context.Unmap(&staging_texture, 0);
    }

    // BGRA → RGBA 转换
    for chunk in data.chunks_exact_mut(4) {
        chunk.swap(0, 2); // B ↔ R
    }

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    Ok(WgcFrame {
        data,
        width,
        height,
        timestamp,
    })
}

impl Clone for WgcFrame {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            timestamp: self.timestamp,
        }
    }
}

/// 检查 WGC 是否可用 (Windows 10 1903+)
pub fn is_wgc_supported() -> bool {
    // 检查 Windows.Graphics.Capture API 是否可用
    windows::Graphics::Capture::GraphicsCaptureSession::IsSupported().unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgc_supported() {
        println!("WGC 支持: {}", is_wgc_supported());
    }
}
