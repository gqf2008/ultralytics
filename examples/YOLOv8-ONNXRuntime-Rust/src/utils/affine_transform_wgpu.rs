/// GPU加速的仿射变换 (使用wgpu)
/// 通过GPU并行处理实现10-100倍性能提升

use super::affine_transform::{AffineMatrix, BorderMode, InterpolationMethod};
use wgpu::util::DeviceExt;

/// GPU加速的仿射变换上下文
/// 复用GPU资源,避免重复初始化
pub struct WgpuAffineTransform {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_bilinear: wgpu::ComputePipeline,
    pipeline_nearest: wgpu::ComputePipeline,
}

impl WgpuAffineTransform {
    /// 创建GPU加速上下文
    /// 
    /// 这个过程会:
    /// 1. 选择GPU设备
    /// 2. 编译compute shader
    /// 3. 创建计算管线
    /// 
    /// 注意: 使用pollster::block_on内部处理异步,外部是同步调用
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // 创建wgpu实例
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // 请求适配器 (GPU) - 使用pollster阻塞等待
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("无法找到合适的GPU")?;

        // 获取设备和队列 - 使用pollster阻塞等待
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Affine Transform Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))?;

        // 编译计算着色器
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Affine Transform Shader"),
            source: wgpu::ShaderSource::Wgsl(AFFINE_SHADER.into()),
        });

        // 创建双线性插值管线
        let pipeline_bilinear = create_pipeline(&device, &shader_module, "warp_affine_bilinear");

        // 创建最近邻插值管线
        let pipeline_nearest = create_pipeline(&device, &shader_module, "warp_affine_nearest");

        Ok(Self {
            device,
            queue,
            pipeline_bilinear,
            pipeline_nearest,
        })
    }

    /// 执行仿射变换
    pub fn warp_affine_rgb(
        &self,
        src: &[u8],
        src_width: u32,
        src_height: u32,
        matrix: &AffineMatrix,
        dst_size: (u32, u32),
        interpolation: InterpolationMethod,
        border_mode: BorderMode,
    ) -> Vec<u8> {
        let (dst_width, dst_height) = dst_size;
        let dst_size_bytes = (dst_width * dst_height * 3) as usize;

        // 使用逆矩阵进行反向映射
        let inv_matrix = matrix.inverse().expect("矩阵不可逆");

        // 创建GPU缓冲区
        let src_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Source Buffer"),
            contents: src,
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dst_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination Buffer"),
            size: dst_size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 创建参数缓冲区
        let border_value = match border_mode {
            BorderMode::Constant(val) => val,
            _ => 0,
        };

        let params = AffineParams {
            src_width,
            src_height,
            dst_width,
            dst_height,
            a11: inv_matrix.a11,
            a12: inv_matrix.a12,
            b1: inv_matrix.b1,
            a21: inv_matrix.a21,
            a22: inv_matrix.a22,
            b2: inv_matrix.b2,
            border_value: border_value as f32,
            _padding: 0.0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 创建绑定组
        let pipeline = match interpolation {
            InterpolationMethod::Bilinear => &self.pipeline_bilinear,
            InterpolationMethod::Nearest => &self.pipeline_nearest,
        };

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Affine Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
            ],
        });

        // 创建命令编码器
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Affine Encoder"),
            });

        // 执行计算着色器
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Affine Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // 分组大小: 8x8 像素
            let workgroup_size_x = 8;
            let workgroup_size_y = 8;
            let num_workgroups_x = (dst_width + workgroup_size_x - 1) / workgroup_size_x;
            let num_workgroups_y = (dst_height + workgroup_size_y - 1) / workgroup_size_y;

            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // 创建读取缓冲区
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: dst_size_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 复制结果
        encoder.copy_buffer_to_buffer(
            &dst_buffer,
            0,
            &output_buffer,
            0,
            dst_size_bytes as u64,
        );

        // 提交命令
        self.queue.submit(Some(encoder.finish()));

        // 读取结果
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // 等待GPU完成
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx).unwrap().unwrap();

        // 复制数据到CPU
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();

        drop(data);
        output_buffer.unmap();

        result
    }
}

/// 辅助函数: 创建计算管线
fn create_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Affine Bind Group Layout"),
        entries: &[
            // 参数缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 源图像缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 目标图像缓冲区
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Affine Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Affine Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: shader_module,
        entry_point,
        cache: None,
        compilation_options: Default::default(),
    })
}

/// 参数结构 (需要16字节对齐)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AffineParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    a11: f32,
    a12: f32,
    b1: f32,
    a21: f32,
    a22: f32,
    b2: f32,
    border_value: f32,
    _padding: f32, // 确保16字节对齐
}

/// WGSL计算着色器
const AFFINE_SHADER: &str = r#"
struct AffineParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    a11: f32,
    a12: f32,
    b1: f32,
    a21: f32,
    a22: f32,
    b2: f32,
    border_value: f32,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> params: AffineParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// 从打包的u32数组中读取RGB像素
fn get_pixel(idx: u32) -> vec3<f32> {
    let base_idx = idx * 3u;
    var rgb: vec3<f32>;
    
    for (var c = 0u; c < 3u; c = c + 1u) {
        let pixel_byte_idx = base_idx + c;
        let word = src[pixel_byte_idx / 4u];
        let shift = (pixel_byte_idx % 4u) * 8u;
        rgb[c] = f32((word >> shift) & 0xFFu);
    }
    
    return rgb;
}

// 将RGB像素写入打包的u32数组
fn set_pixel(idx: u32, rgb: vec3<f32>) {
    let base_idx = idx * 3u;
    
    for (var c = 0u; c < 3u; c = c + 1u) {
        let pixel_byte_idx = base_idx + c;
        let word_idx = pixel_byte_idx / 4u;
        let byte_offset = pixel_byte_idx % 4u;
        let shift = byte_offset * 8u;
        
        let val = u32(clamp(rgb[c], 0.0, 255.0));
        let mask = 0xFFu << shift;
        let old_word = dst[word_idx];
        dst[word_idx] = (old_word & ~mask) | (val << shift);
    }
}

// 双线性插值
@compute @workgroup_size(8, 8)
fn warp_affine_bilinear(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;
    
    if (dst_x >= params.dst_width || dst_y >= params.dst_height) {
        return;
    }
    
    // 计算源坐标
    let dst_x_f = f32(dst_x);
    let dst_y_f = f32(dst_y);
    
    let src_x = params.a11 * dst_x_f + params.a12 * dst_y_f + params.b1;
    let src_y = params.a21 * dst_x_f + params.a22 * dst_y_f + params.b2;
    
    let dst_idx = dst_y * params.dst_width + dst_x;
    
    // 边界检查
    if (src_x < 0.0 || src_x >= f32(params.src_width - 1u) ||
        src_y < 0.0 || src_y >= f32(params.src_height - 1u)) {
        let border_rgb = vec3<f32>(params.border_value);
        set_pixel(dst_idx, border_rgb);
        return;
    }
    
    // 双线性插值
    let x0 = u32(src_x);
    let y0 = u32(src_y);
    let fx = src_x - f32(x0);
    let fy = src_y - f32(y0);
    
    let idx00 = y0 * params.src_width + x0;
    let idx01 = (y0 + 1u) * params.src_width + x0;
    let idx10 = y0 * params.src_width + x0 + 1u;
    let idx11 = (y0 + 1u) * params.src_width + x0 + 1u;
    
    let p00 = get_pixel(idx00);
    let p01 = get_pixel(idx01);
    let p10 = get_pixel(idx10);
    let p11 = get_pixel(idx11);
    
    let v0 = p00 + (p10 - p00) * fx;
    let v1 = p01 + (p11 - p01) * fx;
    let result = v0 + (v1 - v0) * fy;
    
    set_pixel(dst_idx, result);
}

// 最近邻插值
@compute @workgroup_size(8, 8)
fn warp_affine_nearest(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;
    
    if (dst_x >= params.dst_width || dst_y >= params.dst_height) {
        return;
    }
    
    let dst_x_f = f32(dst_x);
    let dst_y_f = f32(dst_y);
    
    let src_x = params.a11 * dst_x_f + params.a12 * dst_y_f + params.b1;
    let src_y = params.a21 * dst_x_f + params.a22 * dst_y_f + params.b2;
    
    let ix = i32(round(src_x));
    let iy = i32(round(src_y));
    
    let dst_idx = dst_y * params.dst_width + dst_x;
    
    if (ix < 0 || ix >= i32(params.src_width) || iy < 0 || iy >= i32(params.src_height)) {
        let border_rgb = vec3<f32>(params.border_value);
        set_pixel(dst_idx, border_rgb);
        return;
    }
    
    let src_idx = u32(iy) * params.src_width + u32(ix);
    let rgb = get_pixel(src_idx);
    set_pixel(dst_idx, rgb);
}
"#;
