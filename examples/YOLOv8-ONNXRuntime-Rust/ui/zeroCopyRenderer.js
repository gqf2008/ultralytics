/**
 * 零拷贝 NV12 视频渲染器
 * 
 * 使用高效 IPC + WebGL 实现接近零拷贝的 4K 视频渲染
 * 
 * 架构:
 * ┌─────────────┐     ┌──────────────────┐    ┌───────────┐
 * │  FFmpeg     │     │   共享内存       │    │  WebGL    │
 * │  QSV/CUDA   │────▶│  (Rust 端)      │───▶│  纹理上传  │
 * │  硬解码     │写入  │  双缓冲环形     │IPC │  NV12→RGB │
 * └─────────────┘     └──────────────────┘    └───────────┘
 * 
 * 数据流 (4K@30fps):
 * 1. Rust: FFmpeg 硬解码 → 写入共享内存 (~12.5MB/帧)
 * 2. JS: 轮询帧 ID (轻量级，仅 24 bytes)
 * 3. 有新帧时: 通过 Raw IPC 读取 NV12 数据
 * 4. WebGL: 上传 Y/UV 纹理，GPU 做 YUV→RGB
 */

import { invoke } from '@tauri-apps/api/core';

export class ZeroCopyRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        
        // 状态
        this.isRunning = false;
        this.lastFrameId = 0;
        this.pollInterval = null;
        
        // 性能统计
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 0;
        this.latency = 0;
        
        // WebGL 资源
        this.program = null;
        this.yTexture = null;
        this.uvTexture = null;
        
        // 视频尺寸
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.yStride = 0;
        this.uvStride = 0;
        
        this._initWebGL();
        this._resizeCanvas();
        
        window.addEventListener('resize', () => this._resizeCanvas());
    }
    
    _resizeCanvas() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        
        this.canvas.width = w;
        this.canvas.height = h;
        this.gl.viewport(0, 0, w, h);
    }
    
    _initWebGL() {
        const gl = this.gl;
        
        // NV12 YUV→RGB 转换 shader (BT.601)
        const vertexShaderSrc = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            varying vec2 v_texCoord;
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;
        
        // 高性能 NV12→RGB shader
        const fragmentShaderSrc = `
            precision highp float;
            
            uniform sampler2D u_yTexture;
            uniform sampler2D u_uvTexture;
            uniform vec2 u_videoSize;
            uniform vec2 u_yStride;
            
            varying vec2 v_texCoord;
            
            void main() {
                // 计算实际纹理坐标 (考虑 stride padding)
                float xRatio = u_videoSize.x / u_yStride.x;
                vec2 coord = vec2(v_texCoord.x * xRatio, v_texCoord.y);
                
                // 采样 Y (单通道)
                float y = texture2D(u_yTexture, coord).r;
                
                // 采样 UV (RG 通道)
                vec2 uv = texture2D(u_uvTexture, coord).rg;
                float u = uv.r - 0.5;
                float v = uv.g - 0.5;
                
                // BT.601 YUV→RGB
                float r = y + 1.402 * v;
                float g = y - 0.344136 * u - 0.714136 * v;
                float b = y + 1.772 * u;
                
                gl_FragColor = vec4(r, g, b, 1.0);
            }
        `;
        
        // 编译 shader
        const vertexShader = this._compileShader(gl.VERTEX_SHADER, vertexShaderSrc);
        const fragmentShader = this._compileShader(gl.FRAGMENT_SHADER, fragmentShaderSrc);
        
        // 创建程序
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertexShader);
        gl.attachShader(this.program, fragmentShader);
        gl.linkProgram(this.program);
        
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            throw new Error('Shader program link failed: ' + gl.getProgramInfoLog(this.program));
        }
        
        gl.useProgram(this.program);
        
        // 设置顶点数据 (全屏四边形)
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1,
        ]);
        
        const texCoords = new Float32Array([
            0, 1,  // 翻转 Y 轴
            1, 1,
            0, 0,
            1, 0,
        ]);
        
        // 位置缓冲
        const posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        
        const posLoc = gl.getAttribLocation(this.program, 'a_position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        
        // 纹理坐标缓冲
        const texBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, texBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
        
        const texLoc = gl.getAttribLocation(this.program, 'a_texCoord');
        gl.enableVertexAttribArray(texLoc);
        gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);
        
        // 获取 uniform 位置
        this.uYTexture = gl.getUniformLocation(this.program, 'u_yTexture');
        this.uUvTexture = gl.getUniformLocation(this.program, 'u_uvTexture');
        this.uVideoSize = gl.getUniformLocation(this.program, 'u_videoSize');
        this.uYStride = gl.getUniformLocation(this.program, 'u_yStride');
        
        // 创建纹理
        this.yTexture = this._createTexture();
        this.uvTexture = this._createTexture();
        
        console.log('✅ 零拷贝 WebGL 渲染器初始化完成');
    }
    
    _compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(shader));
        }
        
        return shader;
    }
    
    _createTexture() {
        const gl = this.gl;
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        return texture;
    }
    
    /**
     * 启动零拷贝流
     */
    async start(url, hardware = 'qsv', width = 3840, height = 2160) {
        if (this.isRunning) {
            console.warn('⚠️ 渲染器已在运行');
            return;
        }
        
        try {
            // 启动 Rust 端解码
            const shmInfo = await invoke('start_zerocopy_stream', {
                url,
                hardware,
                width,
                height,
            });
            
            console.log('📦 共享内存信息:', shmInfo);
            
            this.isRunning = true;
            this.lastFrameId = 0;
            
            // 启动高帧率轮询
            this._startPolling();
            
            console.log(`✅ 零拷贝流已启动: ${url} @ ${width}x${height}`);
        } catch (e) {
            console.error('❌ 启动零拷贝流失败:', e);
            throw e;
        }
    }
    
    /**
     * 停止零拷贝流
     */
    async stop() {
        this.isRunning = false;
        
        if (this.pollInterval) {
            cancelAnimationFrame(this.pollInterval);
            this.pollInterval = null;
        }
        
        try {
            await invoke('stop_zerocopy_stream');
            console.log('🛑 零拷贝流已停止');
        } catch (e) {
            console.error('⚠️ 停止零拷贝流失败:', e);
        }
    }
    
    /**
     * 高帧率轮询
     */
    _startPolling() {
        const poll = async () => {
            if (!this.isRunning) return;
            
            try {
                // 1. 轮询帧信息 (轻量级)
                const frameInfo = await invoke('get_zerocopy_frame_info');
                
                if (frameInfo && frameInfo.frame_id > this.lastFrameId) {
                    const startTime = performance.now();
                    
                    // 2. 有新帧，读取 NV12 数据
                    const rawData = await invoke('read_zerocopy_frame');
                    
                    // 3. 解析数据
                    const data = new Uint8Array(rawData);
                    const view = new DataView(data.buffer);
                    
                    // Header: frame_id(8) width(4) height(4) y_stride(4) uv_stride(4) = 24 bytes
                    const frameId = Number(view.getBigUint64(0, true));
                    const width = view.getUint32(8, true);
                    const height = view.getUint32(12, true);
                    const yStride = view.getUint32(16, true);
                    const uvStride = view.getUint32(20, true);
                    
                    // 更新尺寸
                    if (this.videoWidth !== width || this.videoHeight !== height) {
                        console.log(`📺 视频尺寸: ${width}x${height} (Y stride: ${yStride})`);
                        this.videoWidth = width;
                        this.videoHeight = height;
                        this.yStride = yStride;
                        this.uvStride = uvStride;
                    }
                    
                    // 计算数据偏移
                    const headerEnd = 24;
                    const ySize = yStride * height;
                    const uvSize = uvStride * (height / 2);
                    
                    const yData = data.slice(headerEnd, headerEnd + ySize);
                    const uvData = data.slice(headerEnd + ySize, headerEnd + ySize + uvSize);
                    
                    // 4. 渲染
                    this._renderNv12Frame(yData, uvData);
                    
                    this.lastFrameId = frameId;
                    
                    // 统计
                    this.latency = performance.now() - startTime;
                    this.frameCount++;
                    
                    const now = performance.now();
                    if (now - this.lastFpsTime >= 1000) {
                        this.fps = this.frameCount;
                        this.frameCount = 0;
                        this.lastFpsTime = now;
                    }
                }
            } catch (e) {
                // 静默处理 (避免日志洪水)
                if (e.toString().includes('没有可用的帧')) {
                    // 正常情况，等待帧
                } else {
                    console.error('⚠️ 轮询错误:', e);
                }
            }
            
            // 继续轮询 (使用 rAF 同步显示刷新率)
            this.pollInterval = requestAnimationFrame(poll);
        };
        
        poll();
    }
    
    /**
     * 渲染 NV12 帧
     */
    _renderNv12Frame(yData, uvData) {
        const gl = this.gl;
        
        // 上传 Y 纹理 (单通道)
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.yTexture);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.LUMINANCE,
            this.yStride, this.videoHeight,
            0, gl.LUMINANCE, gl.UNSIGNED_BYTE,
            yData
        );
        
        // 上传 UV 纹理 (双通道，高度减半)
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.uvTexture);
        gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA,
            this.uvStride / 2, this.videoHeight / 2,
            0, gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE,
            uvData
        );
        
        // 设置 uniform
        gl.uniform1i(this.uYTexture, 0);
        gl.uniform1i(this.uUvTexture, 1);
        gl.uniform2f(this.uVideoSize, this.videoWidth, this.videoHeight);
        gl.uniform2f(this.uYStride, this.yStride, this.videoHeight);
        
        // 绘制
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    
    /**
     * 获取性能统计
     */
    getStats() {
        return {
            fps: this.fps,
            latency: this.latency.toFixed(1),
            frameId: this.lastFrameId,
            resolution: `${this.videoWidth}x${this.videoHeight}`,
        };
    }
    
    /**
     * 销毁渲染器
     */
    destroy() {
        this.stop();
        
        const gl = this.gl;
        if (this.program) gl.deleteProgram(this.program);
        if (this.yTexture) gl.deleteTexture(this.yTexture);
        if (this.uvTexture) gl.deleteTexture(this.uvTexture);
    }
}

export default ZeroCopyRenderer;
