/**
 * WebGL NV12 视频渲染器
 * 
 * 使用 WebGL shader 在 GPU 上进行 YUV→RGB 色彩转换
 * 与后端 FFmpeg QSV/CUDA 硬解码配合，实现全程 GPU 加速
 */

export class WebGLNv12Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        
        this.program = null;
        this.yTexture = null;
        this.uvTexture = null;
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.yStride = 0;
        this.uvStride = 0;
        
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 0;
        
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
        
        const fragmentShaderSrc = `
            precision mediump float;
            
            uniform sampler2D u_yTexture;
            uniform sampler2D u_uvTexture;
            uniform vec2 u_videoSize;
            uniform vec2 u_yStride;    // (stride, height)
            uniform vec2 u_uvStride;   // (stride, height/2)
            
            varying vec2 v_texCoord;
            
            void main() {
                // 计算实际纹理坐标 (考虑 stride padding)
                vec2 yCoord = v_texCoord;
                vec2 uvCoord = v_texCoord;
                
                // 如果有 padding，需要调整坐标
                // 实际像素 / stride 得到纹理坐标
                float xRatio = u_videoSize.x / u_yStride.x;
                yCoord.x = v_texCoord.x * xRatio;
                uvCoord.x = v_texCoord.x * xRatio;
                
                // 采样 Y (单通道纹理)
                float y = texture2D(u_yTexture, yCoord).r;
                
                // 采样 UV (RG 通道)
                vec2 uv = texture2D(u_uvTexture, uvCoord).rg;
                float u = uv.r - 0.5;
                float v = uv.g - 0.5;
                
                // BT.601 YUV→RGB 转换
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
            -1, -1,  // 左下
             1, -1,  // 右下
            -1,  1,  // 左上
             1,  1,  // 右上
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
        this.uUvStride = gl.getUniformLocation(this.program, 'u_uvStride');
        
        // 创建纹理
        this.yTexture = this._createTexture();
        this.uvTexture = this._createTexture();
        
        console.log('✅ WebGL NV12 渲染器初始化完成');
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
     * 更新视频尺寸配置
     */
    setVideoSize(width, height, yStride, uvStride) {
        if (this.videoWidth !== width || this.videoHeight !== height) {
            console.log(`📺 WebGL NV12: ${width}x${height} (Y stride: ${yStride}, UV stride: ${uvStride})`);
        }
        this.videoWidth = width;
        this.videoHeight = height;
        this.yStride = yStride;
        this.uvStride = uvStride;
    }
    
    /**
     * 渲染 NV12 帧
     * @param {Uint8Array} yData - Y 平面数据
     * @param {Uint8Array} uvData - UV 平面数据 (交错 UVUV...)
     */
    renderNv12Frame(yData, uvData) {
        const gl = this.gl;
        
        if (this.videoWidth === 0 || this.videoHeight === 0) {
            console.warn('⚠️ 视频尺寸未设置');
            return;
        }
        
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
        gl.uniform2f(this.uUvStride, this.uvStride, this.videoHeight / 2);
        
        // 绘制
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        // FPS 统计
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
    }
    
    /**
     * 处理来自 IPC 的 NV12 帧数据包
     * 格式: [magic(2)] [header(16)] [y_data] [uv_data]
     */
    handleNv12Packet(data) {
        if (!(data instanceof Uint8Array)) {
            data = new Uint8Array(data);
        }
        
        // 检查 magic
        if (data.length < 18 || data[0] !== 0xAA || data[1] !== 0xBB) {
            // 可能是元数据包 (0xFF 0xFE)
            if (data[0] === 0xFF && data[1] === 0xFE) {
                const metadataStr = new TextDecoder().decode(data.slice(2));
                try {
                    const metadata = JSON.parse(metadataStr);
                    if (metadata.type === 'nv12_metadata') {
                        this.setVideoSize(
                            metadata.width,
                            metadata.height,
                            metadata.y_stride,
                            metadata.uv_stride
                        );
                    }
                } catch (e) {
                    console.error('元数据解析错误:', e);
                }
            }
            return;
        }
        
        // 解析 header (little-endian)
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        const width = view.getUint32(2, true);
        const height = view.getUint32(6, true);
        const yStride = view.getUint32(10, true);
        const uvStride = view.getUint32(14, true);
        
        // 更新尺寸
        this.setVideoSize(width, height, yStride, uvStride);
        
        // 计算数据偏移
        const headerEnd = 18;
        const ySize = yStride * height;
        const uvSize = uvStride * (height / 2);
        
        if (data.length < headerEnd + ySize + uvSize) {
            console.warn('⚠️ NV12 数据包不完整');
            return;
        }
        
        const yData = data.slice(headerEnd, headerEnd + ySize);
        const uvData = data.slice(headerEnd + ySize, headerEnd + ySize + uvSize);
        
        // 渲染
        this.renderNv12Frame(yData, uvData);
    }
    
    /**
     * 获取当前 FPS
     */
    getFps() {
        return this.fps;
    }
    
    /**
     * 销毁渲染器
     */
    destroy() {
        const gl = this.gl;
        if (this.program) gl.deleteProgram(this.program);
        if (this.yTexture) gl.deleteTexture(this.yTexture);
        if (this.uvTexture) gl.deleteTexture(this.uvTexture);
    }
}

export default WebGLNv12Renderer;
