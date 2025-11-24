import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// 重写 console.log 以发送到后端
const originalLog = console.log;
const originalError = console.error;
const originalWarn = console.warn;

function sendLog(level, args) {
    try {
        const msg = args.map(arg => 
            typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
        ).join(' ');
        invoke('log_frontend', { msg: `[${level}] ${msg}` }).catch(() => {});
    } catch (e) {}
}

console.log = (...args) => {
    originalLog(...args);
    sendLog('INFO', args);
};

console.error = (...args) => {
    originalError(...args);
    sendLog('ERROR', args);
};

console.warn = (...args) => {
    originalWarn(...args);
    sendLog('WARN', args);
};

window.addEventListener('error', (event) => {
    console.error('Global Error:', event.message, 'at', event.filename, ':', event.lineno);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled Rejection:', event.reason);
});

class WebGLVideoRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 0;
        this.isRunning = false;
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.warnedOnce = false;
        
        this.decoder = null;
        
        // 初始化 Canvas 尺寸为容器大小
        this.resizeCanvas();
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => this.resizeCanvas());
        
        this.initDecoder();
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const w = container.clientWidth;
        const h = container.clientHeight;
        
        console.log(`[Canvas Resize] Container: ${w}x${h}`);
        
        // 设置 Canvas 内部像素分辨率
        this.canvas.width = w;
        this.canvas.height = h;
        
        // 强制设置 CSS 尺寸为容器大小
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.canvas.style.display = 'block';
    }
    
    initDecoder(codec = 'hevc', width = 3840, height = 2160) {
        if (!('VideoDecoder' in window)) {
            console.error("WebCodecs API not supported");
            return;
        }

        // 如果已有解码器,先关闭
        if (this.decoder && this.decoder.state !== 'closed') {
            this.decoder.close();
        }

        this.decoder = new VideoDecoder({
            output: (frame) => {
                this.renderFrame(frame);
                frame.close();
            },
            error: (e) => console.error("Decoder error:", e),
        });

        // 根据编码格式和分辨率动态配置
        let codecString;
        if (codec === 'h264') {
            // H.264 编码字符串 (根据分辨率选择 Level)
            const level = height > 1080 ? '5.1' : '4.0';
            codecString = `avc1.64${level === '5.1' ? '0033' : '0028'}`;
        } else {
            // H.265/HEVC 编码字符串
            const level = height > 2160 ? 'L156' : (height > 1080 ? 'L153' : 'L120');
            codecString = `hvc1.1.6.${level}.B0`;
        }
        
        const config = {
            codec: codecString,
            codedWidth: width,
            codedHeight: height,
        };
        
        console.log(`🎬 Configuring decoder: ${codec.toUpperCase()} ${width}x${height}`);
        console.log(`   Codec string: ${codecString}`);
        
        VideoDecoder.isConfigSupported(config).then((support) => {
            if (support.supported) {
                console.log(`✅ ${codec.toUpperCase()} ${width}x${height} Decoding Supported`);
                this.decoder.configure(config);
            } else {
                console.error(`❌ ${codec.toUpperCase()} ${width}x${height} NOT Supported`);
                alert(`您的浏览器不支持 ${codec.toUpperCase()} ${width}x${height} 解码。`);
            }
        });
    }

    handleChunk(data) {
        let chunkData = data;
        if (Array.isArray(data)) {
            chunkData = new Uint8Array(data);
        }
        
        // 检查是否为元数据消息 (魔数: 0xFF 0xFE)
        if (chunkData.length > 2 && chunkData[0] === 0xFF && chunkData[1] === 0xFE) {
            const metadataStr = new TextDecoder().decode(chunkData.slice(2));
            try {
                const metadata = JSON.parse(metadataStr);
                console.log('📥 Received metadata:', metadata);
                
                if (metadata.type === 'metadata') {
                    // 重新配置解码器
                    this.initDecoder(metadata.codec, metadata.width, metadata.height);
                }
            } catch (e) {
                console.error('Failed to parse metadata:', e);
            }
            return;
        }
        
        if (!this.isRunning || !this.decoder || this.decoder.state !== 'configured') return;

        const chunk = new EncodedVideoChunk({
            type: 'key',
            timestamp: performance.now() * 1000,
            data: chunkData
        });
        
        try {
            this.decoder.decode(chunk);
        } catch(e) {
            console.error("Decode error:", e);
        }
    }
    
    renderFrame(frame) {
        if (!this.isRunning) return;
        
        // 更新视频源分辨率信息
        if (this.videoWidth !== frame.displayWidth || this.videoHeight !== frame.displayHeight) {
            this.videoWidth = frame.displayWidth;
            this.videoHeight = frame.displayHeight;
            console.log(`[Video Source] ${this.videoWidth}x${this.videoHeight}`);
            console.log(`[Canvas Size] ${this.canvas.width}x${this.canvas.height}`);
            console.log(`[Draw Command] drawImage(frame, 0, 0, ${this.canvas.width}, ${this.canvas.height})`);
            document.getElementById('resolution').textContent = 
                `${this.videoWidth}x${this.videoHeight} -> ${this.canvas.width}x${this.canvas.height}`;
        }

        // 清空画布并绘制整个帧（拉伸填充）
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 明确指定源区域和目标区域
        // drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
        this.ctx.drawImage(
            frame, 
            0, 0, frame.displayWidth, frame.displayHeight,  // 源：整个视频帧
            0, 0, this.canvas.width, this.canvas.height     // 目标：整个 Canvas
        );

        // Update FPS
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = now;
            document.getElementById('fps').textContent = this.fps;
        }
    }

    start() {
        this.isRunning = true;
    }
    
    stop() {
        this.isRunning = false;
    }
}

// 初始化
const canvas = document.getElementById('canvas');
const renderer = new WebGLVideoRenderer(canvas);

// UI 控制
const rtspInput = document.getElementById('rtsp-url');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusDiv = document.getElementById('status');

function showStatus(message, duration = 3000) {
    statusDiv.textContent = message;
    statusDiv.classList.add('show');
    setTimeout(() => {
        statusDiv.classList.remove('show');
    }, duration);
}

startBtn.addEventListener('click', async () => {
    const url = rtspInput.value.trim();
    if (!url) {
        showStatus('❌ 请输入 RTSP 地址');
        return;
    }
    
    try {
        startBtn.disabled = true;
        showStatus('🚀 正在启动...');
        
        const channel = new Channel();
        channel.onmessage = (message) => {
            renderer.handleChunk(message);
        };
        
        const result = await invoke('start_rtsp_stream', { url, channel });
        console.log(result);
        
        renderer.start();
        startBtn.style.display = 'none';
        stopBtn.style.display = 'block';
        rtspInput.disabled = true;
        
        showStatus('✅ 监控已启动');
    } catch (err) {
        console.error('启动失败:', err);
        showStatus('❌ 启动失败: ' + err);
        startBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', () => {
    renderer.stop();
    stopBtn.style.display = 'none';
    startBtn.style.display = 'block';
    startBtn.disabled = false;
    rtspInput.disabled = false;
    showStatus('⏹ 监控已停止');
});

console.log('WebGL renderer initialized');
