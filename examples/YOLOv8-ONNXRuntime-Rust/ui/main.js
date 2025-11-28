import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { FrameDetector } from './detector.js';

// 前端加载完成后显示窗口（避免白屏闪烁）
invoke('show_window').catch(console.error);

// ==================== 摄像头/桌面采集 ====================

class CaptureManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.running = false;
        this.animationId = null;
        this.timeoutId = null;  // 🔧 用于 setTimeout 模式
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.fps = 0;
        this.currentDevice = null;
        
        // 共享内存
        this.sharedMemory = null;
        this.shmName = null;
        this.lastFrameId = 0;
        
        // 🔧 遮挡渲染模式: true = 使用 setTimeout (窗口被遮挡时仍渲染)
        this.backgroundRenderMode = true;
        this.targetFps = 30;
    }
    
    async listDevices() {
        try {
            return await invoke('list_capture_devices');
        } catch (e) {
            console.error('列出设备失败:', e);
            return [];
        }
    }
    
    async start(deviceId, deviceType, width, height, fps = 30, region = null) {
        if (this.running) {
            await this.stop();
        }
        
        console.log(`🎥 启动采集: ${deviceId} (${deviceType}) @ ${width}x${height}, region:`, region);
        
        // start_capture 现在返回共享内存信息
        const shmInfo = await invoke('start_capture', {
            deviceId,
            deviceType,
            width,
            height,
            fps,
            region
        });
        
        console.log('📝 共享内存信息:', shmInfo);
        this.shmName = shmInfo.name;
        
        this.running = true;
        this.currentDevice = { id: deviceId, type: deviceType };
        this.lastFrameId = 0;
        this._lastDebugTime = null; // 重置调试时间
        console.log('🔄 启动渲染循环...');
        this.renderLoop();
    }
    
    async stop() {
        this.running = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        try {
            await invoke('stop_capture');
            console.log('🛑 采集已停止');
        } catch (e) {
            console.error('停止采集失败:', e);
        }
        
        this.currentDevice = null;
        this.shmName = null;
        this.lastFrameId = 0;
    }
    
    async renderLoop() {
        if (!this.running) {
            console.log('⏹️ 渲染循环已停止');
            return;
        }
        
        try {
            // 轮询帧信息 (检查是否有新帧)
            const frameInfo = await invoke('get_capture_frame_info');
            
            if (frameInfo && frameInfo.frame_id > this.lastFrameId) {
                // 有新帧，读取帧数据
                const response = await invoke('read_capture_frame');
                
                // Tauri 2 Response 格式 - 需要提取 ArrayBuffer
                let data;
                if (response instanceof ArrayBuffer) {
                    data = new Uint8Array(response);
                } else if (response instanceof Uint8Array) {
                    data = response;
                } else if (response && response.data) {
                    data = new Uint8Array(response.data);
                } else if (Array.isArray(response)) {
                    data = new Uint8Array(response);
                } else {
                    this.animationId = requestAnimationFrame(() => this.renderLoop());
                    return;
                }
                
                if (data.length > 24) {
                    // 解析 header: frame_id(8) + width(4) + height(4) + timestamp(8) = 24 bytes
                    const headerView = new DataView(data.buffer, data.byteOffset, 24);
                    const frameId = Number(headerView.getBigUint64(0, true));
                    const width = headerView.getUint32(8, true);
                    const height = headerView.getUint32(12, true);
                    
                    // 仅第一帧打印调试
                    if (this.lastFrameId === 0) {
                        console.log(`🖼️ 首帧: ${width}x${height}, 帧ID: ${frameId}`);
                    }
                    
                    this.lastFrameId = frameId;
                    
                    const expectedLen = width * height * 4;
                    const dataLen = data.length - 24;
                    
                    if (dataLen >= expectedLen && width > 0 && height > 0) {
                        // RGBA 数据 (从 offset 24 开始)
                        const rgbaData = new Uint8ClampedArray(data.buffer, data.byteOffset + 24, expectedLen);
                        
                        // 调整 canvas 尺寸
                        if (this.canvas.width !== width || this.canvas.height !== height) {
                            this.canvas.width = width;
                            this.canvas.height = height;
                            console.log(`📐 Canvas 调整为: ${width}x${height}`);
                            
                            // 同步 detection-overlay 尺寸
                            syncOverlayToCanvas(width, height);
                        }
                        
                        const imageData = new ImageData(rgbaData, width, height);
                        this.ctx.putImageData(imageData, 0, 0);
                        
                        // FPS 统计
                        this.frameCount++;
                        const now = performance.now();
                        if (now - this.lastFpsTime >= 1000) {
                            this.fps = this.frameCount / ((now - this.lastFpsTime) / 1000);
                            this.frameCount = 0;
                            this.lastFpsTime = now;
                            document.getElementById('fps').textContent = this.fps.toFixed(0);
                        }
                    }
                }
            }
        } catch (e) {
            // 忽略 "没有可用的帧" 错误
            if (!String(e).includes('没有可用的帧')) {
                console.warn('采集帧错误:', e);
            }
        }
        
        this.animationId = requestAnimationFrame(() => this.renderLoop());
    }
    
    // 获取当前帧用于检测
    getCurrentFrame() {
        if (!this.canvas.width || !this.canvas.height) return null;
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        return {
            data: imageData.data,
            width: this.canvas.width,
            height: this.canvas.height
        };
    }
}

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
        this.audioDecoder = null;
        this.audioContext = null;
        this.audioQueue = [];
        this.nextAudioTime = 0;
        this.audioCodec = null; // 记录音频编码格式
        this.audioSampleRate = 8000;
        this.audioChannels = 1;
        this.audioGain = 1.5; // 音频增益降低到1.5倍,避免削波
        
        // 🚀 实时性优化：帧队列管理
        this.pendingFrames = []; // 待处理帧队列
        this.maxPendingFrames = 2; // 最大缓存帧数
        this.lastFrameTime = 0;
        
        // 初始化 Canvas 尺寸为容器大小
        this.resizeCanvas();
        
        // 监听窗口大小变化
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // 初始化音频上下文
        this.initAudioContext();
        
        this.initDecoder();
    }
    
    initAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('🔊 Audio Context initialized:', this.audioContext.sampleRate, 'Hz');
    }
    
    resizeCanvas() {
        // Canvas 直接占满整个窗口
        const w = window.innerWidth;
        const h = window.innerHeight;
        
        console.log(`[Canvas Resize] Window: ${w}x${h}`);
        
        // 设置 Canvas 内部像素分辨率为窗口大小
        this.canvas.width = w;
        this.canvas.height = h;
        
        // CSS 尺寸也设为窗口大小(已通过 CSS 100vw/100vh 设置,这里可省略)
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        
        // 绘制测试背景,确保可见
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, w, h);
        
        // 绘制测试文字
        this.ctx.fillStyle = '#666';
        this.ctx.font = '20px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText('WAITING FOR STREAM...', w / 2, h / 2);
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
        
        // 🚀 清空帧队列
        this.pendingFrames = [];

        this.decoder = new VideoDecoder({
            output: (frame) => {
                // 🚀 实时性优化：帧队列管理
                // 如果队列已满，丢弃旧帧
                while (this.pendingFrames.length >= this.maxPendingFrames) {
                    const oldFrame = this.pendingFrames.shift();
                    oldFrame.close();
                }
                this.pendingFrames.push(frame);
                
                // 立即处理最新帧
                this.processLatestFrame();
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
    
    initAudioDecoder(codec = 'aac', sampleRate = 48000, channels = 2) {
        this.audioCodec = codec;
        this.audioSampleRate = sampleRate;
        this.audioChannels = channels;
        
        // 如果是 PCM 格式,不需要 WebCodecs 解码器
        if (codec === 'pcm_alaw' || codec === 'pcm_mulaw') {
            console.log(`🎵 PCM audio detected: ${codec.toUpperCase()} ${sampleRate}Hz ${channels}ch - 直接播放`);
            return;
        }
        
        if (!('AudioDecoder' in window)) {
            console.error('WebCodecs AudioDecoder not supported');
            return;
        }
        
        if (this.audioDecoder && this.audioDecoder.state !== 'closed') {
            this.audioDecoder.close();
        }
        
        this.audioDecoder = new AudioDecoder({
            output: (audioData) => {
                this.playAudioData(audioData);
                audioData.close();
            },
            error: (e) => console.error('Audio decoder error:', e),
        });
        
        // 根据编码格式配置
        let codecString;
        switch(codec) {
            case 'aac':
                codecString = 'mp4a.40.2'; // AAC-LC
                break;
            case 'opus':
                codecString = 'opus';
                break;
            case 'mp3':
                codecString = 'mp3';
                break;
            default:
                codecString = 'mp4a.40.2';
        }
        
        const config = {
            codec: codecString,
            sampleRate: sampleRate,
            numberOfChannels: channels,
        };
        
        console.log(`🎵 Configuring audio decoder: ${codec.toUpperCase()} ${sampleRate}Hz ${channels}ch`);
        
        AudioDecoder.isConfigSupported(config).then((support) => {
            if (support.supported) {
                console.log(`✅ Audio decoding supported`);
                this.audioDecoder.configure(config);
                this.nextAudioTime = this.audioContext.currentTime;
            } else {
                console.error(`❌ Audio codec not supported`);
            }
        });
    }
    
    playAudioData(audioData) {
        // 将 AudioData 转换为 AudioBuffer 并播放
        const buffer = this.audioContext.createBuffer(
            audioData.numberOfChannels,
            audioData.numberOfFrames,
            audioData.sampleRate
        );
        
        // 复制音频数据到 buffer
        for (let channel = 0; channel < audioData.numberOfChannels; channel++) {
            const channelData = new Float32Array(audioData.numberOfFrames);
            audioData.copyTo(channelData, { planeIndex: channel });
            buffer.copyToChannel(channelData, channel);
        }
        
        // 创建音频源并播放
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        
        // 计算播放时间,避免断续
        const currentTime = this.audioContext.currentTime;
        if (this.nextAudioTime < currentTime) {
            this.nextAudioTime = currentTime;
        }
        
        source.start(this.nextAudioTime);
        this.nextAudioTime += buffer.duration;
    }

    handleVideoChunk(data) {
        if (!this.isRunning) {
            console.warn('Received video chunk but renderer not running');
            return;
        }
        if (!this.decoder || this.decoder.state !== 'configured') {
            console.warn('Decoder not ready, state:', this.decoder?.state);
            return;
        }
        
        const chunk = new EncodedVideoChunk({
            type: 'key',
            timestamp: performance.now() * 1000,
            data: data
        });
        
        try {
            this.decoder.decode(chunk);
        } catch(e) {
            console.error('Video decode error:', e);
        }
    }
    
    handleAudioChunk(data) {
        // PCM 格式 - Rust后端已解码为PCM16,直接播放
        if (this.audioCodec === 'pcm_alaw' || this.audioCodec === 'pcm_mulaw') {
            this.playPCM16(data);
            return;
        }
        
        // 其他格式使用 WebCodecs 解码
        if (!this.audioDecoder || this.audioDecoder.state !== 'configured') {
            console.warn('⚠️ Audio decoder not ready:', this.audioDecoder?.state);
            return; // 音频解码器未就绪,静默跳过
        }
        
        const chunk = new EncodedAudioChunk({
            type: 'key', // 音频帧通常都是关键帧
            timestamp: performance.now() * 1000,
            data: data
        });
        
        try {
            this.audioDecoder.decode(chunk);
        } catch(e) {
            console.error('Audio decode error:', e);
        }
    }
    
    // PCM A-law 解码 (已废弃 - 后端处理)
    playPCM_ALAW(compressedData) {
        // 后端已解码,直接播放 PCM16
        this.playPCM16(compressedData);
    }
    
    // PCM μ-law 解码 (已废弃 - 后端处理)
    playPCM_MULAW(compressedData) {
        // 后端已解码,直接播放 PCM16
        this.playPCM16(compressedData);
    }
    
    // 播放后端已解码的 PCM16 数据 (Little Endian)
    playPCM16(data) {
        // 将 Uint8Array 转换为 Int16Array (Little Endian)
        const pcmData = new Int16Array(data.buffer, data.byteOffset, data.byteLength / 2);
        this.playPCMData(pcmData);
    }
    
    // 播放 PCM 数据
    playPCMData(pcmData) {
        const sampleRate = this.audioSampleRate;
        const channels = this.audioChannels;
        const frameCount = pcmData.length / channels;
        
        // 如果 AudioContext 被挂起,尝试恢复
        if (this.audioContext.state === 'suspended') {
            console.warn('⚠️ AudioContext suspended, resuming...');
            this.audioContext.resume();
        }
        
        // 🚀 实时性优化：检测音频延迟，如果延迟超过 500ms 则重置
        const currentTime = this.audioContext.currentTime;
        const audioDelay = this.nextAudioTime - currentTime;
        
        if (audioDelay > 0.5) {
            // 音频缓冲超过 500ms，重置到当前时间以恢复实时
            console.warn(`⚠️ [Audio] 缓冲过大 ${(audioDelay * 1000).toFixed(0)}ms, 重置音频同步`);
            this.nextAudioTime = currentTime;
        } else if (audioDelay < -0.1) {
            // 音频已落后，跳到当前时间
            this.nextAudioTime = currentTime;
        }
        
        // 创建 AudioBuffer
        const buffer = this.audioContext.createBuffer(channels, frameCount, sampleRate);
        
        // 转换 Int16 PCM 到 Float32 (-1.0 到 1.0),并应用增益
        for (let ch = 0; ch < channels; ch++) {
            const channelData = buffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i++) {
                const sampleIndex = channels === 1 ? i : i * channels + ch;
                let sample = pcmData[sampleIndex] / 32768.0; // 归一化到 -1.0~1.0
                sample *= this.audioGain; // 应用增益
                
                // 软限幅防止削波 (使用 tanh 函数平滑限制)
                if (Math.abs(sample) > 0.95) {
                    sample = Math.tanh(sample * 0.8); // 平滑削波
                }
                
                channelData[i] = sample;
            }
        }
        
        // 创建音频源并播放
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        
        source.start(this.nextAudioTime);
        this.nextAudioTime += buffer.duration;
    }
    
    // 保留旧的 handleChunk 方法用于 WebSocket (如果还在使用)
    handleChunk(data) {
        let chunkData = data;
        if (Array.isArray(data)) {
            chunkData = new Uint8Array(data);
        }
        
        // 调试:统计包类型
        if (!window.packetStats) {
            window.packetStats = { video: 0, audio: 0, metadata: 0, unknown: 0, lastLog: Date.now() };
        }
        
        // 检查是否为元数据消息 (魔数: 0xFF 0xFE)
        if (chunkData.length > 2 && chunkData[0] === 0xFF && chunkData[1] === 0xFE) {
            window.packetStats.metadata++;
            if (Date.now() - window.packetStats.lastLog > 2000) {
                console.log('📊 Packet stats:', window.packetStats);
                window.packetStats.lastLog = Date.now();
            }
            const metadataStr = new TextDecoder().decode(chunkData.slice(2));
            try {
                const metadata = JSON.parse(metadataStr);
                console.log('📥 Received metadata:', metadata);
                
                if (metadata.type === 'metadata') {
                    // 重新配置视频解码器
                    this.initDecoder(metadata.video_codec, metadata.width, metadata.height);
                    
                    // 配置音频解码器
                    if (metadata.audio_codec && metadata.audio_codec !== 'none') {
                        this.initAudioDecoder(metadata.audio_codec, metadata.sample_rate, metadata.channels);
                    }
                }
            } catch (e) {
                console.error('Failed to parse metadata:', e);
            }
            return;
        }
        
        // 检查类型标记 (0xAA=视频, 0xBB=音频)
        if (chunkData.length > 1) {
            const packetType = chunkData[0];
            const packetData = chunkData.slice(1);
            
            if (packetType === 0xAA) {
                // 视频包
                window.packetStats.video++;
                if (Date.now() - window.packetStats.lastLog > 2000) {
                    console.log('📊 Packet stats:', window.packetStats);
                    window.packetStats.lastLog = Date.now();
                }
                
                if (!this.isRunning) {
                    console.warn('Received video packet but renderer not running');
                    return;
                }
                if (!this.decoder || this.decoder.state !== 'configured') {
                    console.warn('Decoder not ready, state:', this.decoder?.state);
                    return;
                }
                
                const chunk = new EncodedVideoChunk({
                    type: 'key',
                    timestamp: performance.now() * 1000,
                    data: packetData
                });
                
                try {
                    this.decoder.decode(chunk);
                } catch(e) {
                    console.error('Video decode error:', e);
                }
            } else if (packetType === 0xBB) {
                // 音频包 - 暂时跳过 PCM_ALAW (WebCodecs 可能不支持)
                window.packetStats.audio++;
                if (Date.now() - window.packetStats.lastLog > 2000) {
                    console.log('📊 Packet stats:', window.packetStats);
                    window.packetStats.lastLog = Date.now();
                }
            } else {
                // 未知包类型
                window.packetStats.unknown++;
                if (window.packetStats.unknown < 5) {
                    console.warn('Unknown packet type:', '0x' + packetType.toString(16), 'first bytes:', Array.from(chunkData.slice(0, 10)));
                }
            }
        }
    }
    
    /**
     * 🚀 实时性优化：处理最新帧，丢弃所有旧帧
     */
    processLatestFrame() {
        if (!this.isRunning || this.pendingFrames.length === 0) return;
        
        // 只保留最新帧，关闭并丢弃其他帧
        while (this.pendingFrames.length > 1) {
            const oldFrame = this.pendingFrames.shift();
            oldFrame.close();
        }
        
        const frame = this.pendingFrames.shift();
        if (frame) {
            this.renderFrame(frame);
            frame.close();
        }
    }
    
    renderFrame(frame) {
        if (!this.isRunning) return;
        
        // 🚀 实时性：记录帧时间，用于延迟监控
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const delta = now - this.lastFrameTime;
            // 如果帧间隔超过 100ms，说明可能有延迟
            if (delta > 100 && !this.warnedOnce) {
                console.warn(`⚠️ 帧间隔过大: ${delta.toFixed(0)}ms`);
            }
        }
        this.lastFrameTime = now;
        
        // 更新视频源分辨率信息
        if (this.videoWidth !== frame.displayWidth || this.videoHeight !== frame.displayHeight) {
            this.videoWidth = frame.displayWidth;
            this.videoHeight = frame.displayHeight;
            console.log(`[Video Source] ${this.videoWidth}x${this.videoHeight}`);
            console.log(`[Canvas Size] ${this.canvas.width}x${this.canvas.height}`);
            console.log(`[Draw Command] drawImage(frame, 0, 0, ${this.canvas.width}, ${this.canvas.height})`);
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

// 初始化渲染器
const canvas = document.getElementById('canvas');
const renderer = new WebGLVideoRenderer(canvas);

// ==================== 图像缩放和拖拽功能 ====================

class CanvasTransform {
    constructor(canvas, overlayCanvas) {
        this.canvas = canvas;
        this.overlayCanvas = overlayCanvas;
        this.scale = 1;
        this.minScale = 0.1;
        this.maxScale = 10;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        // 变换容器
        this.container = document.getElementById('canvas-container');
        
        this.initEvents();
        this.updateTransform();
    }
    
    initEvents() {
        // 鼠标滚轮缩放
        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            // 获取鼠标相对于容器的位置
            const rect = this.container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // 计算缩放前鼠标指向的图像坐标
            const imgX = (mouseX - this.offsetX) / this.scale;
            const imgY = (mouseY - this.offsetY) / this.scale;
            
            // 计算新缩放比例
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * delta));
            
            // 更新偏移量，使鼠标位置保持不变
            this.offsetX = mouseX - imgX * newScale;
            this.offsetY = mouseY - imgY * newScale;
            this.scale = newScale;
            
            this.updateTransform();
            this.updateZoomDisplay();
        }, { passive: false });
        
        // 鼠标拖拽
        this.container.addEventListener('mousedown', (e) => {
            // 只响应左键拖拽
            if (e.button !== 0) return;
            // 如果点击的是控制面板或区域选择，不处理
            if (e.target.closest('#control-panel') || e.target.closest('#region-selector')) return;
            
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.container.style.cursor = 'grabbing';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            this.offsetX += deltaX;
            this.offsetY += deltaY;
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.updateTransform();
        });
        
        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.container.style.cursor = 'grab';
            }
        });
        
        // 双击重置视图
        this.container.addEventListener('dblclick', (e) => {
            if (e.target.closest('#control-panel') || e.target.closest('#region-selector')) return;
            this.resetView();
        });
        
        // 设置初始光标
        this.container.style.cursor = 'grab';
    }
    
    updateTransform() {
        const transform = `translate(${this.offsetX}px, ${this.offsetY}px) scale(${this.scale})`;
        this.canvas.style.transform = transform;
        this.canvas.style.transformOrigin = '0 0';
        this.overlayCanvas.style.transform = transform;
        this.overlayCanvas.style.transformOrigin = '0 0';
    }
    
    updateZoomDisplay() {
        const zoomEl = document.getElementById('zoom-level');
        if (zoomEl) {
            zoomEl.textContent = `${Math.round(this.scale * 100)}%`;
        }
    }
    
    resetView() {
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.updateTransform();
        this.updateZoomDisplay();
        console.log('🔄 视图已重置');
    }
    
    // 适应窗口
    fitToWindow() {
        const containerRect = this.container.getBoundingClientRect();
        const canvasWidth = this.canvas.width || containerRect.width;
        const canvasHeight = this.canvas.height || containerRect.height;
        
        const scaleX = containerRect.width / canvasWidth;
        const scaleY = containerRect.height / canvasHeight;
        this.scale = Math.min(scaleX, scaleY, 1);
        
        // 居中
        this.offsetX = (containerRect.width - canvasWidth * this.scale) / 2;
        this.offsetY = (containerRect.height - canvasHeight * this.scale) / 2;
        
        this.updateTransform();
        this.updateZoomDisplay();
    }
}

// 初始化变换控制（等待 DOM 加载完成）
let canvasTransform = null;
document.addEventListener('DOMContentLoaded', () => {
    const detectionOverlay = document.getElementById('detection-overlay');
    if (canvas && detectionOverlay) {
        canvasTransform = new CanvasTransform(canvas, detectionOverlay);
        window.canvasTransform = canvasTransform;
    }
});

// 初始化检测器
const frameDetector = new FrameDetector(renderer);

// 监听检测结果并在 Canvas 上绘制
window.addEventListener('yolo-detection', (event) => {
    const result = event.detail;
    
    // 调试：检查是否收到检测结果
    if (result.boxes && result.boxes.length > 0) {
        console.log(`🎯 绘制 ${result.boxes.length} 个检测框`);
    }
    
    drawDetections(result.boxes);
    
    // 更新检测统计
    const detectFpsEl = document.getElementById('detect-fps');
    const detectCountEl = document.getElementById('detect-count');
    if (detectFpsEl) detectFpsEl.textContent = result.detect_fps?.toFixed(1) || '0';
    if (detectCountEl) detectCountEl.textContent = result.boxes?.length || '0';
});

// UI 控制
const rtspInput = document.getElementById('rtsp-url');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusDiv = document.getElementById('status');
const historyToggle = document.getElementById('history-toggle');
const clearHistory = document.getElementById('clear-history');
const historyDropdown = document.getElementById('history-dropdown');
const controlPanel = document.getElementById('control-panel');
const panelHeader = document.getElementById('panel-header');
const toggleBtn = document.getElementById('toggle-btn');
const volumeSlider = document.getElementById('volume-slider');
const volumeValue = document.getElementById('volume-value');

// 检测控制按钮
let detectionFpsSlider;

// 检测框 overlay canvas (2D context，独立于 WebGL)
const detectionOverlay = document.getElementById('detection-overlay');
const detectionCtx = detectionOverlay.getContext('2d');

// 检测框缓存 - 用于平滑绘制
let cachedBoxes = [];
let drawScheduled = false;
let overlayWidth = window.innerWidth;
let overlayHeight = window.innerHeight;

// 同步 overlay 到 canvas 尺寸
function syncOverlayToCanvas(width, height) {
    if (detectionOverlay.width !== width || detectionOverlay.height !== height) {
        detectionOverlay.width = width;
        detectionOverlay.height = height;
        overlayWidth = width;
        overlayHeight = height;
        console.log(`🔄 [Overlay] 同步到 Canvas: ${width}x${height}`);
        
        // 立即重绘
        renderBoxesImmediate();
    }
}

// 同步 overlay canvas 尺寸 (窗口 resize 时)
function syncOverlaySize() {
    // 如果有 captureManager 且有 canvas，使用 canvas 尺寸
    const mainCanvas = document.getElementById('canvas');
    if (mainCanvas && mainCanvas.width > 0 && mainCanvas.height > 0) {
        // 只在真正需要时更新
        if (detectionOverlay.width !== mainCanvas.width || detectionOverlay.height !== mainCanvas.height) {
            syncOverlayToCanvas(mainCanvas.width, mainCanvas.height);
        }
    }
}
// 初始化时不立即同步，等待 canvas 尺寸确定
window.addEventListener('resize', syncOverlaySize);

// 立即渲染检测框（不依赖 RAF 调度）
function renderBoxesImmediate() {
    const ctx = detectionCtx;
    const w = overlayWidth;
    const h = overlayHeight;
    
    // 清除
    ctx.clearRect(0, 0, w, h);
    
    if (cachedBoxes.length === 0) return;
    
    // 预设样式 (每次绘制都要设置，因为 resize 会重置 context)
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.font = 'bold 16px monospace';
    
    cachedBoxes.forEach(box => {
        // 归一化坐标 (0-1) → 屏幕坐标
        const x1 = box.x1 * w;
        const y1 = box.y1 * h;
        const x2 = box.x2 * w;
        const y2 = box.y2 * h;
        const bw = x2 - x1;
        const bh = y2 - y1;
        
        // 绘制矩形框
        ctx.strokeRect(x1, y1, bw, bh);
        
        // 绘制标签背景
        const label = `${box.class_name} ${(box.confidence * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.fillRect(x1, y1 - 22, textWidth + 8, 22);
        
        // 绘制标签文字
        ctx.fillStyle = '#000';
        ctx.fillText(label, x1 + 4, y1 - 6);
    });
}

// 实际渲染检测框 (在 RAF 中调用)
function renderBoxes() {
    renderBoxesImmediate();
    drawScheduled = false;
}

// 更新检测框 (使用 RAF 合并绘制)
function drawDetections(boxes) {
    cachedBoxes = boxes || [];
    
    if (!drawScheduled) {
        drawScheduled = true;
        requestAnimationFrame(renderBoxes);
    }
}

// 音量控制
volumeSlider.addEventListener('input', (e) => {
    const gain = parseFloat(e.target.value);
    if (renderer) {
        renderer.audioGain = gain;
    }
    volumeValue.textContent = gain.toFixed(1) + 'x';
    console.log(`🔊 [Volume] Gain set to ${gain.toFixed(1)}x`);
});

// 折叠/展开控制面板
let isPanelCollapsed = false;
toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    isPanelCollapsed = !isPanelCollapsed;
    if (isPanelCollapsed) {
        controlPanel.classList.add('collapsed');
    } else {
        controlPanel.classList.remove('collapsed');
    }
});

// 拖拽功能
let isDragging = false;
let currentX;
let currentY;
let initialX;
let initialY;
let xOffset = 0;
let yOffset = 0;

panelHeader.addEventListener('mousedown', dragStart);
document.addEventListener('mousemove', drag);
document.addEventListener('mouseup', dragEnd);

function dragStart(e) {
    // 防止点击按钮时触发拖拽
    if (e.target.closest('#toggle-btn') || e.target.closest('#history-toggle')) {
        return;
    }
    if (e.target === panelHeader || e.target.closest('#panel-header')) {
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;
        isDragging = true;
        controlPanel.style.transition = 'none';
    }
}

function drag(e) {
    if (isDragging) {
        e.preventDefault();
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
        xOffset = currentX;
        yOffset = currentY;
        
        controlPanel.style.transform = `translate(${currentX}px, ${currentY}px)`;
    }
}

function dragEnd(e) {
    isDragging = false;
    controlPanel.style.transition = 'background 0.3s ease, border-color 0.3s ease';
}

// 加载历史记录
async function loadHistory() {
    try {
        const urls = await invoke('get_rtsp_history');
        
        historyDropdown.innerHTML = '';
        
        if (urls.length === 0) {
            historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,255,255,0.4); cursor: default;">暂无历史记录</div>';
        } else {
            urls.forEach(url => {
                const item = document.createElement('div');
                item.className = 'history-item';
                item.textContent = url;
                item.addEventListener('click', () => {
                    rtspInput.value = url;
                    historyDropdown.classList.remove('show');
                });
                historyDropdown.appendChild(item);
            });
        }
    } catch (err) {
        console.error('加载历史记录失败:', err);
        historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,100,100,0.8); cursor: default;">加载失败</div>';
    }
}

// 切换下拉列表
historyToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    if (historyDropdown.classList.contains('show')) {
        historyDropdown.classList.remove('show');
    } else {
        loadHistory();
        historyDropdown.classList.add('show');
    }
});

// 清空历史记录
clearHistory.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (confirm('确定要清空所有历史记录吗?')) {
        try {
            await invoke('clear_rtsp_history');
            showStatus('✅ 历史记录已清空');
            historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,255,255,0.4); cursor: default;">暂无历史记录</div>';
        } catch (err) {
            console.error('清空历史记录失败:', err);
            showStatus('❌ 清空失败');
        }
    }
});

// 点击外部关闭下拉列表
document.addEventListener('click', (e) => {
    if (!e.target.closest('.input-wrapper')) {
        historyDropdown.classList.remove('show');
    }
});

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
        
        // 创建两个独立的 Channel:视频和音频
        const videoChannel = new Channel();
        const audioChannel = new Channel();
        
        // 视频 Channel 处理器
        videoChannel.onmessage = (data) => {
            let binaryData = null;
            
            if (data instanceof Uint8Array) {
                binaryData = data;
            } else if (data instanceof ArrayBuffer) {
                binaryData = new Uint8Array(data);
            } else if (typeof data === 'object' && data !== null) {
                if (data.data) binaryData = data.data;
                else if (data.payload) binaryData = data.payload;
                else if (data.buffer) binaryData = data.buffer;
            }
            
            if (!binaryData) {
                console.warn('⚠️ 视频数据提取失败');
                return;
            }
            
            if (!(binaryData instanceof Uint8Array)) {
                binaryData = new Uint8Array(binaryData);
            }
            
            // 检查是否为元数据(0xFF 0xFE)
            if (binaryData.length > 2 && binaryData[0] === 0xFF && binaryData[1] === 0xFE) {
                const metadataStr = new TextDecoder().decode(binaryData.slice(2));
                try {
                    const metadata = JSON.parse(metadataStr);
                    console.log('📥 收到元数据:', metadata);
                    
                    // 配置视频解码器
                    renderer.initDecoder(metadata.video_codec, metadata.width, metadata.height);
                    
                    // 配置音频解码器(如果有)
                    if (metadata.audio_codec && metadata.audio_codec !== 'none') {
                        renderer.initAudioDecoder(metadata.audio_codec, metadata.sample_rate, metadata.channels);
                    }
                } catch (e) {
                    console.error('元数据解析错误:', e);
                }
            } else {
                // 纯视频数据
                renderer.handleVideoChunk(binaryData);
            }
        };
        
        // 音频 Channel 处理器
        let audioPacketCount = 0;
        audioChannel.onmessage = (data) => {
            audioPacketCount++;
            
            let binaryData = null;
            
            if (data instanceof Uint8Array) {
                binaryData = data;
            } else if (data instanceof ArrayBuffer) {
                binaryData = new Uint8Array(data);
            } else if (typeof data === 'object' && data !== null) {
                if (data.data) binaryData = data.data;
                else if (data.payload) binaryData = data.payload;
                else if (data.buffer) binaryData = data.buffer;
            }
            
            if (!binaryData) {
                console.warn('⚠️ [Audio Channel] Cannot extract binary data');
                return;
            }
            
            if (!(binaryData instanceof Uint8Array)) {
                binaryData = new Uint8Array(binaryData);
            }
            
            // 处理音频数据
            renderer.handleAudioChunk(binaryData);
        };
        
        const result = await invoke('start_rtsp_stream', { 
            url, 
            videoChannel,
            audioChannel 
        });
        console.log(result);
        
        // 保存到历史记录
        try {
            await invoke('add_rtsp_history', { url });
            console.log('✅ 已保存到历史记录');
        } catch (e) {
            console.warn('保存历史记录失败:', e);
        }
        
        renderer.start();
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        rtspInput.disabled = true;
        
        showStatus('✅ 监控已启动 - 请手动点击开启检测');
    } catch (err) {
        console.error('启动失败:', err);
        showStatus('❌ 启动失败: ' + err);
        startBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', async () => {
    // 先停止检测器
    if (frameDetector.isDetecting) {
        await frameDetector.stopDetector();
    }
    
    renderer.stop();
    stopBtn.classList.add('hidden');
    startBtn.classList.remove('hidden');
    startBtn.disabled = false;
    rtspInput.disabled = false;
    showStatus('⏹ 监控已停止');
});

// 暴露到全局方便调试
window.renderer = renderer;
window.frameDetector = frameDetector;

// 检测器按钮事件
const startDetectorBtn = document.getElementById('start-detector-btn');
const stopDetectorBtn = document.getElementById('stop-detector-btn');
const modelSelect = document.getElementById('model-select');
const deviceSelect = document.getElementById('device-select');

// 加载可用模型和设备列表
async function loadModelsAndDevices() {
    try {
        // 并行加载
        const [models, devices] = await Promise.all([
            invoke('get_available_models'),
            invoke('get_available_devices')
        ]);
        
        // 填充模型下拉框
        modelSelect.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
        
        // 默认选择 yolov8n.onnx (如果存在)
        const defaultModel = models.find(m => m.name === 'yolov8n.onnx');
        if (defaultModel) {
            modelSelect.value = defaultModel.path;
        }
        
        console.log(`📦 已加载 ${models.length} 个模型`);
        
        // 填充设备下拉框
        deviceSelect.innerHTML = '<option value="auto">🔄 Auto</option>';
        devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.id;
            const icon = device.id === 'cuda' ? '🎮' : (device.id === 'directml' ? '🖥️' : '💻');
            const status = device.available ? '✅' : '❌';
            option.textContent = `${icon} ${device.name} ${status}`;
            option.disabled = !device.available;
            deviceSelect.appendChild(option);
        });
        
        console.log(`⚡ 已加载 ${devices.length} 个设备`);
        
    } catch (err) {
        console.error('加载模型/设备列表失败:', err);
        modelSelect.innerHTML = '<option value="">加载失败</option>';
    }
}

// 页面加载时加载模型和设备列表
loadModelsAndDevices();

startDetectorBtn.addEventListener('click', async () => {
    try {
        startDetectorBtn.disabled = true;
        startDetectorBtn.innerHTML = '<span>🔄</span> 加载中...';
        
        // 获取用户选择
        const selectedModel = modelSelect.value;
        const selectedDevice = deviceSelect.value;
        
        if (!selectedModel) {
            showStatus('❌ 请选择模型');
            startDetectorBtn.disabled = false;
            startDetectorBtn.innerHTML = '<span>🎯</span> 开启检测';
            return;
        }
        
        console.log(`🚀 启动检测: 模型=${selectedModel}, 设备=${selectedDevice}`);
        
        const result = await frameDetector.startDetectorWithOptions(selectedModel, selectedDevice, 'bytetrack');
        
        if (result) {
            // 更新检测输入尺寸用于坐标缩放
            if (result.input_width) {
                console.log(`📐 检测输入尺寸: ${result.input_width}x${result.input_width}`);
            }
            
            // 显示实际使用的设备
            const deviceUsed = result.device || selectedDevice;
            showStatus(`✅ 检测器已启动 [${deviceUsed}]`);
            
            startDetectorBtn.classList.add('hidden');
            stopDetectorBtn.classList.remove('hidden');
        }
    } catch (err) {
        console.error('启动检测器失败:', err);
        showStatus('❌ 检测器启动失败: ' + (err.message || err));
        startDetectorBtn.disabled = false;
        startDetectorBtn.innerHTML = '<span>🎯</span> 开启检测';
    }
});

stopDetectorBtn.addEventListener('click', async () => {
    try {
        await frameDetector.stopDetector();
        
        // 清除检测框
        detectionCtx.clearRect(0, 0, detectionOverlay.width, detectionOverlay.height);
        
        stopDetectorBtn.classList.add('hidden');
        startDetectorBtn.classList.remove('hidden');
        startDetectorBtn.disabled = false;
        startDetectorBtn.innerHTML = '<span>🎯</span> 开启检测';
        showStatus('⏹ 检测器已停止');
    } catch (err) {
        console.error('停止检测器失败:', err);
        showStatus('❌ 停止检测器失败: ' + err);
    }
});

console.log('WebGL renderer initialized');
console.log('使用方法:');
console.log('  启动检测: frameDetector.startDetector("yolov8n", "bytetrack")');
console.log('  停止检测: frameDetector.stopDetector()');
console.log('  调整FPS:  frameDetector.setDetectionFps(15)');

// ==================== 摄像头/桌面采集 UI 集成 ====================

// 初始化采集管理器
const captureManager = new CaptureManager(canvas);
window.captureManager = captureManager;

// Tab 切换
const tabs = document.querySelectorAll('.tab');
const rtspInputGroup = document.querySelector('.input-group');
const rtspBtnGroup = document.getElementById('start-btn').parentElement;
const captureSettings = document.getElementById('capture-settings');
const captureControls = document.getElementById('capture-controls');
const captureDeviceSelect = document.getElementById('capture-device-select');
const captureResolution = document.getElementById('capture-resolution');
const startCaptureBtn = document.getElementById('start-capture-btn');
const stopCaptureBtn = document.getElementById('stop-capture-btn');

// 区域选择相关元素
const regionSelectRow = document.getElementById('region-select-row');
const selectRegionBtn = document.getElementById('select-region-btn');
const regionInfo = document.getElementById('region-info');
const regionCoords = document.getElementById('region-coords');
const clearRegionBtn = document.getElementById('clear-region-btn');
const regionSelectorOverlay = document.getElementById('region-selector-overlay');
const selectionBox = document.getElementById('selection-box');
const selectionInfo = document.getElementById('selection-info');
const recordingRegion = document.getElementById('recording-region');

// 当前输入源模式
let currentInputMode = 'rtsp';

// 选中的区域 (null 表示全屏)
let selectedRegion = null;

// Tab 点击事件
tabs.forEach(tab => {
    tab.addEventListener('click', async () => {
        // 更新 Tab 样式
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        const previousMode = currentInputMode;
        const mode = tab.dataset.tab;
        currentInputMode = mode;
        
        // 切换显示内容
        if (mode === 'rtsp') {
            // RTSP 模式 - 关闭录制指示器
            rtspInputGroup?.classList.remove('hidden');
            rtspBtnGroup?.classList.remove('hidden');
            captureSettings?.classList.add('hidden');
            captureControls?.classList.add('hidden');
            regionSelectRow?.classList.add('hidden');
            
            // 关闭录制指示器（如果有）
            if (previousMode === 'screen') {
                try {
                    await invoke('close_recording_indicator');
                } catch (e) {}
            }
        } else {
            // 摄像头/桌面模式
            rtspInputGroup?.classList.add('hidden');
            rtspBtnGroup?.classList.add('hidden');
            captureSettings?.classList.remove('hidden');
            captureControls?.classList.remove('hidden');
            
            // 桌面模式：隐藏区域选择按钮（不再需要），直接弹出录制指示器
            if (mode === 'screen') {
                regionSelectRow?.classList.add('hidden');  // 隐藏区域选择按钮
                regionInfo?.classList.add('hidden');
                
                // 直接打开录制指示器窗口（默认位置居中，640x480）
                try {
                    await invoke('open_recording_indicator', { region: null });
                    showStatus('🎯 拖动录制指示器框选择采集区域，调整大小后点击开始采集');
                } catch (e) {
                    console.error('打开录制指示器失败:', e);
                }
            } else if (mode === 'window') {
                // 窗口模式 - 不需要录制指示器
                regionSelectRow?.classList.add('hidden');
                regionInfo?.classList.add('hidden');
                
                if (previousMode === 'screen') {
                    try {
                        await invoke('close_recording_indicator');
                    } catch (e) {}
                }
            } else {
                // 摄像头模式 - 关闭录制指示器
                regionSelectRow?.classList.add('hidden');
                regionInfo?.classList.add('hidden');
                
                if (previousMode === 'screen') {
                    try {
                        await invoke('close_recording_indicator');
                    } catch (e) {}
                }
            }
            
            // 加载设备列表
            await loadCaptureDevices(mode);
        }
    });
});

// 加载采集设备列表，并自动选择默认设备
async function loadCaptureDevices(filterType) {
    if (!captureDeviceSelect) return;
    
    captureDeviceSelect.innerHTML = '<option value="">加载中...</option>';
    
    try {
        const devices = await captureManager.listDevices();
        captureDeviceSelect.innerHTML = '<option value="">选择设备...</option>';
        
        // 根据当前 Tab 过滤设备
        let filtered = devices;
        if (filterType === 'camera') {
            filtered = devices.filter(d => d.device_type === 'camera');
        } else if (filterType === 'screen') {
            filtered = devices.filter(d => d.device_type === 'screen');
        } else if (filterType === 'window') {
            filtered = devices.filter(d => d.device_type === 'window');
        }
        
        let defaultIndex = -1;
        
        filtered.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = JSON.stringify({ id: device.id, type: device.device_type });
            const icon = device.device_type === 'camera' ? '📷' : 
                        device.device_type === 'screen' ? '🖥️' : '🪟';
            option.textContent = `${icon} ${device.name}`;
            captureDeviceSelect.appendChild(option);
            
            // 自动选择默认设备
            if (filterType === 'camera') {
                // 摄像头模式：选择第一个摄像头
                if (defaultIndex === -1 && device.device_type === 'camera') {
                    defaultIndex = index;
                }
            } else if (filterType === 'screen') {
                // 桌面模式：选择主显示器（包含 "primary" 或 "主显示器"）
                if (device.id.includes('primary') || device.name.includes('主显示器')) {
                    defaultIndex = index;
                } else if (defaultIndex === -1 && device.device_type === 'screen') {
                    // 回退：选择第一个显示器
                    defaultIndex = index;
                }
            } else if (filterType === 'window') {
                // 窗口模式：选择第一个窗口
                if (defaultIndex === -1 && device.device_type === 'window') {
                    defaultIndex = index;
                }
            }
        });
        
        // 设置默认选中项 (+1 是因为有一个 "选择设备..." 占位选项)
        if (defaultIndex >= 0 && captureDeviceSelect.options.length > defaultIndex + 1) {
            captureDeviceSelect.selectedIndex = defaultIndex + 1;
            console.log(`🎯 自动选择默认设备: ${filtered[defaultIndex].name}`);
        }
        
        console.log(`📦 已加载 ${filtered.length} 个采集设备`);
    } catch (e) {
        console.error('加载设备列表失败:', e);
        captureDeviceSelect.innerHTML = '<option value="">加载失败</option>';
    }
}

// 开始采集
startCaptureBtn?.addEventListener('click', async () => {
    const deviceValue = captureDeviceSelect?.value;
    if (!deviceValue) {
        showStatus('❌ 请选择采集设备');
        return;
    }
    
    try {
        const device = JSON.parse(deviceValue);
        const [width, height] = (captureResolution?.value || '1280x720').split('x').map(Number);
        
        startCaptureBtn.disabled = true;
        startCaptureBtn.innerHTML = '<span>🔄</span> 启动中...';
        
        // 停止 RTSP 流（如果有）
        renderer.stop();
        
        // 如果是桌面采集，从录制指示器获取当前区域
        let captureRegion = null;
        if (device.type === 'screen') {
            try {
                const indicatorRegion = await invoke('get_recording_indicator_region');
                if (indicatorRegion) {
                    captureRegion = {
                        x: indicatorRegion.x,
                        y: indicatorRegion.y,
                        width: indicatorRegion.width,
                        height: indicatorRegion.height
                    };
                    console.log('📐 从录制指示器获取采集区域:', captureRegion);
                }
            } catch (e) {
                console.warn('获取录制指示器区域失败:', e);
            }
        }
        
        await captureManager.start(device.id, device.type, width, height, 30, captureRegion);
        
        startCaptureBtn.classList.add('hidden');
        stopCaptureBtn.classList.remove('hidden');
        captureDeviceSelect.disabled = true;
        captureResolution.disabled = true;
        
        if (device.type === 'screen' && captureRegion) {
            showStatus(`✅ 桌面区域采集已启动 (${captureRegion.width}×${captureRegion.height})`);
        } else {
            showStatus(`✅ 采集已启动 (${width}x${height})`);
        }
    } catch (e) {
        console.error('启动采集失败:', e);
        showStatus('❌ 启动采集失败: ' + e);
        startCaptureBtn.disabled = false;
        startCaptureBtn.innerHTML = '<span>▶</span> 开始采集';
    }
});

// 停止采集
stopCaptureBtn?.addEventListener('click', async () => {
    try {
        // 先停止检测器
        if (frameDetector.isDetecting) {
            await frameDetector.stopDetector();
        }
        
        // 注意：停止采集后不关闭录制指示器，让用户可以继续调整区域
        // 只有切换到其他输入模式时才关闭
        
        await captureManager.stop();
        
        stopCaptureBtn.classList.add('hidden');
        startCaptureBtn.classList.remove('hidden');
        startCaptureBtn.disabled = false;
        startCaptureBtn.innerHTML = '<span>▶</span> 开始采集';
        captureDeviceSelect.disabled = false;
        captureResolution.disabled = false;
        
        showStatus('⏹ 采集已停止 (可继续调整录制区域)');
    } catch (e) {
        console.error('停止采集失败:', e);
        showStatus('❌ 停止采集失败: ' + e);
    }
});

// ==================== 缩放按钮事件 ====================

document.getElementById('zoom-reset')?.addEventListener('click', () => {
    if (canvasTransform) {
        canvasTransform.resetView();
    }
});

document.getElementById('zoom-fit')?.addEventListener('click', () => {
    if (canvasTransform) {
        canvasTransform.fitToWindow();
    }
});

// ==================== 桌面区域选择功能 ====================

class RegionSelector {
    constructor() {
        this.recordingRegion = document.getElementById('recording-region');
        this.region = null; // { x, y, width, height }
        this.isListening = false;
        this.screenRegion = null; // 屏幕坐标区域（用于录制指示器）
        
        // 画中选择相关
        this.overlay = document.getElementById('region-select-overlay');
        this.selectBox = document.getElementById('region-select-box');
        this.isSelecting = false;
        this.startX = 0;
        this.startY = 0;
        this.currentBox = null;
        
        this.initEventListener();
        this.initInCanvasSelector();
    }
    
    initEventListener() {
        // 监听区域选择窗口发来的事件（全屏窗口选择方式）
        listen('region-selected', (event) => {
            const data = event.payload;
            
            if (data.cancelled) {
                console.log('❌ 区域选择已取消');
                showStatus('❌ 区域选择已取消');
            } else if (data.region) {
                this.region = data.region;
                this.screenRegion = data.region; // 全屏选择时，region 就是屏幕坐标
                this.onRegionSelected(this.region);
            }
        });
    }
    
    // 初始化画中选择功能
    initInCanvasSelector() {
        if (!this.overlay) return;
        
        const mainCanvas = document.getElementById('canvas');
        
        this.overlay.addEventListener('mousedown', (e) => {
            this.isSelecting = true;
            
            // 获取相对于 canvas 的坐标
            const rect = mainCanvas.getBoundingClientRect();
            this.startX = e.clientX - rect.left;
            this.startY = e.clientY - rect.top;
            
            // 保存 canvas 到屏幕的缩放比例
            this.scaleX = mainCanvas.width / rect.width;
            this.scaleY = mainCanvas.height / rect.height;
            this.canvasRect = rect;
            
            this.selectBox.style.left = `${e.clientX}px`;
            this.selectBox.style.top = `${e.clientY}px`;
            this.selectBox.style.width = '0';
            this.selectBox.style.height = '0';
            this.selectBox.style.display = 'block';
        });
        
        this.overlay.addEventListener('mousemove', (e) => {
            if (!this.isSelecting) return;
            
            const currentX = e.clientX - this.canvasRect.left;
            const currentY = e.clientY - this.canvasRect.top;
            
            const x = Math.min(this.startX, currentX);
            const y = Math.min(this.startY, currentY);
            const w = Math.abs(currentX - this.startX);
            const h = Math.abs(currentY - this.startY);
            
            // 屏幕坐标（用于显示）
            this.selectBox.style.left = `${this.canvasRect.left + x}px`;
            this.selectBox.style.top = `${this.canvasRect.top + y}px`;
            this.selectBox.style.width = `${w}px`;
            this.selectBox.style.height = `${h}px`;
            
            // 保存当前选择框（canvas 坐标）
            this.currentBox = {
                x: Math.round(x * this.scaleX),
                y: Math.round(y * this.scaleY),
                width: Math.round(w * this.scaleX),
                height: Math.round(h * this.scaleY)
            };
        });
        
        this.overlay.addEventListener('mouseup', () => {
            this.isSelecting = false;
        });
        
        // ESC 取消, Enter 确认
        document.addEventListener('keydown', (e) => {
            if (!this.overlay || this.overlay.classList.contains('hidden')) return;
            
            if (e.key === 'Escape') {
                this.cancelInCanvasSelect();
            } else if (e.key === 'Enter') {
                this.confirmInCanvasSelect();
            }
        });
    }
    
    // 开始画中选择
    startInCanvasSelect() {
        if (!this.overlay) {
            // 后备：使用旧的全屏窗口方式
            this.show();
            return;
        }
        
        // 检查是否正在采集
        if (!captureManager.running) {
            showStatus('⚠️ 请先开始全屏采集，然后再选择区域');
            return;
        }
        
        this.overlay.classList.remove('hidden');
        this.selectBox.style.display = 'none';
        this.currentBox = null;
        showStatus('🎯 在视频画面上拖拽选择区域，ESC 取消，Enter 确认');
    }
    
    // 取消画中选择
    cancelInCanvasSelect() {
        this.overlay.classList.add('hidden');
        this.selectBox.style.display = 'none';
        this.currentBox = null;
        showStatus('❌ 区域选择已取消');
    }
    
    // 确认画中选择
    async confirmInCanvasSelect() {
        if (!this.currentBox || this.currentBox.width < 50 || this.currentBox.height < 50) {
            showStatus('⚠️ 请选择一个有效区域（至少 50x50）');
            return;
        }
        
        this.overlay.classList.add('hidden');
        this.selectBox.style.display = 'none';
        
        // 保存区域
        this.region = this.currentBox;
        this.onRegionSelected(this.region);
        
        // 询问是否重新采集
        const shouldRestart = confirm(`已选择区域 ${this.region.width}×${this.region.height}\n\n是否重新启动采集（只采集选中区域）？`);
        
        if (shouldRestart && captureManager.running) {
            const device = captureManager.currentDevice;
            if (device && device.type === 'screen') {
                showStatus('🔄 重新启动区域采集...');
                
                // 停止当前采集
                await captureManager.stop();
                
                // 以区域模式重新启动
                const captureRegion = {
                    x: this.region.x,
                    y: this.region.y,
                    width: this.region.width,
                    height: this.region.height
                };
                
                await captureManager.start(device.id, device.type, 1920, 1080, 30, captureRegion);
                showStatus(`✅ 区域采集已启动 (${captureRegion.width}×${captureRegion.height})`);
                
                // 开始闪烁指示
                this.startRecordingBlink();
            }
        }
    }
    
    async show() {
        try {
            // 调用后端打开全屏选择窗口
            await invoke('open_region_selector');
            showStatus('🎯 在屏幕上拖动选择区域，ESC 取消，Enter 确认');
        } catch (e) {
            console.error('打开区域选择器失败:', e);
            showStatus('❌ 打开区域选择器失败: ' + e);
        }
    }
    
    onRegionSelected(region) {
        selectedRegion = region;
        
        // 更新 UI
        if (regionCoords) {
            regionCoords.textContent = `(${region.x}, ${region.y}) ${region.width}×${region.height}`;
        }
        if (regionInfo) {
            regionInfo.classList.remove('hidden');
        }
        
        showStatus(`✂️ 已选择区域: ${region.width}×${region.height}`);
        console.log(`✂️ 已选择区域: (${region.x}, ${region.y}) ${region.width}×${region.height}`);
    }
    
    clearRegion() {
        this.region = null;
        this.screenRegion = null;
        selectedRegion = null;
        
        if (regionInfo) {
            regionInfo.classList.add('hidden');
        }
        if (regionCoords) {
            regionCoords.textContent = '未选择';
        }
        
        this.stopRecordingBlink();
        console.log('🗑️ 已清除区域选择');
    }
    
    // 开始录制闪烁 - 在屏幕上被采集的区域显示闪烁边框
    async startRecordingBlink() {
        if (!this.screenRegion) {
            console.warn('⚠️ 没有屏幕区域信息，无法显示录制指示器');
            return;
        }
        
        try {
            // 调用后端创建录制指示器窗口
            await invoke('open_recording_indicator', {
                region: {
                    x: Math.round(this.screenRegion.x),
                    y: Math.round(this.screenRegion.y),
                    width: Math.round(this.screenRegion.width),
                    height: Math.round(this.screenRegion.height)
                }
            });
            
            console.log(`🔴 录制指示器已显示 (${this.screenRegion.width}×${this.screenRegion.height}) @ (${this.screenRegion.x}, ${this.screenRegion.y})`);
        } catch (e) {
            console.error('打开录制指示器失败:', e);
        }
    }
    
    // 停止录制闪烁
    async stopRecordingBlink() {
        try {
            await invoke('close_recording_indicator');
            console.log('⬛ 录制指示器已关闭');
        } catch (e) {
            // 忽略关闭失败（可能窗口已不存在）
        }
    }
    
    getRegion() {
        return this.region;
    }
}

// 初始化区域选择器
const regionSelector = new RegionSelector();
window.regionSelector = regionSelector;

// 区域选择按钮事件 - 使用画中选择
selectRegionBtn?.addEventListener('click', async () => {
    // 如果正在采集，使用画中选择；否则使用全屏窗口选择
    if (captureManager.running) {
        regionSelector.startInCanvasSelect();
    } else {
        // 提示用户先开始采集
        showStatus('💡 提示：先开始全屏采集，然后点击此按钮在画面上选择区域');
        // 也可以使用旧的全屏窗口方式
        await regionSelector.show();
    }
});

// 清除区域按钮事件
clearRegionBtn?.addEventListener('click', () => {
    regionSelector.clearRegion();
    showStatus('🗑️ 已清除区域选择');
});

console.log('🎥 摄像头/桌面采集模块已加载');
console.log('🔍 缩放功能: 滚轮缩放, 拖拽平移, 双击重置');
console.log('✂️ 区域选择: 采集时点击按钮在画面上直接选择');
