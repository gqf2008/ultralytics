import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { FrameDetector } from './detector.js';

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
        
        // 添加测试音按钮
        this.addTestAudioButton();
    }
    
    addTestAudioButton() {
        const testBtn = document.createElement('button');
        testBtn.textContent = '🔊 测试音频';
        testBtn.style.cssText = 'position: fixed; top: 10px; right: 10px; z-index: 10000; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-family: monospace;';
        testBtn.onclick = () => this.playTestTone();
        document.body.appendChild(testBtn);
    }
    
    playTestTone() {
        console.log('🔔 Playing test tone...');
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(440, this.audioContext.currentTime); // A4 音符
        
        gainNode.gain.setValueAtTime(0.3, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.5);
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + 0.5);
        
        console.log('✅ Test tone played');
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

// 同步 overlay canvas 尺寸
function syncOverlaySize() {
    const newW = window.innerWidth;
    const newH = window.innerHeight;
    
    // 只有尺寸真正改变时才更新
    if (detectionOverlay.width !== newW || detectionOverlay.height !== newH) {
        detectionOverlay.width = newW;
        detectionOverlay.height = newH;
        overlayWidth = newW;
        overlayHeight = newH;
        console.log(`🔄 [Overlay] 尺寸更新: ${newW}x${newH}`);
        
        // 立即重绘（不使用 RAF 调度，避免被跳过）
        renderBoxesImmediate();
    }
}
syncOverlaySize();
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
        startBtn.style.display = 'none';
        stopBtn.style.display = 'block';
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
    stopBtn.style.display = 'none';
    startBtn.style.display = 'block';
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
        startDetectorBtn.textContent = '🔄 加载中...';
        
        // 获取用户选择
        const selectedModel = modelSelect.value;
        const selectedDevice = deviceSelect.value;
        
        if (!selectedModel) {
            showStatus('❌ 请选择模型');
            startDetectorBtn.disabled = false;
            startDetectorBtn.textContent = '▶ 开启检测';
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
            
            startDetectorBtn.style.display = 'none';
            stopDetectorBtn.style.display = 'block';
        }
    } catch (err) {
        console.error('启动检测器失败:', err);
        showStatus('❌ 检测器启动失败: ' + (err.message || err));
        startDetectorBtn.disabled = false;
        startDetectorBtn.textContent = '▶ 开启检测';
    }
});

stopDetectorBtn.addEventListener('click', async () => {
    try {
        await frameDetector.stopDetector();
        
        // 清除检测框
        detectionCtx.clearRect(0, 0, detectionOverlay.width, detectionOverlay.height);
        
        stopDetectorBtn.style.display = 'none';
        startDetectorBtn.style.display = 'block';
        startDetectorBtn.disabled = false;
        startDetectorBtn.textContent = '▶ 开启检测';
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
