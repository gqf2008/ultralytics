import { invoke, Channel } from '@tauri-apps/api/core';
import { llmManager, defaultLlmConfig } from './llmInference.js';
import { ZeroCopyRenderer } from './zeroCopyRenderer.js';

// 前端加载完成后显示窗口（避免白屏闪烁）
invoke('show_window').catch(console.error);

// ==================== 日志发送到后端 ====================

const originalLog = console.log;
const originalError = console.error;
const originalWarn = console.warn;

async function sendLog(level, args) {
    const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ');
    try {
        await invoke('frontend_log', { level, message });
    } catch (e) { }
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

// ==================== WebGL 视频渲染器 ====================

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
        this.audioCodec = null;
        this.audioSampleRate = 8000;
        this.audioChannels = 1;
        this.audioGain = 1.5;
        
        // 实时性优化：帧队列管理
        this.pendingFrames = [];
        this.maxPendingFrames = 2;
        this.lastFrameTime = 0;
        
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.initAudioContext();
        this.initDecoder();
    }
    
    initAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('🔊 Audio Context initialized:', this.audioContext.sampleRate, 'Hz');
    }
    
    resizeCanvas() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        
        console.log(`[Canvas Resize] Window: ${w}x${h}`);
        
        this.canvas.width = w;
        this.canvas.height = h;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, w, h);
        
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

        if (this.decoder && this.decoder.state !== 'closed') {
            this.decoder.close();
        }
        
        this.pendingFrames = [];

        this.decoder = new VideoDecoder({
            output: (frame) => {
                while (this.pendingFrames.length >= this.maxPendingFrames) {
                    const oldFrame = this.pendingFrames.shift();
                    oldFrame.close();
                }
                this.pendingFrames.push(frame);
                this.processLatestFrame();
            },
            error: (e) => console.error("Decoder error:", e),
        });

        let codecString;
        if (codec === 'h264') {
            const level = height > 1080 ? '5.1' : '4.0';
            codecString = `avc1.64${level === '5.1' ? '0033' : '0028'}`;
        } else {
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
        
        let codecString;
        switch(codec) {
            case 'aac':
                codecString = 'mp4a.40.2';
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
        const buffer = this.audioContext.createBuffer(
            audioData.numberOfChannels,
            audioData.numberOfFrames,
            audioData.sampleRate
        );
        
        for (let channel = 0; channel < audioData.numberOfChannels; channel++) {
            const channelData = new Float32Array(audioData.numberOfFrames);
            audioData.copyTo(channelData, { planeIndex: channel });
            buffer.copyToChannel(channelData, channel);
        }
        
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        
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
        if (this.audioCodec === 'pcm_alaw' || this.audioCodec === 'pcm_mulaw') {
            this.playPCM16(data);
            return;
        }
        
        if (!this.audioDecoder || this.audioDecoder.state !== 'configured') {
            console.warn('⚠️ Audio decoder not ready:', this.audioDecoder?.state);
            return;
        }
        
        const chunk = new EncodedAudioChunk({
            type: 'key',
            timestamp: performance.now() * 1000,
            data: data
        });
        
        try {
            this.audioDecoder.decode(chunk);
        } catch(e) {
            console.error('Audio decode error:', e);
        }
    }
    
    playPCM16(data) {
        const pcmData = new Int16Array(data.buffer, data.byteOffset, data.byteLength / 2);
        this.playPCMData(pcmData);
    }
    
    playPCMData(pcmData) {
        const sampleRate = this.audioSampleRate;
        const channels = this.audioChannels;
        const frameCount = pcmData.length / channels;
        
        if (this.audioContext.state === 'suspended') {
            console.warn('⚠️ AudioContext suspended, resuming...');
            this.audioContext.resume();
        }
        
        const currentTime = this.audioContext.currentTime;
        const audioDelay = this.nextAudioTime - currentTime;
        
        if (audioDelay > 0.5) {
            console.warn(`⚠️ [Audio] 缓冲过大 ${(audioDelay * 1000).toFixed(0)}ms, 重置音频同步`);
            this.nextAudioTime = currentTime;
        } else if (audioDelay < -0.1) {
            this.nextAudioTime = currentTime;
        }
        
        const buffer = this.audioContext.createBuffer(channels, frameCount, sampleRate);
        
        for (let ch = 0; ch < channels; ch++) {
            const channelData = buffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i++) {
                const sampleIndex = channels === 1 ? i : i * channels + ch;
                let sample = pcmData[sampleIndex] / 32768.0;
                sample *= this.audioGain;
                
                if (Math.abs(sample) > 0.95) {
                    sample = Math.tanh(sample * 0.8);
                }
                
                channelData[i] = sample;
            }
        }
        
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        
        source.start(this.nextAudioTime);
        this.nextAudioTime += buffer.duration;
    }
    
    processLatestFrame() {
        if (!this.isRunning || this.pendingFrames.length === 0) return;
        
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
        
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const delta = now - this.lastFrameTime;
            if (delta > 100 && !this.warnedOnce) {
                console.warn(`⚠️ 帧间隔过大: ${delta.toFixed(0)}ms`);
            }
        }
        this.lastFrameTime = now;
        
        if (this.videoWidth !== frame.displayWidth || this.videoHeight !== frame.displayHeight) {
            this.videoWidth = frame.displayWidth;
            this.videoHeight = frame.displayHeight;
            console.log(`[Video Source] ${this.videoWidth}x${this.videoHeight}`);
            console.log(`[Canvas Size] ${this.canvas.width}x${this.canvas.height}`);
        }

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(
            frame, 
            0, 0, frame.displayWidth, frame.displayHeight,
            0, 0, this.canvas.width, this.canvas.height
        );

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
        this.hwDecodeRunning = false;
    }
    
    // 硬件解码模式 - 事件驱动，收到帧就绪信号后读取数据
    startHardwareDecodeMode(frameChannel) {
        this.isRunning = true;
        this.hwDecodeRunning = true;
        this.lastFrameId = 0;
        console.log('🎬 启动后端硬解渲染模式 (事件驱动)');
        
        // 监听帧就绪信号
        frameChannel.onmessage = async (signal) => {
            if (!this.hwDecodeRunning) return;
            
            try {
                // 收到帧就绪信号，读取完整数据
                const response = await invoke('get_shared_memory_frame');
                
                // 确保是 Uint8Array
                let data;
                if (response instanceof ArrayBuffer) {
                    data = new Uint8Array(response);
                } else if (response instanceof Uint8Array) {
                    data = response;
                } else if (Array.isArray(response)) {
                    data = new Uint8Array(response);
                } else {
                    console.warn('未知的响应类型:', typeof response);
                    return;
                }
                
                if (data && data.length > 24) {
                    // 解析 header: frame_id(8) + width(4) + height(4) + timestamp(8) = 24 bytes
                    const headerView = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    const frameId = Number(headerView.getBigUint64(0, true));
                    const width = headerView.getUint32(8, true);
                    const height = headerView.getUint32(12, true);
                    
                    // 首帧日志
                    if (this.lastFrameId === 0) {
                        console.log(`🎬 首帧: id=${frameId}, ${width}x${height}, 数据大小=${data.length}`);
                    }
                    
                    this.lastFrameId = frameId;
                    
                    // 提取 RGBA 数据 (跳过 24 字节 header)
                    const frameData = new Uint8ClampedArray(data.buffer, data.byteOffset + 24, width * height * 4);
                    
                    // 更新 canvas 尺寸
                    if (this.canvas.width !== width || this.canvas.height !== height) {
                        this.canvas.width = width;
                        this.canvas.height = height;
                        this.videoWidth = width;
                        this.videoHeight = height;
                        console.log(`📐 Canvas 尺寸更新: ${width}x${height}`);
                        
                        if (typeof syncOverlayToCanvas === 'function') {
                            syncOverlayToCanvas(width, height);
                        }
                    }
                    
                    // 创建 ImageData 并绘制
                    const imageData = new ImageData(frameData, width, height);
                    this.ctx.putImageData(imageData, 0, 0);
                    
                    // 更新 FPS
                    this.frameCount++;
                    const now = performance.now();
                    if (now - this.lastTime >= 1000) {
                        this.fps = this.frameCount;
                        this.frameCount = 0;
                        this.lastTime = now;
                        const fpsEl = document.getElementById('fps');
                        if (fpsEl) fpsEl.textContent = this.fps;
                    }
                }
            } catch (e) {
                if (!e.toString().includes('没有可用')) {
                    console.warn('获取帧失败:', e);
                }
            }
        };
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
        
        this.container = document.getElementById('canvas-container');
        
        this.initEvents();
        this.updateTransform();
    }
    
    initEvents() {
        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const rect = this.container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const imgX = (mouseX - this.offsetX) / this.scale;
            const imgY = (mouseY - this.offsetY) / this.scale;
            
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * delta));
            
            this.offsetX = mouseX - imgX * newScale;
            this.offsetY = mouseY - imgY * newScale;
            this.scale = newScale;
            
            this.updateTransform();
            this.updateZoomDisplay();
        }, { passive: false });
        
        this.container.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            if (e.target.closest('#control-panel')) return;
            
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
        
        this.container.addEventListener('dblclick', (e) => {
            if (e.target.closest('#control-panel')) return;
            this.resetView();
        });
        
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
    
    fitToWindow() {
        const containerRect = this.container.getBoundingClientRect();
        const canvasWidth = this.canvas.width || containerRect.width;
        const canvasHeight = this.canvas.height || containerRect.height;
        
        const scaleX = containerRect.width / canvasWidth;
        const scaleY = containerRect.height / canvasHeight;
        this.scale = Math.min(scaleX, scaleY, 1);
        
        this.offsetX = (containerRect.width - canvasWidth * this.scale) / 2;
        this.offsetY = (containerRect.height - canvasHeight * this.scale) / 2;
        
        this.updateTransform();
        this.updateZoomDisplay();
    }
}

// 初始化变换控制（等待 DOM 加载完成）
let canvasTransform = null;
document.addEventListener('DOMContentLoaded', () => {
    if (canvas) {
        canvasTransform = new CanvasTransform(canvas, null);
        window.canvasTransform = canvasTransform;
    }
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

// 解码模式选择
const decodeModeSelect = document.getElementById('decode-mode');
let currentDecodeMode = 'frontend'; // 默认前端解码
let zeroCopyRenderer = null; // 零拷贝渲染器实例
let wgpuActive = false; // wgpu 渲染状态

// 监听解码模式变更
decodeModeSelect?.addEventListener('change', (e) => {
    currentDecodeMode = e.target.value;
    const modeNames = {
        'frontend': '前端硬解 (WebCodecs)',
        'backend': '后端硬解 (共享内存)',
        'zerocopy': '零拷贝 NV12 (WebGL)',
        'wgpu': 'wgpu GPU 渲染'
    };
    console.log(`🎬 解码模式切换为: ${modeNames[currentDecodeMode]}`);
});

startBtn.addEventListener('click', async () => {
    const url = rtspInput.value.trim();
    if (!url) {
        showStatus('❌ 请输入 RTSP 地址');
        return;
    }
    
    try {
        startBtn.disabled = true;
        const modeNames = {
            'frontend': '前端 WebCodecs',
            'backend': '后端共享内存',
            'zerocopy': '零拷贝 NV12',
            'wgpu': 'wgpu GPU',
            'native': '原生窗口 wgpu'
        };
        showStatus(`🚀 正在启动 (${modeNames[currentDecodeMode]})...`);
        
        if (currentDecodeMode === 'native') {
            // 原生窗口 wgpu 渲染模式 - 会弹出独立窗口
            console.log('🖥️ 启动原生窗口 wgpu 渲染模式');
            
            // 启动 RTSP 流 + 原生窗口渲染
            const result = await invoke('start_rtsp_stream_native', { url });
            console.log(result);
            
            wgpuActive = true;
            showStatus('✅ 监控已启动 (原生窗口)');
            
        } else if (currentDecodeMode === 'wgpu') {
            // wgpu GPU 渲染模式
            console.log('🎮 启动 wgpu GPU 渲染模式');
            
            // 隐藏 canvas，让 wgpu 渲染可见
            canvas.style.display = 'none';
            document.body.classList.add('wgpu-mode');
            
            // 启动 wgpu 渲染器
            await invoke('start_wgpu_render');
            
            // 启动 RTSP 流（后端硬解 + wgpu 渲染）
            const result = await invoke('start_rtsp_stream_wgpu', { url });
            console.log(result);
            
            wgpuActive = true;
            showStatus('✅ 监控已启动 (wgpu GPU 渲染)');
            
        } else if (currentDecodeMode === 'zerocopy') {
            // 零拷贝 NV12 渲染模式
            if (!zeroCopyRenderer) {
                zeroCopyRenderer = new ZeroCopyRenderer(canvas);
            }
            await zeroCopyRenderer.start(url, 'qsv', 3840, 2160);
            showStatus('✅ 监控已启动 (零拷贝 NV12 模式)');
            
        } else if (currentDecodeMode === 'backend') {
            // 后端硬件解码 + Channel 事件驱动模式
            const onFrame = new Channel();
            
            // 设置帧就绪信号处理
            renderer.startHardwareDecodeMode(onFrame);
            
            const result = await invoke('start_rtsp_stream', { 
                url,
                hardwareDecode: true,
                onFrame  // 传入 Channel
            });
            console.log(result);
            
            showStatus('✅ 监控已启动 (后端硬解)');
            
        } else {
            // 前端 WebCodecs 解码模式
            const onData = new Channel();
            onData.onmessage = (message) => {
                if (message.type === 'video_config') {
                    const { codec, width, height } = message;
                    console.log(`🎬 [Video Config] ${codec.toUpperCase()} ${width}x${height}`);
                    renderer.initDecoder(codec, width, height);
                    syncOverlayToCanvas(width, height);
                } else if (message.type === 'audio_config') {
                    const { codec, sample_rate, channels } = message;
                    console.log(`🎵 [Audio Config] ${codec} ${sample_rate}Hz ${channels}ch`);
                    renderer.initAudioDecoder(codec, sample_rate, channels);
                } else if (message.type === 'video') {
                    renderer.handleVideoChunk(message.data);
                } else if (message.type === 'audio') {
                    renderer.handleAudioChunk(message.data);
                }
            };
            
            const result = await invoke('start_rtsp_stream', { 
                url, 
                onData,
                hardwareDecode: false
            });
            console.log(result);
            
            renderer.start();
            showStatus('✅ 监控已启动 (前端 WebCodecs)');
        }
        
        try {
            await invoke('add_rtsp_history', { url });
            console.log('✅ 已保存到历史记录');
        } catch (e) {
            console.warn('保存历史记录失败:', e);
        }
        
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        rtspInput.disabled = true;
        decodeModeSelect.disabled = true;
    } catch (err) {
        console.error('启动失败:', err);
        showStatus('❌ 启动失败: ' + err);
        startBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', async () => {
    if (frameDetector.isDetecting) {
        await frameDetector.stopDetector();
    }
    
    // 停止对应的渲染器
    if (wgpuActive && currentDecodeMode === 'native') {
        // 原生窗口模式
        await invoke('stop_rtsp_stream_native');
        wgpuActive = false;
    } else if (wgpuActive && currentDecodeMode === 'wgpu') {
        await invoke('stop_wgpu_render');
        await invoke('stop_rtsp_stream');
        wgpuActive = false;
        
        // 恢复 canvas 显示
        canvas.style.display = '';
        document.body.classList.remove('wgpu-mode');
    } else if (zeroCopyRenderer && currentDecodeMode === 'zerocopy') {
        await zeroCopyRenderer.stop();
    } else {
        renderer.stop();
    }
    
    stopBtn.classList.add('hidden');
    startBtn.classList.remove('hidden');
    startBtn.disabled = false;
    rtspInput.disabled = false;
    decodeModeSelect.disabled = false;
    showStatus('⏹ 监控已停止');
});

// 暴露到全局方便调试
window.renderer = renderer;

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

console.log('WebGL renderer initialized');
console.log('🔍 缩放功能: 滚轮缩放, 拖拽平移, 双击重置');

// ==================== LLM 视频推理 ====================

// LLM 推理 UI 元素
const llmOutput = document.getElementById('llm-output');
const llmOutputContainer = document.getElementById('llm-output-container');
const startLlmBtn = document.getElementById('start-llm-btn');
const stopLlmBtn = document.getElementById('stop-llm-btn');
const llmModelInput = document.getElementById('llm-model');
const llmApiUrlInput = document.getElementById('llm-api-url');
const llmSystemPromptInput = document.getElementById('llm-system-prompt');
const llmUserPromptInput = document.getElementById('llm-user-prompt');
const llmFrameIntervalInput = document.getElementById('llm-frame-interval');
const llmTemperatureInput = document.getElementById('llm-temperature');
const llmConfigToggle = document.getElementById('llm-config-toggle');
const llmConfigPanel = document.getElementById('llm-config-panel');
const llmClearOutput = document.getElementById('llm-clear-output');

// 当前 RTSP URL (用于 LLM 推理)
let currentRtspUrl = '';

// 配置面板切换
llmConfigToggle?.addEventListener('click', () => {
    llmConfigPanel?.classList.toggle('hidden');
    llmConfigToggle.textContent = llmConfigPanel?.classList.contains('hidden') ? '配置' : '收起';
});

// 清空输出
llmClearOutput?.addEventListener('click', () => {
    if (llmOutput) llmOutput.textContent = '';
});

// 追加输出文本
function appendLlmOutput(text) {
    if (llmOutput) {
        llmOutput.textContent += text;
        llmOutput.scrollTop = llmOutput.scrollHeight;
    }
}

// 设置 LLM 回调
llmManager.onFrameInfo = (chunk) => {
    appendLlmOutput(`\n\n━━━ 📷 帧 #${chunk.frame_id} (${new Date().toLocaleTimeString()}) ━━━\n`);
};

llmManager.onChunk = (chunk, fullText) => {
    // 流式追加文本
    appendLlmOutput(chunk.content);
};

llmManager.onEnd = (chunk, fullText) => {
    appendLlmOutput('\n');
    console.log(`✅ 帧 #${chunk.frame_id} 分析完成`);
};

llmManager.onError = (chunk) => {
    appendLlmOutput(`\n❌ 错误: ${chunk.content}\n`);
    showStatus('❌ LLM 推理错误: ' + chunk.content);
};

// 启动 LLM 推理
startLlmBtn?.addEventListener('click', async () => {
    // 获取当前 RTSP URL
    const url = rtspInput.value.trim();
    if (!url) {
        showStatus('❌ 请先输入 RTSP 地址');
        return;
    }
    currentRtspUrl = url;

    // 更新配置
    llmManager.setConfig({
        model: llmModelInput?.value || 'llava',
        api_url: llmApiUrlInput?.value || 'http://localhost:11434/api/chat',
        system_prompt: llmSystemPromptInput?.value || '',
        user_prompt: llmUserPromptInput?.value || '描述这张监控画面中的场景和活动',
        frame_interval: parseFloat(llmFrameIntervalInput?.value) || 2.0,
        temperature: parseFloat(llmTemperatureInput?.value) || 0.7
    });

    try {
        startLlmBtn.disabled = true;
        startLlmBtn.innerHTML = '<span>🔄</span> 启动中...';

        // 显示输出容器
        llmOutputContainer?.classList.remove('hidden');
        if (llmOutput) llmOutput.textContent = '🚀 启动 LLM 推理...\n';

        await llmManager.start(currentRtspUrl);

        startLlmBtn.classList.add('hidden');
        stopLlmBtn.classList.remove('hidden');
        startLlmBtn.disabled = false;
        startLlmBtn.innerHTML = '<span>🤖</span> 开启LLM推理';
        showStatus('🤖 LLM 推理已启动');
        
        appendLlmOutput('✅ 已连接，等待视频帧...\n');
    } catch (e) {
        console.error('启动 LLM 推理失败:', e);
        showStatus('❌ LLM 推理启动失败: ' + e);
        startLlmBtn.disabled = false;
        startLlmBtn.innerHTML = '<span>🤖</span> 开启LLM推理';
    }
});

// 停止 LLM 推理
stopLlmBtn?.addEventListener('click', async () => {
    try {
        await llmManager.stop();

        stopLlmBtn.classList.add('hidden');
        startLlmBtn.classList.remove('hidden');
        startLlmBtn.disabled = false;
        startLlmBtn.innerHTML = '<span>🤖</span> 开启LLM推理';
        showStatus('⏹ LLM 推理已停止');
        appendLlmOutput('\n\n🛑 LLM 推理已停止\n');
    } catch (e) {
        console.error('停止 LLM 推理失败:', e);
        showStatus('❌ 停止失败: ' + e);
    }
});

// 暴露到全局
window.llmManager = llmManager;

console.log('🤖 LLM 推理模块已加载');
console.log('使用方法:');
console.log('  启动推理: llmManager.start("rtsp://...")');
console.log('  停止推理: llmManager.stop()');
