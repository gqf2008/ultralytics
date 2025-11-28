import { invoke, Channel } from '@tauri-apps/api/core';
import { FrameDetector } from './detector.js';
import { llmManager, defaultLlmConfig } from './llmInference.js';

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
    
    // 硬件解码模式 - 从后端轮询 RGBA 帧
    startHardwareDecodeMode() {
        this.isRunning = true;
        this.hwDecodeRunning = true;
        console.log('🎬 启动硬件解码渲染模式');
        this.pollFrames();
    }
    
    async pollFrames() {
        if (!this.hwDecodeRunning) return;
        
        try {
            const [frameData, width, height] = await invoke('get_latest_frame');
            
            if (frameData && frameData.length > 0) {
                // 更新 canvas 尺寸
                if (this.canvas.width !== width || this.canvas.height !== height) {
                    this.canvas.width = width;
                    this.canvas.height = height;
                    this.videoWidth = width;
                    this.videoHeight = height;
                    console.log(`📐 Canvas 尺寸更新: ${width}x${height}`);
                    
                    // 同步 overlay
                    if (typeof syncOverlayToCanvas === 'function') {
                        syncOverlayToCanvas(width, height);
                    }
                }
                
                // 创建 ImageData 并绘制
                const imageData = new ImageData(
                    new Uint8ClampedArray(frameData),
                    width,
                    height
                );
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
                
                // 标记帧已处理
                await invoke('mark_frame_processed');
            }
        } catch (e) {
            // 没有帧可用，静默忽略
            if (!e.toString().includes('No frame available')) {
                console.warn('获取帧失败:', e);
            }
        }
        
        // 继续轮询（约 60fps）
        if (this.hwDecodeRunning) {
            requestAnimationFrame(() => this.pollFrames());
        }
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
    
    if (result.boxes && result.boxes.length > 0) {
        console.log(`🎯 绘制 ${result.boxes.length} 个检测框`);
    }
    
    drawDetections(result.boxes);
    
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
        renderBoxesImmediate();
    }
}

// 同步 overlay canvas 尺寸 (窗口 resize 时)
function syncOverlaySize() {
    const mainCanvas = document.getElementById('canvas');
    if (mainCanvas && mainCanvas.width > 0 && mainCanvas.height > 0) {
        if (detectionOverlay.width !== mainCanvas.width || detectionOverlay.height !== mainCanvas.height) {
            syncOverlayToCanvas(mainCanvas.width, mainCanvas.height);
        }
    }
}
window.addEventListener('resize', syncOverlaySize);

// 立即渲染检测框
function renderBoxesImmediate() {
    const ctx = detectionCtx;
    const w = overlayWidth;
    const h = overlayHeight;
    
    ctx.clearRect(0, 0, w, h);
    
    if (cachedBoxes.length === 0) return;
    
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.font = 'bold 16px monospace';
    
    cachedBoxes.forEach(box => {
        const x1 = box.x1 * w;
        const y1 = box.y1 * h;
        const x2 = box.x2 * w;
        const y2 = box.y2 * h;
        const bw = x2 - x1;
        const bh = y2 - y1;
        
        ctx.strokeRect(x1, y1, bw, bh);
        
        const label = `${box.class_name} ${(box.confidence * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.fillRect(x1, y1 - 22, textWidth + 8, 22);
        
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

// 监听解码模式变更
decodeModeSelect?.addEventListener('change', (e) => {
    currentDecodeMode = e.target.value;
    console.log(`🎬 解码模式切换为: ${currentDecodeMode === 'frontend' ? '前端硬解 (WebCodecs)' : '后端硬解 (QSV)'}`);
});

startBtn.addEventListener('click', async () => {
    const url = rtspInput.value.trim();
    if (!url) {
        showStatus('❌ 请输入 RTSP 地址');
        return;
    }
    
    const useBackendDecode = currentDecodeMode === 'backend';
    
    try {
        startBtn.disabled = true;
        showStatus(`🚀 正在启动 (${useBackendDecode ? '后端 QSV 硬解' : '前端 WebCodecs 硬解'})...`);
        
        if (useBackendDecode) {
            // 后端 QSV 硬件解码模式
            const result = await invoke('start_rtsp_stream', { 
                url,
                hardwareDecode: true
            });
            console.log(result);
            
            // 启动帧轮询渲染
            renderer.startHardwareDecodeMode();
            showStatus('✅ 监控已启动 (后端 QSV 硬解)');
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
            showStatus('✅ 监控已启动 (前端 WebCodecs 硬解)');
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
    
    renderer.stop();
    stopBtn.classList.add('hidden');
    startBtn.classList.remove('hidden');
    startBtn.disabled = false;
    rtspInput.disabled = false;
    decodeModeSelect.disabled = false;
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
        const [models, devices] = await Promise.all([
            invoke('get_available_models'),
            invoke('get_available_devices')
        ]);
        
        modelSelect.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            modelSelect.appendChild(option);
        });
        
        const defaultModel = models.find(m => m.name === 'yolov8n.onnx');
        if (defaultModel) {
            modelSelect.value = defaultModel.path;
        }
        
        console.log(`📦 已加载 ${models.length} 个模型`);
        
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

loadModelsAndDevices();

startDetectorBtn.addEventListener('click', async () => {
    try {
        startDetectorBtn.disabled = true;
        startDetectorBtn.innerHTML = '<span>🔄</span> 加载中...';
        
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
            if (result.input_width) {
                console.log(`📐 检测输入尺寸: ${result.input_width}x${result.input_width}`);
            }
            
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
console.log('使用方法:');
console.log('  启动检测: frameDetector.startDetector("yolov8n", "bytetrack")');
console.log('  停止检测: frameDetector.stopDetector()');
console.log('  调整FPS:  frameDetector.setDetectionFps(15)');
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
