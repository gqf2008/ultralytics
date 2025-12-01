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

// ==================== 流信息管理器 ====================

class StreamInfoManager {
    constructor() {
        this.panel = document.getElementById('stream-info-panel');
        this.toggleBtn = document.getElementById('info-toggle-btn');
        this.header = document.getElementById('info-panel-header');
        
        // 信息显示元素
        this.elements = {
            protocol: document.getElementById('info-protocol'),
            backend: document.getElementById('info-backend'),
            videoCodec: document.getElementById('info-video-codec'),
            resolution: document.getElementById('info-resolution'),
            fps: document.getElementById('info-fps'),
            videoBitrate: document.getElementById('info-video-bitrate'),
            audioCodec: document.getElementById('info-audio-codec'),
            sampleRate: document.getElementById('info-sample-rate'),
            channels: document.getElementById('info-channels'),
            audioBitrate: document.getElementById('info-audio-bitrate'),
            decodeFps: document.getElementById('info-decode-fps'),
            latency: document.getElementById('info-latency'),
            packets: document.getElementById('info-packets'),
            bytes: document.getElementById('info-bytes'),
            keyframes: document.getElementById('info-keyframes'),
            runtime: document.getElementById('info-runtime'),
            // 解码状态元素
            decoderState: document.getElementById('info-decoder-state'),
            waitingKeyframe: document.getElementById('info-waiting-keyframe'),
            decodeQueue: document.getElementById('info-decode-queue'),
            decodedFrames: document.getElementById('info-decoded-frames'),
            skippedFrames: document.getElementById('info-skipped-frames'),
            decodeErrors: document.getElementById('info-decode-errors'),
            lastError: document.getElementById('info-last-error'),
            lastErrorRow: document.getElementById('info-last-error-row'),
        };
        
        // 统计数据
        this.stats = {
            packets: 0,
            bytes: 0,
            keyframes: 0,
            startTime: 0,
            decodeFps: 0,
            latency: 0,
        };
        
        // 解码统计
        this.decodeStats = {
            state: 'idle',
            waitingKeyframe: false,
            queueSize: 0,
            decodedFrames: 0,
            skippedFrames: 0,
            errors: 0,
            lastError: '',
        };
        
        this.runtimeTimer = null;
        
        // 拖动状态
        this.isDragging = false;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.panelX = 0;
        this.panelY = 0;
        
        // 面板可见性状态 (由开关控制，默认关闭)
        this.isEnabled = false;
        
        this.setupEventListeners();
        
        // 初始化时隐藏面板 (使用 hidden class，与 setEnabled 逻辑一致)
        if (this.panel) {
            this.panel.classList.add('hidden');
        }
    }
    
    setupEventListeners() {
        // 折叠/展开面板 (点击折叠按钮)
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.panel?.classList.toggle('collapsed');
            });
        }
        
        // 拖动功能
        if (this.header) {
            this.header.addEventListener('mousedown', (e) => this.startDrag(e));
        }
        document.addEventListener('mousemove', (e) => this.drag(e));
        document.addEventListener('mouseup', () => this.endDrag());
    }
    
    // 开始拖动
    startDrag(e) {
        // 忽略折叠按钮点击
        if (e.target.closest('#info-toggle-btn')) return;
        
        this.isDragging = true;
        
        const rect = this.panel.getBoundingClientRect();
        this.dragOffsetX = e.clientX - rect.left;
        this.dragOffsetY = e.clientY - rect.top;
        
        this.panel.style.transition = 'none';
        this.panel.style.cursor = 'grabbing';
    }
    
    // 拖动中
    drag(e) {
        if (!this.isDragging || !this.panel) return;
        
        e.preventDefault();
        
        const newX = e.clientX - this.dragOffsetX;
        const newY = e.clientY - this.dragOffsetY;
        
        // 边界限制
        const maxX = window.innerWidth - this.panel.offsetWidth;
        const maxY = window.innerHeight - this.panel.offsetHeight;
        
        this.panelX = Math.max(0, Math.min(newX, maxX));
        this.panelY = Math.max(0, Math.min(newY, maxY));
        
        // 使用 left/top 定位 (替代默认的 right/top)
        this.panel.style.right = 'auto';
        this.panel.style.left = this.panelX + 'px';
        this.panel.style.top = this.panelY + 'px';
    }
    
    // 结束拖动
    endDrag() {
        if (this.isDragging) {
            this.isDragging = false;
            if (this.panel) {
                this.panel.style.transition = 'all 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
                this.panel.style.cursor = '';
            }
        }
    }
    
    // 设置面板启用状态 (由开关控制)
    setEnabled(enabled) {
        this.isEnabled = enabled;
        if (!enabled) {
            this.panel?.classList.add('hidden');
        } else {
            // 启用时显示面板
            this.panel?.classList.remove('hidden');
        }
    }
    
    // 显示面板 (仅当开关启用时)
    show() {
        if (this.isEnabled) {
            this.panel?.classList.remove('hidden');
        }
    }
    
    // 隐藏面板
    hide() {
        this.panel?.classList.add('hidden');
        this.stopRuntimeTimer();
    }
    
    // 显示错误信息
    showError(errorMsg) {
        // 显示面板 (如果启用)
        this.show();
        
        // 设置协议为错误状态
        if (this.elements.protocol) {
            this.elements.protocol.textContent = '❌ 错误';
            this.elements.protocol.className = 'value info-badge red';
        }
        if (this.elements.backend) {
            this.elements.backend.textContent = errorMsg;
            this.elements.backend.style.color = '#ff6b6b';
        }
    }
    
    // 更新解码状态
    updateDecodeStatus(status) {
        this.decodeStats = { ...this.decodeStats, ...status };
        
        // 更新状态显示
        if (this.elements.decoderState) {
            const state = status.state || this.decodeStats.state;
            this.elements.decoderState.textContent = state;
            this.elements.decoderState.className = 'value info-badge';
            if (state === 'configured') {
                this.elements.decoderState.classList.add('green');
            } else if (state === 'closed' || state === 'error') {
                this.elements.decoderState.classList.add('red');
            } else {
                this.elements.decoderState.classList.add('orange');
            }
        }
        
        // 等待关键帧状态
        if (this.elements.waitingKeyframe && status.waitingKeyframe !== undefined) {
            const waiting = status.waitingKeyframe;
            this.elements.waitingKeyframe.textContent = waiting ? '是' : '否';
            this.elements.waitingKeyframe.className = 'value info-badge';
            if (waiting) {
                this.elements.waitingKeyframe.classList.add('orange');
            } else {
                this.elements.waitingKeyframe.classList.add('green');
            }
        }
        
        if (this.elements.decodeQueue && status.queueSize !== undefined) {
            this.elements.decodeQueue.textContent = status.queueSize;
            // 队列过大时变红
            this.elements.decodeQueue.style.color = status.queueSize > 5 ? '#ff6b6b' : '';
        }
        
        if (this.elements.decodedFrames && status.decodedFrames !== undefined) {
            this.elements.decodedFrames.textContent = status.decodedFrames;
        }
        
        if (this.elements.skippedFrames && status.skippedFrames !== undefined) {
            this.elements.skippedFrames.textContent = status.skippedFrames;
            this.elements.skippedFrames.style.color = status.skippedFrames > 0 ? '#f59e0b' : '';
        }
        
        if (this.elements.decodeErrors && status.errors !== undefined) {
            this.elements.decodeErrors.textContent = status.errors;
            this.elements.decodeErrors.style.color = status.errors > 0 ? '#ff6b6b' : '';
        }
        
        // 显示最后错误
        if (status.lastError && this.elements.lastError && this.elements.lastErrorRow) {
            this.elements.lastError.textContent = status.lastError;
            this.elements.lastErrorRow.style.display = 'flex';
        }
    }
    
    // 重置解码状态
    resetDecodeStatus() {
        this.decodeStats = {
            state: 'idle',
            waitingKeyframe: false,
            queueSize: 0,
            decodedFrames: 0,
            skippedFrames: 0,
            errors: 0,
            lastError: '',
        };
        this.updateDecodeStatus(this.decodeStats);
        if (this.elements.lastErrorRow) {
            this.elements.lastErrorRow.style.display = 'none';
        }
    }
    
    // 重置统计数据
    reset() {
        this.stats = {
            packets: 0,
            bytes: 0,
            keyframes: 0,
            startTime: Date.now(),
        };
        this.updateStats();
        this.startRuntimeTimer();
    }
    
    // 更新流信息 (从后端接收)
    updateStreamInfo(info) {
        console.log('📊 更新流信息:', info);
        
        // 连接信息
        if (this.elements.protocol) {
            this.elements.protocol.textContent = info.protocol || '-';
            // 根据协议设置徽章颜色
            this.elements.protocol.className = 'value info-badge';
            if (info.protocol === 'RTSP') {
                this.elements.protocol.classList.add('green');
            } else if (info.protocol === 'HTTP-FLV') {
                this.elements.protocol.classList.add('orange');
            }
        }
        if (this.elements.backend) {
            this.elements.backend.textContent = info.backend || '-';
        }
        
        // 视频信息
        if (this.elements.videoCodec) {
            this.elements.videoCodec.textContent = info.video_codec || '-';
        }
        if (this.elements.resolution && info.video_width && info.video_height) {
            this.elements.resolution.textContent = `${info.video_width}×${info.video_height}`;
        }
        if (this.elements.fps && info.video_fps) {
            this.elements.fps.textContent = `${info.video_fps.toFixed(2)} fps`;
        }
        if (this.elements.videoBitrate) {
            this.elements.videoBitrate.textContent = this.formatBitrate(info.video_bitrate);
        }
        
        // 音频信息
        if (this.elements.audioCodec) {
            this.elements.audioCodec.textContent = info.audio_codec || '-';
        }
        if (this.elements.sampleRate && info.audio_sample_rate) {
            this.elements.sampleRate.textContent = `${info.audio_sample_rate} Hz`;
        }
        if (this.elements.channels) {
            if (info.audio_channels === 1) {
                this.elements.channels.textContent = '单声道';
            } else if (info.audio_channels === 2) {
                this.elements.channels.textContent = '立体声';
            } else if (info.audio_channels > 0) {
                this.elements.channels.textContent = `${info.audio_channels} 声道`;
            } else {
                this.elements.channels.textContent = '-';
            }
        }
        if (this.elements.audioBitrate) {
            this.elements.audioBitrate.textContent = this.formatBitrate(info.audio_bitrate);
        }
        
        // 设置开始时间
        if (info.start_time) {
            this.stats.startTime = info.start_time;
        } else {
            this.stats.startTime = Date.now();
        }
        
        this.startRuntimeTimer();
    }
    
    // 更新音频信息 (当收到 audio_config 消息时)
    updateAudioInfo(codec, sampleRate, channels) {
        // 转换 codec 显示名称
        let codecName;
        if (codec === 'mp4a.40.2') {
            codecName = 'AAC';
        } else if (codec === 'pcm_alaw') {
            codecName = 'PCM A-Law';
        } else if (codec === 'pcm_mulaw') {
            codecName = 'PCM μ-Law';
        } else if (codec === 'mp3') {
            codecName = 'MP3';
        } else {
            codecName = codec;
        }
        
        if (this.elements.audioCodec) {
            this.elements.audioCodec.textContent = codecName;
        }
        if (this.elements.sampleRate) {
            this.elements.sampleRate.textContent = `${sampleRate} Hz`;
        }
        if (this.elements.channels) {
            if (channels === 1) {
                this.elements.channels.textContent = '单声道';
            } else if (channels === 2) {
                this.elements.channels.textContent = '立体声';
            } else if (channels > 0) {
                this.elements.channels.textContent = `${channels} 声道`;
            } else {
                this.elements.channels.textContent = '-';
            }
        }
        
        console.log(`📊 流信息面板音频更新: ${codecName} ${sampleRate}Hz ${channels}ch`);
    }
    
    // 更新数据包统计 (每个包调用)
    addPacket(size, isKeyframe) {
        this.stats.packets++;
        this.stats.bytes += size;
        if (isKeyframe) {
            this.stats.keyframes++;
        }
        
        // 每 100 包更新一次 UI (避免频繁更新)
        if (this.stats.packets % 100 === 0 || this.stats.packets <= 10) {
            this.updateStats();
        }
    }
    
    // 更新统计显示
    updateStats() {
        if (this.elements.packets) {
            this.elements.packets.textContent = `${this.stats.packets.toLocaleString()} 包`;
        }
        if (this.elements.bytes) {
            this.elements.bytes.textContent = this.formatBytes(this.stats.bytes);
        }
        if (this.elements.keyframes) {
            this.elements.keyframes.textContent = this.stats.keyframes.toLocaleString();
        }
        if (this.elements.decodeFps) {
            this.elements.decodeFps.textContent = this.stats.decodeFps;
        }
        if (this.elements.latency) {
            this.elements.latency.textContent = `${this.stats.latency} ms`;
        }
    }
    
    // 更新 FPS 和 Latency (由渲染器调用)
    updatePerformance(fps, latency) {
        this.stats.decodeFps = fps;
        this.stats.latency = latency;
        if (this.elements.decodeFps) {
            this.elements.decodeFps.textContent = fps;
        }
        if (this.elements.latency) {
            this.elements.latency.textContent = `${latency} ms`;
        }
    }
    
    // 启动运行时间计时器
    startRuntimeTimer() {
        this.stopRuntimeTimer();
        this.runtimeTimer = setInterval(() => {
            const elapsed = Date.now() - this.stats.startTime;
            if (this.elements.runtime) {
                this.elements.runtime.textContent = this.formatDuration(elapsed);
            }
        }, 1000);
    }
    
    // 停止运行时间计时器
    stopRuntimeTimer() {
        if (this.runtimeTimer) {
            clearInterval(this.runtimeTimer);
            this.runtimeTimer = null;
        }
    }
    
    // 格式化比特率
    formatBitrate(bps) {
        if (!bps || bps === 0) return '-';
        if (bps >= 1000000) {
            return `${(bps / 1000000).toFixed(2)} Mbps`;
        } else if (bps >= 1000) {
            return `${(bps / 1000).toFixed(0)} kbps`;
        }
        return `${bps} bps`;
    }
    
    // 格式化字节数
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${units[i]}`;
    }
    
    // 格式化持续时间
    formatDuration(ms) {
        const seconds = Math.floor(ms / 1000);
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
}

// 全局流信息管理器实例
let streamInfoManager = null;

// ==================== WebGL 视频渲染器 ====================

class WebGLVideoRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        // 使用默认设置，让浏览器自动处理色彩空间
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
        this.audioConfigured = false;
        
        // 音频缓冲管理（简化版，避免发抖）
        this.audioBufferAhead = 0.1;    // 预缓冲 100ms
        this.audioMaxBuffer = 0.5;      // 最大缓冲 500ms
        
        // 音视频同步：视频参考音频
        this.audioClockBase = 0;        // 第一个音频包的 PTS (毫秒)
        this.audioClockStart = 0;       // 第一个音频包到达时的 AudioContext.currentTime
        this.audioClockReady = false;   // 音频时钟是否就绪
        
        // 实时性优化：帧队列管理
        this.pendingFrames = [];
        this.maxPendingFrames = 2;
        this.lastFrameTime = 0;
        this.lastImageBitmap = null;  // 保存最后一帧的副本，用于窗口 resize 时重绘
        
        // 编解码器信息
        this.currentCodec = null;
        this.extradata = null;  // SPS/PPS for H.264, VPS/SPS/PPS for HEVC
        this.decoderConfigured = false;
        this.decoderHasDescription = false;  // 是否使用了 AVCC description
        this.waitingForKeyframe = true;
        this.videoChunkCount = 0;  // 帧计数器
        
        // 调色滤镜 (默认值修复 WebView2 色彩偏白)
        this.colorFilter = 'brightness(1) contrast(1.2) saturate(1.3)';
        
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.initAudioContext();
    }
    
    initAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('🔊 Audio Context initialized:', this.audioContext.sampleRate, 'Hz');
        
        // ========== 音频滤波器链（降噪） ==========
        // 高通滤波器：去除低频嗡嗡声（如电源噪声 50/60Hz）
        this.highpassFilter = this.audioContext.createBiquadFilter();
        this.highpassFilter.type = 'highpass';
        this.highpassFilter.frequency.value = 80;  // 80Hz 以下截止
        this.highpassFilter.Q.value = 0.7;
        
        // 第二个高通：更陡峭的斜率
        this.highpassFilter2 = this.audioContext.createBiquadFilter();
        this.highpassFilter2.type = 'highpass';
        this.highpassFilter2.frequency.value = 80;
        this.highpassFilter2.Q.value = 0.7;
        
        // 低通滤波器：去除高频噪声（如电流噪声、嘶嘶声）
        this.lowpassFilter = this.audioContext.createBiquadFilter();
        this.lowpassFilter.type = 'lowpass';
        this.lowpassFilter.frequency.value = 12000;  // 12kHz 以上截止（保留语音）
        this.lowpassFilter.Q.value = 0.7;
        
        // 第二个低通：更陡峭的斜率
        this.lowpassFilter2 = this.audioContext.createBiquadFilter();
        this.lowpassFilter2.type = 'lowpass';
        this.lowpassFilter2.frequency.value = 12000;
        this.lowpassFilter2.Q.value = 0.7;
        
        // 陷波滤波器：消除电源工频干扰 (50Hz)
        this.notchFilter = this.audioContext.createBiquadFilter();
        this.notchFilter.type = 'notch';
        this.notchFilter.frequency.value = 50;  // 50Hz 电源干扰
        this.notchFilter.Q.value = 30;  // 窄带
        
        // 第二个陷波：消除谐波 (100Hz/150Hz)
        this.notchFilter2 = this.audioContext.createBiquadFilter();
        this.notchFilter2.type = 'notch';
        this.notchFilter2.frequency.value = 100;  // 50Hz的二次谐波
        this.notchFilter2.Q.value = 30;
        
        // 动态压缩器：减少突发噪声，平衡音量（更激进的设置）
        this.compressor = this.audioContext.createDynamicsCompressor();
        this.compressor.threshold.value = -30;  // 更低的阈值
        this.compressor.knee.value = 10;
        this.compressor.ratio.value = 8;  // 更高的压缩比
        this.compressor.attack.value = 0.001;  // 更快的响应
        this.compressor.release.value = 0.1;
        
        // 增益节点：音量控制
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = 1.5;  // 默认音量 1.5x（与 UI 滑块一致）
        
        // 连接滤波器链（双重滤波，更陡峭的斜率）：
        // source -> hp1 -> hp2 -> notch1 -> notch2 -> lp1 -> lp2 -> compressor -> gain -> destination
        this.highpassFilter.connect(this.highpassFilter2);
        this.highpassFilter2.connect(this.notchFilter);
        this.notchFilter.connect(this.notchFilter2);
        this.notchFilter2.connect(this.lowpassFilter);
        this.lowpassFilter.connect(this.lowpassFilter2);
        this.lowpassFilter2.connect(this.compressor);
        this.compressor.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);
        
        // 保存滤波器链入口，供音频源连接
        this.audioFilterInput = this.highpassFilter;
        // 保存直连节点（用于关闭降噪时）
        this.audioDirectOutput = this.gainNode;
        // 降噪是否启用
        this.noiseReductionEnabled = true;
        
        console.log('🔇 音频降噪滤波器已启用: 双重高通 + 双重陷波 + 双重低通 + 压缩器');
        
        // 音频上下文可能被浏览器挂起，需要用户交互后恢复
        document.addEventListener('click', () => {
            if (this.audioContext && this.audioContext.state === 'suspended') {
                this.audioContext.resume();
                console.log('🔊 Audio Context resumed on user interaction');
            }
        }, { once: true });
    }
    
    // ========== 音频控制方法 ==========
    
    /**
     * 设置音量
     */
    setVolume(gain) {
        this.audioGain = gain;
        if (this.gainNode) {
            this.gainNode.gain.value = gain;
            console.log(`🔊 音量设置: ${gain.toFixed(1)}x`);
        }
    }
    
    /**
     * 启用/禁用降噪
     */
    setNoiseReduction(enabled) {
        this.noiseReductionEnabled = enabled;
        if (enabled) {
            // 启用降噪：音频源连接到滤波器链
            this.audioFilterInput = this.highpassFilter;
        } else {
            // 禁用降噪：音频源直接连接到增益节点
            this.audioFilterInput = this.gainNode;
        }
        console.log(`🔇 降噪${enabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 设置高通滤波器截止频率
     */
    setHighpassFrequency(freq) {
        if (this.highpassFilter) {
            this.highpassFilter.frequency.value = freq;
            this.highpassFilter2.frequency.value = freq;
            console.log(`🔇 高通截止: ${freq}Hz (双重滤波)`);
        }
    }
    
    /**
     * 设置低通滤波器截止频率
     */
    setLowpassFrequency(freq) {
        if (this.lowpassFilter) {
            this.lowpassFilter.frequency.value = freq;
            this.lowpassFilter2.frequency.value = freq;
            console.log(`🔇 低通截止: ${freq}Hz (双重滤波)`);
        }
    }
    
    /**
     * 设置陷波滤波器频率（电源噪声）
     */
    setNotchFrequency(freq) {
        if (this.notchFilter) {
            if (freq > 0) {
                this.notchFilter.frequency.value = freq;
                this.notchFilter.Q.value = 30;
                // 同时设置二次谐波
                this.notchFilter2.frequency.value = freq * 2;
                this.notchFilter2.Q.value = 30;
                console.log(`🔇 陷波滤波: ${freq}Hz + ${freq * 2}Hz (含谐波)`);
            } else {
                this.notchFilter.Q.value = 0.001;
                this.notchFilter2.Q.value = 0.001;
                console.log(`🔇 陷波滤波: 关闭`);
            }
        }
    }
    
    /**
     * 应用降噪预设
     */
    applyNoisePreset(preset) {
        if (!preset) return;
        
        // 设置是否启用滤波器链
        this.setNoiseReduction(preset.enabled);
        
        if (preset.enabled) {
            // 设置高通
            this.setHighpassFrequency(preset.highpass);
            // 设置低通
            this.setLowpassFrequency(preset.lowpass);
            
            // 压缩器设置
            if (preset.compression && this.compressor) {
                this.compressor.threshold.value = -30;
                this.compressor.ratio.value = 8;
            } else if (this.compressor) {
                this.compressor.threshold.value = 0;
                this.compressor.ratio.value = 1;
            }
        }
        
        console.log(`🔇 应用降噪预设: HP=${preset.highpass}Hz LP=${preset.lowpass}Hz 压缩=${preset.compression}`);
    }
    
    /**
     * 重置音频缓冲
     */
    resetAudioBuffer() {
        this.nextAudioTime = this.audioContext ? this.audioContext.currentTime + this.audioBufferAhead : 0;
        this.audioClockBase = 0;
        this.audioClockStart = 0;
        this.audioClockReady = false;
        this._videoSyncBaseSet = false;  // 同时重置视频同步标志
        console.log(`🔄 音频缓冲和时钟重置`);
    }
    
    /**
     * 获取当前音频时钟时间（毫秒）
     * 用于视频帧同步参考
     */
    getAudioClockTime() {
        if (!this.audioClockReady || !this.audioContext) {
            return null;
        }
        // 音频时钟 = 基准PTS + (当前AudioContext时间 - 开始时间) * 1000
        const elapsed = (this.audioContext.currentTime - this.audioClockStart) * 1000;
        return this.audioClockBase + elapsed;
    }
    
    /**
     * 在画布上显示错误信息，并创建可点击的安装按钮
     */
    showErrorMessage(lines, showInstallButton = false) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        this.ctx.fillStyle = '#1a1a1a';
        this.ctx.fillRect(0, 0, w, h);
        
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        const lineHeight = 30;
        const startY = h / 2 - (lines.length * lineHeight) / 2 - 30;
        
        lines.forEach((line, i) => {
            if (i === 0) {
                this.ctx.fillStyle = '#ff4444';
                this.ctx.font = 'bold 24px monospace';
            } else {
                this.ctx.fillStyle = '#cccccc';
                this.ctx.font = '18px monospace';
            }
            this.ctx.fillText(line, w / 2, startY + i * lineHeight);
        });
        
        // 创建安装 HEVC 按钮
        if (showInstallButton) {
            this.createInstallHEVCButton(w / 2, startY + lines.length * lineHeight + 40);
        }
    }
    
    /**
     * 检测当前运行平台
     * @returns {'windows'|'macos'|'linux'}
     */
    detectPlatform() {
        const ua = navigator.userAgent.toLowerCase();
        const platform = navigator.platform.toLowerCase();
        
        if (platform.includes('win') || ua.includes('windows')) {
            return 'windows';
        } else if (platform.includes('mac') || ua.includes('macintosh')) {
            return 'macos';
        } else {
            return 'linux';
        }
    }
    
    /**
     * 创建安装 HEVC 扩展的按钮 (仅 Windows)
     */
    createInstallHEVCButton(x, y) {
        // 移除已存在的按钮
        const existingBtn = document.getElementById('hevc-install-btn');
        if (existingBtn) existingBtn.remove();
        
        const btn = document.createElement('button');
        btn.id = 'hevc-install-btn';
        btn.textContent = '📦 打开 Microsoft Store 安装 HEVC 扩展';
        btn.style.cssText = `
            position: fixed;
            left: ${x}px;
            top: ${y}px;
            transform: translateX(-50%);
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            color: white;
            background: linear-gradient(135deg, #0078d4, #106ebe);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 120, 212, 0.4);
            transition: all 0.2s;
            z-index: 1000;
        `;
        
        btn.onmouseenter = () => {
            btn.style.transform = 'translateX(-50%) scale(1.05)';
            btn.style.boxShadow = '0 6px 16px rgba(0, 120, 212, 0.5)';
        };
        btn.onmouseleave = () => {
            btn.style.transform = 'translateX(-50%) scale(1)';
            btn.style.boxShadow = '0 4px 12px rgba(0, 120, 212, 0.4)';
        };
        
        btn.onclick = () => {
            // 打开 Microsoft Store 的 HEVC 扩展页面
            // ms-windows-store://pdp/?ProductId=9n4wgh0z6vhq - 免费版 (OEM)
            // ms-windows-store://pdp/?ProductId=9nmzlz57r3t7 - 付费版
            const storeUrl = 'ms-windows-store://pdp/?ProductId=9n4wgh0z6vhq';
            console.log('🏪 打开 Microsoft Store:', storeUrl);
            
            // 使用 Tauri 的 shell 打开
            if (window.__TAURI__) {
                import('@tauri-apps/plugin-shell').then(({ open }) => {
                    open(storeUrl).catch(e => {
                        console.error('打开 Store 失败:', e);
                        // 回退到网页版
                        window.open('https://apps.microsoft.com/detail/9n4wgh0z6vhq', '_blank');
                    });
                }).catch(() => {
                    window.open('https://apps.microsoft.com/detail/9n4wgh0z6vhq', '_blank');
                });
            } else {
                window.open('https://apps.microsoft.com/detail/9n4wgh0z6vhq', '_blank');
            }
        };
        
        document.body.appendChild(btn);
    }
    
    resizeCanvas() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        
        console.log(`[Canvas Resize] Window: ${w}x${h}`);
        
        this.canvas.width = w;
        this.canvas.height = h;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        
        // 如果有保存的最后一帧，立即重绘
        if (this.lastImageBitmap) {
            this.redrawLastFrame();
        } else if (!this.decoderConfigured && this.pendingFrames.length === 0) {
            // 没有活跃流时显示等待文字
            this.ctx.fillStyle = '#1a1a1a';
            this.ctx.fillRect(0, 0, w, h);
            
            this.ctx.fillStyle = '#666';
            this.ctx.font = '20px monospace';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText('WAITING FOR STREAM...', w / 2, h / 2);
        }
    }
    
    /**
     * 重绘最后一帧（窗口 resize 时使用）
     */
    redrawLastFrame() {
        if (!this.lastImageBitmap) return;
        
        const bitmap = this.lastImageBitmap;
        
        // 应用色彩校正滤镜
        this.ctx.filter = this.colorFilter || 'contrast(1.2) saturate(1.3)';
        
        // 清空画布
        this.ctx.fillStyle = '#0a0a0f';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 计算等比例缩放后的目标尺寸和位置 (居中显示)
        const canvasAspect = this.canvas.width / this.canvas.height;
        const frameAspect = bitmap.width / bitmap.height;
        
        let drawWidth, drawHeight, drawX, drawY;
        
        if (frameAspect > canvasAspect) {
            drawWidth = this.canvas.width;
            drawHeight = this.canvas.width / frameAspect;
            drawX = 0;
            drawY = (this.canvas.height - drawHeight) / 2;
        } else {
            drawHeight = this.canvas.height;
            drawWidth = this.canvas.height * frameAspect;
            drawX = (this.canvas.width - drawWidth) / 2;
            drawY = 0;
        }
        
        this.ctx.drawImage(bitmap, drawX, drawY, drawWidth, drawHeight);
        
        // 重置滤镜
        this.ctx.filter = 'none';
    }
    
    /**
     * 初始化视频解码器
     * @param {string} codec - 'h264' 或 'hevc'
     * @param {number} width - 视频宽度
     * @param {number} height - 视频高度
     * @param {Uint8Array} extradata - SPS/PPS 数据 (可选)
     */
    initDecoder(codec = 'hevc', width = 3840, height = 2160, extradata = null) {
        if (!('VideoDecoder' in window)) {
            console.error("WebCodecs API not supported");
            return;
        }

        // 关闭之前的解码器
        if (this.decoder && this.decoder.state !== 'closed') {
            this.decoder.close();
        }
        
        this.pendingFrames = [];
        this.currentCodec = codec;
        this.extradata = extradata;
        this.decoderConfigured = false;
        this.waitingForKeyframe = true;  // 将在 configure 成功后根据情况调整
        this.videoChunkCount = 0;
        this.basePts = null;  // 重置基准时间戳
        this.baseTime = null;
        this.decodedFrameCount = 0;  // 解码输出帧计数
        this.skippedFrameCount = 0;  // 跳帧计数
        this.decodeErrorCount = 0;   // 解码错误计数
        this.lastDecodeError = '';   // 最后一次解码错误

        // 重置流信息面板的解码状态
        if (typeof streamInfoManager !== 'undefined' && streamInfoManager) {
            streamInfoManager.resetDecodeStatus();
        }

        this.decoder = new VideoDecoder({
            output: (frame) => {
                this.decodedFrameCount++;
                // 每30帧更新一次状态到面板
                if (this.decodedFrameCount % 30 === 1) {
                    if (typeof streamInfoManager !== 'undefined' && streamInfoManager) {
                        streamInfoManager.updateDecodeStatus({
                            state: this.decoder.state,
                            waitingKeyframe: this.waitingForKeyframe,
                            queueSize: this.decoder.decodeQueueSize,
                            decodedFrames: this.decodedFrameCount,
                            skippedFrames: this.skippedFrameCount,
                            errors: this.decodeErrorCount,
                        });
                    }
                }
                while (this.pendingFrames.length >= this.maxPendingFrames) {
                    const oldFrame = this.pendingFrames.shift();
                    oldFrame.close();
                }
                this.pendingFrames.push(frame);
                this.processLatestFrame();
            },
            error: (e) => {
                this.decodeErrorCount++;
                this.lastDecodeError = e.message;
                console.error("❌ Decoder error:", e.message);
                console.error("   decoder state:", this.decoder?.state);
                // 解码错误时等待下一个关键帧
                this.waitingForKeyframe = true;
                this.basePts = null;  // 重置时间戳基准
                
                // 更新面板显示错误
                if (typeof streamInfoManager !== 'undefined' && streamInfoManager) {
                    streamInfoManager.updateDecodeStatus({
                        state: this.decoder?.state || 'error',
                        waitingKeyframe: this.waitingForKeyframe,
                        errors: this.decodeErrorCount,
                        lastError: e.message,
                    });
                }
            },
        });

        // 构建 codec string
        let codecString;
        // 标准化 codec 名称 (支持完整的 codec string 如 "avc1.640028" 或 "hvc1.1.6.L93.B0")
        const codecLower = codec.toLowerCase();
        const isH264 = codecLower === 'h264' || codecLower.startsWith('avc1') || codecLower.startsWith('avc');
        const isHEVC = codecLower === 'hevc' || codecLower.startsWith('hvc1') || codecLower.startsWith('hev1') || codecLower === 'h265';
        
        if (isH264) {
            // H.264/AVC codec string: avc1.PPCCLL
            // PP=profile_idc, CC=constraint_set flags, LL=level_idc
            // 尝试从 extradata 解析 profile/level (Annex-B SPS)
            let profile = 0x64; // 默认 High Profile (100)
            let constraints = 0x00;
            let level = 0x1f; // 默认 Level 3.1
            
            if (extradata && extradata.length >= 8) {
                // extradata 格式: 00 00 00 01 67 [profile] [constraints] [level] ...
                // 找到 SPS NAL unit (type 7, 即 0x67 & 0x1F = 7)
                for (let i = 0; i < extradata.length - 4; i++) {
                    if (extradata[i] === 0 && extradata[i+1] === 0 && 
                        extradata[i+2] === 0 && extradata[i+3] === 1) {
                        const nalType = extradata[i+4] & 0x1F;
                        if (nalType === 7 && i + 7 < extradata.length) {
                            // SPS: profile_idc, constraint_set_flags, level_idc
                            profile = extradata[i+5];
                            constraints = extradata[i+6];
                            level = extradata[i+7];
                            console.log(`📊 从 SPS 解析: profile=${profile.toString(16)}, constraints=${constraints.toString(16)}, level=${level.toString(16)}`);
                            break;
                        }
                    }
                }
            }
            
            // 构建 codec string
            codecString = `avc1.${profile.toString(16).padStart(2, '0')}${constraints.toString(16).padStart(2, '0')}${level.toString(16).padStart(2, '0')}`;
            console.log(`📊 H.264 codec string: ${codecString}`);
            
        } else if (isHEVC) {
            // HEVC/H.265 codec string
            // hvc1.P.T.Lxx.Cx - P=profile, T=tier, Lxx=level, Cx=constraints
            // 如果分辨率未知，默认使用 Level 5.1 (4K 兼容)
            const effectiveHeight = height > 0 ? height : 1080;
            if (effectiveHeight > 2160) {
                codecString = 'hvc1.1.6.L186.B0'; // Main Profile Level 6.2 (8K)
            } else if (effectiveHeight > 1080) {
                codecString = 'hvc1.1.6.L153.B0'; // Main Profile Level 5.1 (4K)
            } else {
                codecString = 'hvc1.1.6.L153.B0'; // Main Profile Level 5.1 (默认兼容 4K)
            }
        } else {
            console.error(`不支持的编解码器: ${codec}`);
            return;
        }
        
        const config = {
            codec: codecString,
            // 如果分辨率未知，使用 1920x1080 作为默认值
            // WebCodecs 解码器会从视频流中自动检测实际分辨率
            codedWidth: width > 0 ? width : 1920,
            codedHeight: height > 0 ? height : 1080,
            optimizeForLatency: true,  // 优化延迟
        };
        
        // 如果有 extradata，添加到配置中
        if (extradata && extradata.length > 0) {
            if (isH264) {
                // H.264: 检查 extradata 格式：Annex-B (以 00 00 00 01 开头) 或 AVCC (version byte = 1)
                const isAnnexB = extradata[0] === 0 && extradata[1] === 0 && 
                                ((extradata[2] === 0 && extradata[3] === 1) || extradata[2] === 1);
                const isAVCC = extradata[0] === 1;  // AVCC extradata 版本字节 = 1
                
                if (isAnnexB) {
                    // 将 Annex-B 格式的 extradata 转换为 AVCC 格式
                    const avccData = this.annexBToAvcc(extradata);
                    if (avccData) {
                        config.description = avccData;
                        console.log(`📦 H.264 AVCC description (from Annex-B): ${avccData.length} bytes`);
                    }
                } else if (isAVCC) {
                    // extradata 已经是 AVCC 格式，直接使用
                    config.description = extradata;
                    console.log(`📦 H.264 AVCC description (native): ${extradata.length} bytes`);
                }
            } else if (isHEVC) {
                // HEVC: 检查 extradata 格式：Annex-B (以 00 00 00 01 开头) 或 HVCC (version byte = 1)
                const isAnnexB = extradata[0] === 0 && extradata[1] === 0 && 
                                ((extradata[2] === 0 && extradata[3] === 1) || extradata[2] === 1);
                const isHVCC = extradata[0] === 1;  // HVCC extradata 版本字节 = 1
                
                if (isAnnexB) {
                    // Annex-B 格式需要转换为 HVCC (较复杂，暂不实现)
                    console.log(`⚠️ HEVC Annex-B extradata detected, conversion not yet implemented`);
                    // 可以尝试不带 description 解码
                } else if (isHVCC) {
                    // extradata 已经是 HVCC 格式，直接使用
                    config.description = extradata;
                    console.log(`📦 HEVC HVCC description (native): ${extradata.length} bytes`);
                }
            }
        }
        
        console.log(`🎬 Configuring decoder: ${codec.toUpperCase()} ${width}x${height}`);
        console.log(`   Codec string: ${codecString}`);
        console.log(`   Has description: ${!!config.description}`);
        if (extradata && extradata.length > 0) {
            console.log(`   Extradata header: [${Array.from(extradata.slice(0, 8)).join(',')}]`);
        }
        
        VideoDecoder.isConfigSupported(config).then((support) => {
            if (support.supported) {
                console.log(`✅ ${codec.toUpperCase()} ${width}x${height} Decoding Supported`);
                try {
                    this.decoder.configure(config);
                    this.decoderConfigured = true;
                    this.decoderHasDescription = !!config.description;
                    console.log(`✅ Decoder configured successfully (description=${this.decoderHasDescription})`);
                } catch (e) {
                    console.error(`❌ Decoder configure failed:`, e);
                    this.decoderHasDescription = false;
                }
            } else {
                console.error(`❌ ${codec.toUpperCase()} ${width}x${height} NOT Supported`);
                
                // 检测是否是 HEVC 不支持的情况
                if (isHEVC) {
                    const platform = this.detectPlatform();
                    console.error('❌ ========================================');
                    console.error('❌ HEVC/H.265 解码不受支持！');
                    console.error(`❌ 检测到平台: ${platform}`);
                    
                    if (platform === 'windows') {
                        console.error('❌ 可能的原因:');
                        console.error('❌ 1. Windows N/KN 版本缺少媒体功能包');
                        console.error('❌ 2. 未安装 "HEVC视频扩展" (Microsoft Store)');
                        console.error('❌ 3. WebView2/Edge 版本过旧');
                        console.error('❌ ');
                        console.error('❌ 解决方案:');
                        console.error('❌ 1. 在 Microsoft Store 搜索安装 "HEVC视频扩展"');
                        console.error('❌    或者搜索 "HEVC Video Extensions from Device Manufacturer" (免费)');
                        console.error('❌ 2. 更新 Windows 和 Edge 浏览器到最新版本');
                        console.error('❌ 3. 如果是 Windows N/KN 版，安装媒体功能包');
                        console.error('❌ ========================================');
                        
                        // Windows: 显示安装按钮
                        this.showErrorMessage([
                            '❌ HEVC/H.265 解码不受支持',
                            '',
                            '您的系统未安装 HEVC 视频编解码器',
                            '',
                            '请点击下方按钮安装免费的 HEVC 扩展',
                            '安装完成后重启本程序即可'
                        ], true);  // true = 显示安装按钮
                    } else if (platform === 'macos') {
                        console.error('❌ macOS 通常原生支持 HEVC，但当前检测不支持');
                        console.error('❌ 可能的原因:');
                        console.error('❌ 1. macOS 版本过旧 (需要 10.13+)');
                        console.error('❌ 2. 使用的浏览器/WebView 不支持');
                        console.error('❌ 解决方案: 更新 macOS 到最新版本');
                        console.error('❌ ========================================');
                        
                        this.showErrorMessage([
                            '❌ HEVC/H.265 解码不受支持',
                            '',
                            'macOS 通常原生支持 HEVC',
                            '',
                            '请尝试更新 macOS 到最新版本',
                            '或检查系统偏好设置中的安全性设置'
                        ], false);
                    } else {
                        console.error('❌ Linux 系统可能需要安装额外的编解码器');
                        console.error('❌ 解决方案: 安装 ffmpeg 或 gstreamer HEVC 插件');
                        console.error('❌ ========================================');
                        
                        this.showErrorMessage([
                            '❌ HEVC/H.265 解码不受支持',
                            '',
                            'Linux 系统需要安装 HEVC 编解码器',
                            '',
                            '请安装: sudo apt install gstreamer1.0-libav',
                            '或安装完整的 ffmpeg 支持'
                        ], false);
                    }
                }
                
                // 尝试不带 description 的配置
                if (config.description) {
                    console.log('🔄 Retrying without description...');
                    delete config.description;
                    VideoDecoder.isConfigSupported(config).then((support2) => {
                        if (support2.supported) {
                            this.decoder.configure(config);
                            this.decoderConfigured = true;
                            this.decoderHasDescription = false;
                            console.log(`✅ Decoder configured (without description)`);
                        } else if (isHEVC) {
                            console.error('❌ HEVC 完全不支持，即使不带 description');
                        }
                    });
                }
            }
        }).catch(e => {
            console.error(`❌ isConfigSupported error:`, e);
            this.decoderHasDescription = false;
        });
    }
    
    /**
     * 将 Annex-B 格式的 SPS/PPS 转换为 AVCC 格式
     * Annex-B: 00 00 00 01 [SPS] 00 00 00 01 [PPS]
     * AVCC: [header] [SPS count] [SPS len] [SPS] [PPS count] [PPS len] [PPS]
     */
    annexBToAvcc(annexB) {
        // 解析 Annex-B 中的 NAL units
        const nalUnits = [];
        let i = 0;
        
        while (i < annexB.length) {
            // 查找起始码 00 00 00 01 或 00 00 01
            if (i + 3 < annexB.length && 
                annexB[i] === 0 && annexB[i+1] === 0 && annexB[i+2] === 0 && annexB[i+3] === 1) {
                i += 4;
            } else if (i + 2 < annexB.length && 
                annexB[i] === 0 && annexB[i+1] === 0 && annexB[i+2] === 1) {
                i += 3;
            } else {
                i++;
                continue;
            }
            
            // 找到下一个起始码或结束
            let end = i;
            while (end < annexB.length) {
                if (end + 3 < annexB.length && 
                    annexB[end] === 0 && annexB[end+1] === 0 && annexB[end+2] === 0 && annexB[end+3] === 1) {
                    break;
                }
                if (end + 2 < annexB.length && 
                    annexB[end] === 0 && annexB[end+1] === 0 && annexB[end+2] === 1) {
                    break;
                }
                end++;
            }
            
            if (end > i) {
                nalUnits.push(annexB.slice(i, end));
            }
            i = end;
        }
        
        // 分离 SPS 和 PPS
        const spsList = [];
        const ppsList = [];
        
        for (const nalu of nalUnits) {
            if (nalu.length === 0) continue;
            const nalType = nalu[0] & 0x1F;
            if (nalType === 7) { // SPS
                spsList.push(nalu);
                console.log(`📊 Found SPS: ${nalu.length} bytes`);
            } else if (nalType === 8) { // PPS
                ppsList.push(nalu);
                console.log(`📊 Found PPS: ${nalu.length} bytes`);
            }
        }
        
        if (spsList.length === 0 || ppsList.length === 0) {
            console.warn('⚠️ Missing SPS or PPS in extradata');
            return null;
        }
        
        const sps = spsList[0];
        
        // 构建 AVCC 格式
        // [0] version = 1
        // [1] profile_idc
        // [2] profile_compatibility
        // [3] level_idc
        // [4] 0xFF (6 bits reserved + 2 bits NAL length size - 1 = 3)
        // [5] 0xE0 | num_sps (3 bits reserved + 5 bits SPS count)
        // [6-7] SPS length (big-endian)
        // [...] SPS data
        // [n] num_pps
        // [n+1, n+2] PPS length (big-endian)
        // [...] PPS data
        
        let totalSize = 6; // header
        for (const s of spsList) {
            totalSize += 2 + s.length;
        }
        totalSize += 1; // PPS count
        for (const p of ppsList) {
            totalSize += 2 + p.length;
        }
        
        const avcc = new Uint8Array(totalSize);
        let offset = 0;
        
        // Header
        avcc[offset++] = 1; // version
        avcc[offset++] = sps[1]; // profile_idc
        avcc[offset++] = sps[2]; // profile_compatibility
        avcc[offset++] = sps[3]; // level_idc
        avcc[offset++] = 0xFF; // NAL length size = 4
        avcc[offset++] = 0xE0 | spsList.length; // SPS count
        
        // SPS entries
        for (const s of spsList) {
            avcc[offset++] = (s.length >> 8) & 0xFF;
            avcc[offset++] = s.length & 0xFF;
            avcc.set(s, offset);
            offset += s.length;
        }
        
        // PPS count
        avcc[offset++] = ppsList.length;
        
        // PPS entries
        for (const p of ppsList) {
            avcc[offset++] = (p.length >> 8) & 0xFF;
            avcc[offset++] = p.length & 0xFF;
            avcc.set(p, offset);
            offset += p.length;
        }
        
        console.log(`📦 AVCC created: ${avcc.length} bytes (SPS: ${spsList.length}, PPS: ${ppsList.length})`);
        return avcc;
    }
    
    initAudioDecoder(codec = 'aac', sampleRate = 48000, channels = 2, description = null) {
        // 避免重复初始化相同配置
        if (this.audioCodec === codec && 
            this.audioSampleRate === sampleRate && 
            this.audioChannels === channels &&
            this.audioDecoder && 
            this.audioDecoder.state === 'configured') {
            return;
        }
        
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
                // 使用保存的 PTS 建立音频时钟
                this.playAudioData(audioData, this._pendingAudioPts);
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
        
        // 如果有 AAC description (AudioSpecificConfig)，添加到配置
        if (description && codec === 'aac') {
            config.description = description;
            console.log(`🎵 AAC AudioSpecificConfig: [${Array.from(description).map(b => b.toString(16).padStart(2, '0')).join(' ')}]`);
        }
        
        console.log(`🎵 Configuring audio decoder: ${codec.toUpperCase()} ${sampleRate}Hz ${channels}ch`);
        
        // 同步配置解码器
        try {
            this.audioDecoder.configure(config);
            this.nextAudioTime = this.audioContext.currentTime;
            console.log(`✅ Audio decoder configured: ${codecString} ${sampleRate}Hz ${channels}ch`);
        } catch (e) {
            console.error(`❌ Audio decoder configure failed:`, e);
        }
    }
    
    playAudioData(audioData, pts = null) {
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
        
        this.scheduleAudioBuffer(buffer, pts);
    }
    
    /**
     * 调度音频缓冲区播放
     * 同时建立音频时钟，供视频同步参考
     */
    scheduleAudioBuffer(buffer, pts = null) {
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        // 连接到滤波器链进行降噪，而不是直接输出
        source.connect(this.audioFilterInput);
        
        const currentTime = this.audioContext.currentTime;
        const bufferAhead = this.nextAudioTime - currentTime;
        
        // 建立音频时钟（首个有 PTS 的音频包）
        if (pts !== null && !this.audioClockReady) {
            this.audioClockBase = pts;
            this.audioClockStart = currentTime + this.audioBufferAhead; // 预缓冲后开始播放的时间
            this.audioClockReady = true;
            console.log(`🎵 音频时钟建立: basePts=${pts}ms, startTime=${this.audioClockStart.toFixed(3)}s`);
        }
        
        // 缓冲管理
        if (bufferAhead < 0) {
            // 缓冲耗尽，从当前时间+预缓冲开始
            this.nextAudioTime = currentTime + this.audioBufferAhead;
        } else if (bufferAhead > this.audioMaxBuffer) {
            // 缓冲过大，重置
            console.warn(`⚠️ 音频缓冲过大 ${(bufferAhead * 1000).toFixed(0)}ms，重置`);
            this.nextAudioTime = currentTime + this.audioBufferAhead;
        }
        
        source.start(this.nextAudioTime);
        this.nextAudioTime += buffer.duration;
    }

    handleVideoChunk(data, isKeyframe = true, pts = 0) {
        if (!this.isRunning) {
            return;
        }
        
        if (!this.decoder || this.decoder.state === 'closed') {
            return;
        }
        
        if (this.decoder.state !== 'configured') {
            if (!this.decoderConfigured) {
                return;
            }
        }
        
        // ========== 音视频同步：暂时禁用，让音视频独立播放 ==========
        // 直播流的音视频通常已经在服务器端同步好了，前端不需要额外处理
        // 如果需要启用同步，取消下面的注释
        /*
        if (this.audioClockReady && pts > 0) {
            // 同步逻辑暂时禁用
        }
        */
        
        // 如果正在等待关键帧，只处理关键帧
        if (this.waitingForKeyframe) {
            if (isKeyframe) {
                console.log('🔑 收到关键帧，恢复解码');
                this.waitingForKeyframe = false;
                // 更新面板
                if (typeof streamInfoManager !== 'undefined' && streamInfoManager) {
                    streamInfoManager.updateDecodeStatus({
                        waitingKeyframe: false
                    });
                }
            } else {
                // 跳过非关键帧
                this.skippedFrameCount++;
                return;
            }
        }
        
        // 检测帧数据格式: Annex-B vs AVCC
        const isInputAnnexB = data.length >= 4 && 
            ((data[0] === 0 && data[1] === 0 && data[2] === 0 && data[3] === 1) ||
             (data[0] === 0 && data[1] === 0 && data[2] === 1));
        
        // 首包打印日志
        if (this.videoChunkCount === 0) {
            const header = Array.from(data.slice(0, Math.min(20, data.length)));
            console.log(`🔑 First packet: size=${data.length}, isAnnexB=${isInputAnnexB}, hasDescription=${this.decoderHasDescription}, header=[${header.join(',')}]`);
        }
        
        // 准备帧数据 - 格式转换
        // 后端统一发送 Annex-B 格式，前端根据解码器配置决定是否转换
        let frameData = data;
        
        if (this.decoderHasDescription) {
            // 有 description (AVCC/HVCC): WebCodecs 需要 AVCC 格式 (4字节长度前缀)
            // 但实际上有些浏览器/配置也接受 Annex-B
            // 尝试直接使用，如果失败再转换
            if (isInputAnnexB) {
                // 尝试 AVCC 格式
                frameData = this.annexBFrameToAvcc(data);
            }
        } else {
            // 无 description: 帧数据需要 Annex-B 格式
            if (!isInputAnnexB) {
                frameData = this.avccFrameToAnnexB(data);
            }
        }
        
        this.videoChunkCount = (this.videoChunkCount || 0) + 1;
        
        // 时间戳处理 (微秒)
        // 直播流的 PTS 可能很大（累积时间），需要转换为相对时间戳
        if (pts > 0) {
            // 记录第一帧的基准时间戳
            if (!this.basePts) {
                this.basePts = pts;
                this.baseTime = performance.now();
                console.log(`🕐 视频基准时间戳: pts=${pts}ms, 本地=${this.baseTime.toFixed(0)}ms`);
            }
            // 计算相对时间戳
            const relativePts = pts - this.basePts;
            // 转换为微秒
            var timestamp = relativePts * 1000;
            
            // 每100帧打印状态（包含同步信息）
            if (this.videoChunkCount % 100 === 0) {
                const audioTime = this.getAudioClockTime();
                let syncInfo = '';
                if (audioTime !== null && this.audioClockReady) {
                    const videoRelative = relativePts;
                    const audioRelative = audioTime - this.audioClockBase;
                    const drift = videoRelative - audioRelative;
                    syncInfo = `, 音频相对=${audioRelative.toFixed(0)}ms, 视频相对=${videoRelative.toFixed(0)}ms, 偏差=${drift.toFixed(0)}ms`;
                }
                console.log(`📊 解码状态 #${this.videoChunkCount}: queueSize=${this.decoder.decodeQueueSize}, state=${this.decoder.state}${syncInfo}`);
            }
        } else {
            var timestamp = performance.now() * 1000;
        }
        
        // 正确设置帧类型
        const chunkType = isKeyframe ? 'key' : 'delta';
        
        const chunk = new EncodedVideoChunk({
            type: chunkType,
            timestamp: timestamp,
            data: frameData
        });
        
        try {
            // 检查解码器状态
            if (this.decoder.state !== 'configured') {
                console.warn(`⚠️ 解码器状态异常: ${this.decoder.state}`);
                return;
            }
            
            // 检查解码队列是否过大，避免阻塞
            if (this.decoder.decodeQueueSize > 10) {
                // 队列过大，跳过非关键帧
                if (!isKeyframe) {
                    this.skippedFrameCount++;
                    if (this.videoChunkCount % 100 === 0) {
                        console.warn(`⚠️ 解码队列过大 (${this.decoder.decodeQueueSize})，跳过非关键帧，累计跳帧: ${this.skippedFrameCount}`);
                    }
                    // 更新面板
                    if (typeof streamInfoManager !== 'undefined' && streamInfoManager && this.videoChunkCount % 10 === 0) {
                        streamInfoManager.updateDecodeStatus({
                            state: 'configured',
                            waitingKeyframe: this.waitingForKeyframe,
                            queueSize: this.decoder.decodeQueueSize,
                            decodedFrames: this.decodedFrameCount,
                            skippedFrames: this.skippedFrameCount,
                            errors: this.decodeErrorCount,
                            lastError: this.lastDecodeError
                        });
                    }
                    return;
                }
            }
            this.decoder.decode(chunk);
        } catch(e) {
            // 解码错误时打印更多信息
            this.decodeErrorCount++;
            this.lastDecodeError = e.message;
            this.waitingForKeyframe = true;  // 错误后等待关键帧
            console.warn(`❌ Decode error #${this.videoChunkCount}:`, e.message, 
                `isKeyframe=${isKeyframe}, size=${frameData.length}, timestamp=${timestamp}, state=${this.decoder.state}`);
            // 更新面板显示错误
            if (typeof streamInfoManager !== 'undefined' && streamInfoManager) {
                streamInfoManager.updateDecodeStatus({
                    state: 'error',
                    waitingKeyframe: this.waitingForKeyframe,
                    queueSize: this.decoder?.decodeQueueSize || 0,
                    decodedFrames: this.decodedFrameCount,
                    skippedFrames: this.skippedFrameCount,
                    errors: this.decodeErrorCount,
                    lastError: this.lastDecodeError
                });
            }
        }
    }
    
    /**
     * 检测 AVCC 格式数据中是否包含关键帧
     */
    detectKeyframeInAvcc(data) {
        const isHevc = this.currentCodec === 'hevc' || this.currentCodec === 'hvc1' || this.currentCodec === 'hev1';
        
        let offset = 0;
        while (offset + 4 < data.length) {
            // 读取 4 字节长度 (大端序)
            const naluLen = (data[offset] << 24) | (data[offset + 1] << 16) | 
                           (data[offset + 2] << 8) | data[offset + 3];
            offset += 4;
            
            if (naluLen <= 0 || offset + naluLen > data.length) {
                break;
            }
            
            const nalByte = data[offset];
            
            if (isHevc) {
                const nalType = (nalByte >> 1) & 0x3F;
                // HEVC IDR: 16-21 (真正的关键帧)
                if (nalType >= 16 && nalType <= 21) {
                    return true;
                }
            } else {
                const nalType = nalByte & 0x1F;
                // H.264 IDR slice (type=5) 才是真正的关键帧
                // SPS(7)/PPS(8) 是参数集，不是关键帧
                if (nalType === 5) {
                    return true;
                }
            }
            
            offset += naluLen;
        }
        return false;
    }
    
    /**
     * 将 AVCC 格式帧数据转换为 Annex-B 格式
     * AVCC: [4字节长度][NALU]...
     * Annex-B: [00 00 00 01][NALU]...
     */
    avccFrameToAnnexB(avcc) {
        const nalus = [];
        let offset = 0;
        
        while (offset + 4 <= avcc.length) {
            // 读取 4 字节长度 (大端序)
            const naluLen = (avcc[offset] << 24) | (avcc[offset + 1] << 16) | 
                           (avcc[offset + 2] << 8) | avcc[offset + 3];
            offset += 4;
            
            if (naluLen <= 0 || offset + naluLen > avcc.length) {
                break;
            }
            
            nalus.push(avcc.slice(offset, offset + naluLen));
            offset += naluLen;
        }
        
        // 计算总大小
        let totalSize = 0;
        for (const nalu of nalus) {
            totalSize += 4 + nalu.length;  // 4 字节起始码 + NALU
        }
        
        // 构建 Annex-B 数据
        const annexB = new Uint8Array(totalSize);
        let pos = 0;
        
        for (const nalu of nalus) {
            // 写入起始码
            annexB[pos++] = 0;
            annexB[pos++] = 0;
            annexB[pos++] = 0;
            annexB[pos++] = 1;
            // 写入 NALU
            annexB.set(nalu, pos);
            pos += nalu.length;
        }
        
        return annexB;
    }
    
    /**
     * 检测 Annex-B 数据中是否包含关键帧
     */
    detectKeyframeInAnnexB(data) {
        const isHevc = this.currentCodec === 'hevc' || this.currentCodec === 'hvc1' || this.currentCodec === 'hev1';
        
        let i = 0;
        while (i + 4 < data.length) {
            // 查找起始码
            if (data[i] === 0 && data[i + 1] === 0) {
                let startCodeLen = 0;
                if (data[i + 2] === 0 && data[i + 3] === 1) {
                    startCodeLen = 4;
                } else if (data[i + 2] === 1) {
                    startCodeLen = 3;
                }
                
                if (startCodeLen > 0 && i + startCodeLen < data.length) {
                    const nalByte = data[i + startCodeLen];
                    
                    if (isHevc) {
                        // HEVC: NAL type = (byte >> 1) & 0x3F
                        const nalType = (nalByte >> 1) & 0x3F;
                        // IDR: 16-21 (真正的关键帧)
                        if (nalType >= 16 && nalType <= 21) {
                            return true;
                        }
                    } else {
                        // H.264: NAL type = byte & 0x1F
                        const nalType = nalByte & 0x1F;
                        // 只有 IDR slice (type=5) 才是真正的关键帧
                        // SPS(7)/PPS(8) 是参数集，不是关键帧
                        if (nalType === 5) {
                            return true;
                        }
                    }
                    i += startCodeLen;
                    continue;
                }
            }
            i++;
        }
        return false;
    }
    
    /**
     * 检查帧数据中是否已包含参数集 (VPS/SPS/PPS)
     */
    hasParameterSets(data) {
        const isHevc = this.currentCodec === 'hevc' || this.currentCodec === 'hvc1' || this.currentCodec === 'hev1';
        
        let i = 0;
        while (i + 4 < data.length) {
            if (data[i] === 0 && data[i + 1] === 0) {
                let startCodeLen = 0;
                if (data[i + 2] === 0 && data[i + 3] === 1) {
                    startCodeLen = 4;
                } else if (data[i + 2] === 1) {
                    startCodeLen = 3;
                }
                
                if (startCodeLen > 0 && i + startCodeLen < data.length) {
                    const nalByte = data[i + startCodeLen];
                    
                    if (isHevc) {
                        const nalType = (nalByte >> 1) & 0x3F;
                        // VPS: 32, SPS: 33, PPS: 34
                        if (nalType >= 32 && nalType <= 34) {
                            return true;
                        }
                    } else {
                        const nalType = nalByte & 0x1F;
                        // SPS: 7, PPS: 8
                        if (nalType === 7 || nalType === 8) {
                            return true;
                        }
                    }
                    i += startCodeLen;
                    continue;
                }
            }
            i++;
        }
        return false;
    }
    
    /**
     * 检查 AVCC 格式帧数据中是否已包含参数集 (VPS/SPS/PPS)
     */
    hasParameterSetsAvcc(data) {
        const isHevc = this.currentCodec === 'hevc' || this.currentCodec === 'hvc1' || this.currentCodec === 'hev1';
        
        let i = 0;
        while (i + 4 < data.length) {
            const naluLen = (data[i] << 24) | (data[i + 1] << 16) | (data[i + 2] << 8) | data[i + 3];
            if (naluLen <= 0 || naluLen > data.length - i - 4) break;
            
            const nalByte = data[i + 4];
            
            if (isHevc) {
                const nalType = (nalByte >> 1) & 0x3F;
                if (nalType >= 32 && nalType <= 34) return true;
            } else {
                const nalType = nalByte & 0x1F;
                if (nalType === 7 || nalType === 8) return true;
            }
            
            i += 4 + naluLen;
        }
        return false;
    }
    
    /**
     * 将 Annex-B 格式的帧数据转换为 AVCC 格式
     * Annex-B: [00 00 00 01][NALU]... 或 [00 00 01][NALU]...
     * AVCC: [4字节长度][NALU]...
     */
    annexBFrameToAvcc(annexB) {
        const nalus = [];
        let i = 0;
        
        while (i < annexB.length) {
            // 查找起始码
            let startCodeLen = 0;
            if (i + 4 <= annexB.length && 
                annexB[i] === 0 && annexB[i+1] === 0 && annexB[i+2] === 0 && annexB[i+3] === 1) {
                startCodeLen = 4;
            } else if (i + 3 <= annexB.length && 
                annexB[i] === 0 && annexB[i+1] === 0 && annexB[i+2] === 1) {
                startCodeLen = 3;
            }
            
            if (startCodeLen === 0) {
                i++;
                continue;
            }
            
            i += startCodeLen;
            const naluStart = i;
            
            // 找到下一个起始码或数据结束
            while (i < annexB.length) {
                if (i + 4 <= annexB.length && 
                    annexB[i] === 0 && annexB[i+1] === 0 && annexB[i+2] === 0 && annexB[i+3] === 1) {
                    break;
                }
                if (i + 3 <= annexB.length && 
                    annexB[i] === 0 && annexB[i+1] === 0 && annexB[i+2] === 1) {
                    break;
                }
                i++;
            }
            
            if (i > naluStart) {
                nalus.push(annexB.slice(naluStart, i));
            }
        }
        
        // 计算总大小: 每个 NALU 有 4 字节长度前缀
        let totalSize = 0;
        for (const nalu of nalus) {
            totalSize += 4 + nalu.length;
        }
        
        // 构建 AVCC 格式数据
        const avcc = new Uint8Array(totalSize);
        let offset = 0;
        
        for (const nalu of nalus) {
            // 写入 4 字节长度 (大端序)
            const len = nalu.length;
            avcc[offset++] = (len >> 24) & 0xFF;
            avcc[offset++] = (len >> 16) & 0xFF;
            avcc[offset++] = (len >> 8) & 0xFF;
            avcc[offset++] = len & 0xFF;
            // 写入 NALU 数据
            avcc.set(nalu, offset);
            offset += nalu.length;
        }
        
        return avcc;
    }
    
    handleAudioChunk(data, pts = null) {
        if (this.audioCodec === 'pcm_alaw') {
            // A-Law 解码为 16-bit PCM
            const pcm16 = this.decodeAlaw(data);
            this.playPCMData(pcm16, pts);
            return;
        }
        
        if (this.audioCodec === 'pcm_mulaw') {
            // μ-Law 解码为 16-bit PCM
            const pcm16 = this.decodeMulaw(data);
            this.playPCMData(pcm16, pts);
            return;
        }
        
        if (!this.audioDecoder || this.audioDecoder.state !== 'configured') {
            console.warn('⚠️ Audio decoder not ready:', this.audioDecoder?.state);
            return;
        }
        
        // 保存 PTS 供解码回调使用
        this._pendingAudioPts = pts;
        
        const chunk = new EncodedAudioChunk({
            type: 'key',
            timestamp: pts !== null ? pts * 1000 : performance.now() * 1000,  // 微秒
            data: data
        });
        
        try {
            this.audioDecoder.decode(chunk);
        } catch(e) {
            console.error('Audio decode error:', e);
        }
    }
    
    /**
     * A-Law 解码 (ITU-T G.711 A)
     * 8-bit A-Law → 16-bit Linear PCM
     */
    decodeAlaw(alawData) {
        const pcm16 = new Int16Array(alawData.length);
        
        for (let i = 0; i < alawData.length; i++) {
            let alaw = alawData[i] ^ 0x55;  // A-Law 反转
            
            const sign = alaw & 0x80;
            const exponent = (alaw >> 4) & 0x07;
            const mantissa = alaw & 0x0F;
            
            let sample;
            if (exponent === 0) {
                sample = (mantissa << 4) + 8;
            } else {
                sample = ((mantissa << 4) + 0x108) << (exponent - 1);
            }
            
            pcm16[i] = sign ? -sample : sample;
        }
        
        return pcm16;
    }
    
    /**
     * μ-Law 解码 (ITU-T G.711 μ)
     * 8-bit μ-Law → 16-bit Linear PCM
     */
    decodeMulaw(mulawData) {
        const pcm16 = new Int16Array(mulawData.length);
        
        for (let i = 0; i < mulawData.length; i++) {
            let mulaw = ~mulawData[i];  // μ-Law 反转
            
            const sign = mulaw & 0x80;
            const exponent = (mulaw >> 4) & 0x07;
            const mantissa = mulaw & 0x0F;
            
            let sample = ((mantissa << 3) + 0x84) << exponent;
            sample -= 0x84;
            
            pcm16[i] = sign ? -sample : sample;
        }
        
        return pcm16;
    }
    
    playPCM16(data, pts = null) {
        const pcmData = new Int16Array(data.buffer, data.byteOffset, data.byteLength / 2);
        this.playPCMData(pcmData, pts);
    }
    
    playPCMData(pcmData, pts = null) {
        const sampleRate = this.audioSampleRate;
        const channels = this.audioChannels;
        const frameCount = pcmData.length / channels;
        
        if (this.audioContext.state === 'suspended') {
            console.warn('⚠️ AudioContext suspended, resuming...');
            this.audioContext.resume();
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
        
        // 使用统一的调度逻辑，传递 PTS 建立音频时钟
        this.scheduleAudioBuffer(buffer, pts);
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
            // 打印帧的色彩空间信息
            console.log(`[Frame ColorSpace] format=${frame.format}, colorSpace=${JSON.stringify(frame.colorSpace)}`);
        }

        // 应用色彩校正滤镜（使用调色面板设置）
        // 默认值：contrast(1.2) saturate(1.3) 修复 WebView2 色彩偏白问题
        this.ctx.filter = this.colorFilter || 'contrast(1.2) saturate(1.3)';
        
        // 清空画布
        this.ctx.fillStyle = '#0a0a0f';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 计算等比例缩放后的目标尺寸和位置 (居中显示)
        const canvasAspect = this.canvas.width / this.canvas.height;
        const frameAspect = frame.displayWidth / frame.displayHeight;
        
        let drawWidth, drawHeight, drawX, drawY;
        
        if (frameAspect > canvasAspect) {
            // 视频更宽，以宽度为准
            drawWidth = this.canvas.width;
            drawHeight = this.canvas.width / frameAspect;
            drawX = 0;
            drawY = (this.canvas.height - drawHeight) / 2;
        } else {
            // 视频更高，以高度为准
            drawHeight = this.canvas.height;
            drawWidth = this.canvas.height * frameAspect;
            drawX = (this.canvas.width - drawWidth) / 2;
            drawY = 0;
        }
        
        this.ctx.drawImage(
            frame, 
            0, 0, frame.displayWidth, frame.displayHeight,
            drawX, drawY, drawWidth, drawHeight
        );
        
        // 重置滤镜
        this.ctx.filter = 'none';
        
        // 保存最后一帧的副本，用于窗口 resize 时重绘
        // 使用 createImageBitmap 异步创建副本，不会阻塞渲染
        createImageBitmap(frame).then(bitmap => {
            // 释放旧的 bitmap
            if (this.lastImageBitmap) {
                this.lastImageBitmap.close();
            }
            this.lastImageBitmap = bitmap;
        }).catch(() => {
            // 忽略错误（帧可能已被关闭）
        });

        this.frameCount++;
        if (now - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = now;
            // 更新流信息面板的 FPS 和 Latency
            if (streamInfoManager) {
                const latency = Math.round(this.lastFrameTime > 0 ? (now - this.lastFrameTime) : 0);
                streamInfoManager.updatePerformance(this.fps, latency);
            }
        }
    }

    start() {
        this.isRunning = true;
    }
    
    stop() {
        this.isRunning = false;
        this.hwDecodeRunning = false;
        // 清理保存的最后一帧
        if (this.lastImageBitmap) {
            this.lastImageBitmap.close();
            this.lastImageBitmap = null;
        }
        // 重置音频配置
        this.audioConfigured = false;
        this.audioCodec = null;
        // 重置时间戳基准
        this.basePts = null;
        this.baseTime = null;
        this.resetAudioBuffer();
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
                        // 更新流信息面板
                        if (streamInfoManager) {
                            streamInfoManager.updatePerformance(this.fps, 0);
                        }
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
        if (this.overlayCanvas) {
            this.overlayCanvas.style.transform = transform;
            this.overlayCanvas.style.transformOrigin = '0 0';
        }
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
    const overlayCanvas = document.getElementById('overlay-canvas');
    if (canvas) {
        canvasTransform = new CanvasTransform(canvas, overlayCanvas);
        window.canvasTransform = canvasTransform;
    }
    
    // 初始化流信息管理器
    streamInfoManager = new StreamInfoManager();
    console.log('📊 StreamInfoManager 已初始化');
    
    // 初始化调色面板
    colorGrading = new ColorGrading(renderer);
    console.log('🎨 ColorGrading 已初始化');
    
    // 流信息面板开关 (需要在 streamInfoManager 初始化后注册)
    const streamInfoToggle = document.getElementById('stream-info-toggle');
    streamInfoToggle?.addEventListener('change', (e) => {
        const enabled = e.target.checked;
        if (streamInfoManager) {
            streamInfoManager.setEnabled(enabled);
            console.log(`📊 流信息面板: ${enabled ? '启用' : '禁用'}`);
        }
    });
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

// ==================== 调色面板 ====================

class ColorGrading {
    constructor(renderer) {
        this.renderer = renderer;
        
        // 默认值
        this.defaults = {
            brightness: 1,
            contrast: 1.2,
            saturate: 1.3,
            hue: 0,
            blur: 0
        };
        
        // 当前值
        this.values = { ...this.defaults };
        
        // 预设
        this.presets = {
            vivid: { brightness: 1.05, contrast: 1.3, saturate: 1.6, hue: 0, blur: 0 },
            soft: { brightness: 1.1, contrast: 0.95, saturate: 0.9, hue: 0, blur: 0.5 }
        };
        
        this.setupEventListeners();
        this.updateFilter();
    }
    
    setupEventListeners() {
        // 展开/收起按钮
        const toggleBtn = document.getElementById('color-config-toggle');
        const panel = document.getElementById('color-config-panel');
        toggleBtn?.addEventListener('click', () => {
            panel?.classList.toggle('hidden');
            toggleBtn.textContent = panel?.classList.contains('hidden') ? '展开' : '收起';
        });
        
        // 滑块事件
        this.bindSlider('brightness', '%', 100);
        this.bindSlider('contrast', '%', 100);
        this.bindSlider('saturate', '%', 100);
        this.bindSlider('hue', '°', 1);
        this.bindSlider('blur', 'px', 1);
        
        // 重置按钮
        document.getElementById('color-reset-btn')?.addEventListener('click', () => {
            this.applyPreset(this.defaults);
        });
        
        // 预设按钮
        document.getElementById('color-preset-vivid')?.addEventListener('click', () => {
            this.applyPreset(this.presets.vivid);
        });
        document.getElementById('color-preset-soft')?.addEventListener('click', () => {
            this.applyPreset(this.presets.soft);
        });
    }
    
    bindSlider(name, suffix, multiplier) {
        const slider = document.getElementById(`${name}-slider`);
        const valueEl = document.getElementById(`${name}-value`);
        
        slider?.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.values[name] = val;
            
            if (valueEl) {
                if (suffix === '%') {
                    valueEl.textContent = Math.round(val * multiplier) + suffix;
                } else {
                    valueEl.textContent = val + suffix;
                }
            }
            
            this.updateFilter();
        });
    }
    
    applyPreset(preset) {
        this.values = { ...preset };
        
        // 更新滑块
        const sliders = ['brightness', 'contrast', 'saturate', 'hue', 'blur'];
        sliders.forEach(name => {
            const slider = document.getElementById(`${name}-slider`);
            const valueEl = document.getElementById(`${name}-value`);
            if (slider) slider.value = this.values[name];
            if (valueEl) {
                if (name === 'hue') {
                    valueEl.textContent = this.values[name] + '°';
                } else if (name === 'blur') {
                    valueEl.textContent = this.values[name] + 'px';
                } else {
                    valueEl.textContent = Math.round(this.values[name] * 100) + '%';
                }
            }
        });
        
        this.updateFilter();
        console.log('🎨 应用预设:', preset);
    }
    
    updateFilter() {
        const { brightness, contrast, saturate, hue, blur } = this.values;
        
        // 构建 CSS filter 字符串
        let filter = `brightness(${brightness}) contrast(${contrast}) saturate(${saturate})`;
        
        if (hue !== 0) {
            filter += ` hue-rotate(${hue}deg)`;
        }
        if (blur > 0) {
            filter += ` blur(${blur}px)`;
        }
        
        // 保存到渲染器
        if (this.renderer) {
            this.renderer.colorFilter = filter;
        }
        
        console.log('🎨 滤镜:', filter);
    }
    
    getFilter() {
        return this.renderer?.colorFilter || 'none';
    }
}

// 全局调色实例
let colorGrading = null;

// 音量控制
volumeSlider.addEventListener('input', (e) => {
    const gain = parseFloat(e.target.value);
    if (renderer) {
        renderer.setVolume(gain);
    }
    volumeValue.textContent = gain.toFixed(1) + 'x';
    console.log(`🔊 [Volume] Gain set to ${gain.toFixed(1)}x`);
});

// 降噪预设控制
const noisePreset = document.getElementById('noise-preset');
const notchSelect = document.getElementById('notch-select');

// 降噪预设配置
const NOISE_PRESETS = {
    off: { enabled: false, highpass: 20, lowpass: 20000, compression: false },
    light: { enabled: true, highpass: 60, lowpass: 14000, compression: false },
    normal: { enabled: true, highpass: 100, lowpass: 12000, compression: true },
    strong: { enabled: true, highpass: 150, lowpass: 8000, compression: true },
    extreme: { enabled: true, highpass: 200, lowpass: 4000, compression: true }
};

noisePreset?.addEventListener('change', (e) => {
    const preset = NOISE_PRESETS[e.target.value];
    if (renderer && preset) {
        renderer.applyNoisePreset(preset);
    }
    console.log(`🔇 [NoisePreset] ${e.target.value}`);
});

notchSelect?.addEventListener('change', (e) => {
    const freq = parseInt(e.target.value);
    if (renderer) {
        renderer.setNotchFrequency(freq);
    }
    console.log(`🔇 [Notch] ${freq > 0 ? freq + 'Hz' : '关闭'}`);
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
        const urls = await invoke('get_stream_history');
        
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
            await invoke('clear_stream_history');
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

function showStatus(message, isError = false, duration = 5000) {
    statusDiv.textContent = message;
    statusDiv.classList.add('show');
    
    // 错误状态样式
    if (isError) {
        statusDiv.style.background = 'rgba(239, 68, 68, 0.9)';
        statusDiv.style.borderColor = 'rgba(239, 68, 68, 0.5)';
    } else {
        statusDiv.style.background = '';
        statusDiv.style.borderColor = '';
    }
    
    setTimeout(() => {
        statusDiv.classList.remove('show');
        statusDiv.style.background = '';
        statusDiv.style.borderColor = '';
    }, isError ? 8000 : duration);  // 错误显示更长时间
}

// 解码模式：固定使用前端 WebCodecs 解码
const currentDecodeMode = 'frontend';

startBtn.addEventListener('click', async () => {
    const url = rtspInput.value.trim();
    if (!url) {
        showStatus('❌ 请输入 RTSP 地址');
        return;
    }
    
    try {
        startBtn.disabled = true;
        showStatus('🚀 正在启动 (WebCodecs 解码)...');
        
        // 前端 WebCodecs 解码模式
        // 使用 InvokeResponseBody: Json (配置) 或 Raw (视频数据)
        const onData = new Channel();
        let packetCount = 0;
        
        onData.onmessage = (response) => {
            // 调试：首次收到数据时打印类型
            if (packetCount < 5) {
                console.log(`📨 收到数据 #${packetCount}:`, typeof response, response?.constructor?.name, 
                    response instanceof ArrayBuffer ? `ArrayBuffer(${response.byteLength})` :
                    response instanceof Uint8Array ? `Uint8Array(${response.length})` :
                    Array.isArray(response) ? `Array(${response.length})` :
                    typeof response === 'string' ? `String(${response.length}): ${response.slice(0, 100)}` : 
                    typeof response === 'object' ? JSON.stringify(response).slice(0, 100) : 'unknown');
            }
            packetCount++;
            
            // 统一解析 JSON (字符串或对象)
            let message = null;
            if (typeof response === 'string') {
                try {
                    message = JSON.parse(response);
                } catch (e) {
                    console.error('解析 JSON 失败:', e);
                }
            } else if (typeof response === 'object' && response !== null && !(response instanceof ArrayBuffer)) {
                message = response;
            }
            
            // 处理 JSON 消息
            if (message && message.type) {
                console.log('📦 收到消息:', message.type);
                
                if (message.type === 'video_config') {
                    const { codec, width, height, extradata } = message;
                    console.log(`🎬 [Video Config] ${codec.toUpperCase()} ${width}x${height}`);
                    
                    // 将 extradata 数组转为 Uint8Array
                    let extradataBytes = null;
                    if (extradata && extradata.length > 0) {
                        extradataBytes = new Uint8Array(extradata);
                        console.log(`📦 Extradata: ${extradataBytes.length} bytes`);
                    }
                    
                    renderer.initDecoder(codec, width, height, extradataBytes);
                    if (typeof syncOverlayToCanvas === 'function') {
                        syncOverlayToCanvas(width, height);
                    }
                } else if (message.type === 'audio_config') {
                    const { audio_codec, audio_sample_rate, audio_channels, description } = message;
                    console.log(`🎵 [Audio Config] ${audio_codec} ${audio_sample_rate}Hz ${audio_channels}ch, desc=${description?.length || 0} bytes`);
                    
                    // 将后端 codec 字符串转换为前端格式
                    let frontendCodec;
                    if (audio_codec === 'mp4a.40.2') {
                        frontendCodec = 'aac';
                    } else if (audio_codec === 'pcm_alaw') {
                        frontendCodec = 'pcm_alaw';
                    } else if (audio_codec === 'pcm_mulaw') {
                        frontendCodec = 'pcm_mulaw';
                    } else if (audio_codec === 'mp3') {
                        frontendCodec = 'mp3';
                    } else {
                        frontendCodec = audio_codec;
                    }
                    
                    // 将 description 数组转换为 Uint8Array
                    const descBuffer = description && description.length > 0 ? new Uint8Array(description) : null;
                    renderer.initAudioDecoder(frontendCodec, audio_sample_rate, audio_channels, descBuffer);
                    renderer.audioConfigured = true;
                    
                    // 更新流信息面板的音频信息
                    if (streamInfoManager) {
                        streamInfoManager.updateAudioInfo(audio_codec, audio_sample_rate, audio_channels);
                    }
                } else if (message.type === 'stream_info') {
                    // 更新流信息面板
                    if (streamInfoManager) {
                        streamInfoManager.updateStreamInfo(message);
                        streamInfoManager.show();
                    }
                } else if (message.type === 'error') {
                    // 后端发送的错误消息
                    const errorMsg = message.error || '未知错误';
                    console.error('❌ 后端错误:', errorMsg);
                    showStatus(`❌ ${errorMsg}`, true);
                    // 显示在流信息面板
                    if (streamInfoManager) {
                        streamInfoManager.showError(errorMsg);
                    }
                }
            } else if (response instanceof ArrayBuffer) {
                // InvokeResponseBody::Raw 在前端是 ArrayBuffer
                const data = new Uint8Array(response);
                
                if (data.length < 2) {
                    console.warn('数据包太小:', data.length);
                    return;
                }
                
                const packetType = data[0]; // 1=video, 2=audio
                
                if (packetType === 1) {
                    // 视频包格式:
                    // [0]: packet_type = 1
                    // [1]: is_keyframe
                    // [2..10]: pts (i64 le)
                    // [10..18]: dts (i64 le)
                    // [18..26]: packet_id (u64 le)
                    // [26..30]: data_len (u32 le)
                    // [30..]: video data
                    if (data.length < 30) {
                        console.warn('视频包太小:', data.length);
                        return;
                    }
                    
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    const isKeyframe = data[1] === 1;
                    const pts = Number(view.getBigInt64(2, true));
                    const dataLen = view.getUint32(26, true);
                    const encodedData = data.slice(30, 30 + dataLen);
                    
                    renderer.handleVideoChunk(encodedData, isKeyframe, pts);
                    // 更新流信息面板统计
                    if (streamInfoManager) {
                        streamInfoManager.addPacket(encodedData.length, isKeyframe);
                    }
                } else if (packetType === 2) {
                    // 音频包格式:
                    // [0]: packet_type = 2
                    // [1]: codec_len
                    // [2..2+codec_len]: codec string
                    // [next 4 bytes]: sample_rate (u32 le)
                    // [next 1 byte]: channels
                    // [next 8 bytes]: pts (i64 le)
                    // [next 4 bytes]: data_len (u32 le)
                    // [rest]: audio data
                    const codecLen = data[1];
                    let pos = 2;
                    
                    const codecBytes = data.slice(pos, pos + codecLen);
                    const codec = new TextDecoder().decode(codecBytes);
                    pos += codecLen;
                    
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    const sampleRate = view.getUint32(pos, true);
                    pos += 4;
                    
                    const channels = data[pos];
                    pos += 1;
                    
                    const pts = Number(view.getBigInt64(pos, true));
                    pos += 8;
                    
                    const audioDataLen = view.getUint32(pos, true);
                    pos += 4;
                    
                    const audioData = data.slice(pos, pos + audioDataLen);
                    
                    // 将后端 codec 字符串转换为前端格式
                    let frontendCodec;
                    if (codec === 'mp4a.40.2') {
                        frontendCodec = 'aac';
                    } else if (codec === 'pcm_alaw') {
                        frontendCodec = 'pcm_alaw';
                    } else if (codec === 'pcm_mulaw') {
                        frontendCodec = 'pcm_mulaw';
                    } else if (codec === 'mp3') {
                        frontendCodec = 'mp3';
                    } else {
                        frontendCodec = codec;
                    }
                    
                    // 初始化解码器（内部会检查是否需要重新配置）
                    renderer.initAudioDecoder(frontendCodec, sampleRate, channels);
                    // 传递 PTS 进行音视频同步
                    renderer.handleAudioChunk(audioData, pts);
                }
            } else if (!message) {
                // 非 ArrayBuffer 且非 JSON 消息
                console.warn('未知响应类型:', typeof response, response);
            }
        };
        
        // 启动渲染器
        renderer.start();
        
        // 重置流信息面板统计
        if (streamInfoManager) {
            streamInfoManager.reset();
        }
        
        // Tauri 2.0 自动将 Rust 的 snake_case 转为 camelCase
        const result = await invoke('start_stream', { 
            url, 
            onData
        });
        console.log(result);
        
        showStatus('✅ 监控已启动 (WebCodecs)');
        
        try {
            await invoke('add_stream_history', { url });
            console.log('✅ 已保存到历史记录');
        } catch (e) {
            console.warn('保存历史记录失败:', e);
        }
        
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        rtspInput.disabled = true;
    } catch (err) {
        console.error('启动失败:', err);
        showStatus('❌ 启动失败: ' + err);
        startBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', async () => {
    try {
        // 停止渲染器
        renderer.stop();
        
        // 停止后端视频流
        await invoke('stop_stream');
        
        // 隐藏流信息面板
        if (streamInfoManager) {
            streamInfoManager.hide();
        }
        
        stopBtn.classList.add('hidden');
        startBtn.classList.remove('hidden');
        startBtn.disabled = false;
        rtspInput.disabled = false;
        showStatus('⏹ 监控已停止');
    } catch (e) {
        console.error('停止失败:', e);
        showStatus('❌ 停止失败: ' + e);
    }
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
