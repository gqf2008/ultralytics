/**
 * 流信息管理器模块
 * 显示视频流的元数据、统计信息和解码状态
 */

export class StreamInfoManager {
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

export default StreamInfoManager;
