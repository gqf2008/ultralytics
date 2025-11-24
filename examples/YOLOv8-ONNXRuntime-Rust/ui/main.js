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
        this.audioDecoder = null;
        this.audioContext = null;
        this.audioQueue = [];
        this.nextAudioTime = 0;
        
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
    
    initAudioDecoder(codec = 'aac', sampleRate = 48000, channels = 2) {
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
        // 音频处理 - 暂时跳过 (PCM_ALAW WebCodecs 可能不支持)
        // 可以在这里实现 Web Audio API 播放
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
const historyToggle = document.getElementById('history-toggle');
const historyDropdown = document.getElementById('history-dropdown');
const controlPanel = document.getElementById('control-panel');
const panelHeader = document.getElementById('panel-header');
const toggleBtn = document.getElementById('toggle-btn');

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
        const history = await invoke('read_text_file', { 
            path: 'rtsp_history.txt' 
        });
        
        const urls = history.split('\n').filter(line => line.trim());
        
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
        
        // 创建单个 Channel 只接收视频
        const channel = new Channel();
        
        channel.onmessage = (data) => {
            console.log('[DEBUG] Channel received:', typeof data, data);
            
            // 尝试多种方式解析
            let binaryData = null;
            
            if (data instanceof Uint8Array) {
                binaryData = data;
            } else if (data instanceof ArrayBuffer) {
                binaryData = new Uint8Array(data);
            } else if (typeof data === 'object' && data !== null) {
                console.log('[DEBUG] Object keys:', Object.keys(data));
                console.log('[DEBUG] Object values:', Object.values(data));
                // 尝试提取各种可能的字段
                if (data.data) binaryData = data.data;
                else if (data.payload) binaryData = data.payload;
                else if (data.buffer) binaryData = data.buffer;
            }
            
            if (!binaryData) {
                console.warn('⚠️ Cannot extract binary data from:', data);
                return;
            }
            
            // 确保是 Uint8Array
            if (!(binaryData instanceof Uint8Array)) {
                binaryData = new Uint8Array(binaryData);
            }
            
            console.log('[DEBUG] Binary data length:', binaryData.length, 'first bytes:', Array.from(binaryData.slice(0, 10)));
            
            // 检查是否为元数据(0xFF 0xFE)
            if (binaryData.length > 2 && binaryData[0] === 0xFF && binaryData[1] === 0xFE) {
                const metadataStr = new TextDecoder().decode(binaryData.slice(2));
                try {
                    const metadata = JSON.parse(metadataStr);
                    console.log('📥 Metadata:', metadata);
                    renderer.initDecoder(metadata.video_codec, metadata.width, metadata.height);
                } catch (e) {
                    console.error('Metadata parse error:', e);
                }
            } else {
                // 纯视频数据
                renderer.handleVideoChunk(binaryData);
            }
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
