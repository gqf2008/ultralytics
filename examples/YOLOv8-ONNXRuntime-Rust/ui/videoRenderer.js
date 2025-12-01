/**
 * WebGL 视频渲染器 - 使用 WebCodecs 解码和 Canvas2D 渲染
 * 音频处理已分离到独立模块: AudioDecoder, AudioPlayer, AudioDenoiser
 */
import { invoke } from '@tauri-apps/api/core';
import { AudioDecoder } from './audioDecoder.js';
import { AudioPlayer } from './audioPlayer.js';

let streamInfoManager = null;
export function setStreamInfoManager(manager) { streamInfoManager = manager; }
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
        
        // ========== 音频模块 (解耦) ==========
        this.audioPlayer = new AudioPlayer();
        this.audioDecoderModule = new AudioDecoder((samples, sampleRate, channels, timestamp) => {
            // 解码完成后播放
            this.audioPlayer.play(samples, sampleRate, channels);
        });
        
        // 兼容旧 API
        this.audioCodec = null;
        this.audioSampleRate = 8000;
        this.audioChannels = 1;
        this.audioGain = 1.0;
        
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
        
        // ========== 顶级降噪参数 ==========
        this.noiseGateThreshold = -35;  // 噪声门限阈值 (dB)
        this.noiseGateEnabled = true;
        this.wienerFilterEnabled = false;  // 默认关闭，容易产生回声
        this.noiseProfile = null;  // 噪声频谱特征
        this.denoiseStrength = 1.0;  // 降噪强度 0-2
        this.spectralFloor = 0.01;  // 频谱底噪
        
        // ========== RNNoise AI 降噪 ==========
        this.rnnoiseEnabled = false;  // RNNoise 开关
        this.rnnoiseReady = false;    // RNNoise 是否加载完成
        this.rnnoiseInstance = null;  // Rnnoise 实例
        this.rnnoiseState = null;     // DenoiseState
        this.rnnoiseFrameSize = 480;  // 默认帧大小，加载后更新
        this.rnnoiseBuffer = null;    // 处理缓冲，initRnnoise 后创建
        this.rnnoiseBufferPos = 0;    // 缓冲区位置
        this.rnnoiseOutputBuffer = [];  // 输出缓冲
        this.initRnnoise();  // 预加载
        
        // 初始化频谱降噪
        this.initSpectralDenoising();
        
        // ========== 音频滤波器链 ==========
        // 高通滤波器：去除低频（默认250Hz，人声下限）
        this.highpassFilter = this.audioContext.createBiquadFilter();
        this.highpassFilter.type = 'highpass';
        this.highpassFilter.frequency.value = 250;
        this.highpassFilter.Q.value = 1.0;
        
        // 第二个高通：级联 = 24dB/oct 更陡峭
        this.highpassFilter2 = this.audioContext.createBiquadFilter();
        this.highpassFilter2.type = 'highpass';
        this.highpassFilter2.frequency.value = 250;
        this.highpassFilter2.Q.value = 1.0;
        
        // 低通滤波器：去除高频（默认6kHz，人声上限）
        this.lowpassFilter = this.audioContext.createBiquadFilter();
        this.lowpassFilter.type = 'lowpass';
        this.lowpassFilter.frequency.value = 6000;
        this.lowpassFilter.Q.value = 0.7;
        
        // 第二个低通
        this.lowpassFilter2 = this.audioContext.createBiquadFilter();
        this.lowpassFilter2.type = 'lowpass';
        this.lowpassFilter2.frequency.value = 6000;
        this.lowpassFilter2.Q.value = 0.7;
        
        // 陷波滤波器：消除电源工频干扰 (50Hz)
        this.notchFilter = this.audioContext.createBiquadFilter();
        this.notchFilter.type = 'notch';
        this.notchFilter.frequency.value = 50;
        this.notchFilter.Q.value = 30;
        
        // 第二个陷波：消除谐波 (100Hz)
        this.notchFilter2 = this.audioContext.createBiquadFilter();
        this.notchFilter2.type = 'notch';
        this.notchFilter2.frequency.value = 100;
        this.notchFilter2.Q.value = 30;
        
        // 动态压缩器：更强的压缩防止爆音
        this.compressor = this.audioContext.createDynamicsCompressor();
        this.compressor.threshold.value = -30;  // 更低阈值
        this.compressor.knee.value = 20;        // 柔和过渡
        this.compressor.ratio.value = 16;       // 高压缩比
        this.compressor.attack.value = 0.001;   // 快速响应
        this.compressor.release.value = 0.1;    // 稍慢释放
        
        // 增益节点：音量控制
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = 1.0;  // 降低增益防止爆音
        
        // 连接滤波器链：
        // source -> hp1 -> hp2 -> notch1 -> notch2 -> lp1 -> lp2 -> compressor -> gain -> destination
        this.highpassFilter.connect(this.highpassFilter2);
        this.highpassFilter2.connect(this.notchFilter);
        this.notchFilter.connect(this.notchFilter2);
        this.notchFilter2.connect(this.lowpassFilter);
        this.lowpassFilter.connect(this.lowpassFilter2);
        this.lowpassFilter2.connect(this.compressor);
        this.compressor.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);
        
        // 保存滤波器链入口
        this.audioFilterInput = this.highpassFilter;
        this.audioDirectOutput = this.gainNode;
        this.noiseReductionEnabled = true;
        
        console.log('🔇 音频降噪: HP=200Hz LP=8kHz (强力模式，去除低频轰隆)');
        
        // 音频上下文可能被浏览器挂起
        document.addEventListener('click', () => {
            if (this.audioContext && this.audioContext.state === 'suspended') {
                this.audioContext.resume();
                console.log('🔊 Audio Context resumed');
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
            
            // 噪声门限设置
            if (preset.gate !== undefined) {
                this.setNoiseGateEnabled(preset.gate);
            }
            if (preset.gateThreshold !== undefined) {
                this.setNoiseGateThreshold(preset.gateThreshold);
            }
            // 降噪强度
            if (preset.strength !== undefined) {
                this.setDenoiseStrength(preset.strength / 100);
            }
        }
        
        console.log(`🔇 预设: HP=${preset.highpass}Hz LP=${preset.lowpass}Hz 门限=${preset.gateThreshold}dB 强度=${preset.strength}%`);
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
                // 直接使用 AudioData 的内置时间戳 (微秒) → 转换为毫秒
                const pts = audioData.timestamp / 1000;
                this.playAudioData(audioData, pts);
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
            
            // ========== 顶级降噪处理 ==========
            // 优先使用 RNNoise AI 降噪
            if (this.rnnoiseEnabled && this.rnnoiseReady) {
                this.applyRnnoise(channelData, audioData.sampleRate);
            } else if (this.noiseReductionEnabled) {
                // 回退到传统频谱降噪
                this.applySpectralDenoising(channelData);
            }
            
            // 采样噪声底噪（如果正在采样）
            if (this._samplingNoise) {
                this.collectNoiseProfile(channelData);
            }
            
            // ========== 软限幅防止爆音 ==========
            for (let i = 0; i < channelData.length; i++) {
                const x = channelData[i];
                // 使用 tanh 软限幅，超过 0.8 开始压缩
                if (Math.abs(x) > 0.8) {
                    channelData[i] = Math.tanh(x);
                }
            }
            
            buffer.copyToChannel(channelData, channel);
        }
        
        this.scheduleAudioBuffer(buffer, pts);
    }
    
    /**
     * 初始化 RNNoise AI 降噪
     */
    async initRnnoise() {
        try {
            console.log('🤖 正在加载 RNNoise AI 降噪引擎...');
            this.rnnoiseInstance = await Rnnoise.load();
            this.rnnoiseState = this.rnnoiseInstance.createDenoiseState();
            this.rnnoiseFrameSize = this.rnnoiseInstance.frameSize;  // 通常是 480
            this.rnnoiseBuffer = new Float32Array(this.rnnoiseFrameSize);
            this.rnnoiseReady = true;
            console.log(`🤖 RNNoise AI 降噪引擎加载成功！帧大小: ${this.rnnoiseFrameSize}`);
        } catch (error) {
            console.error('❌ RNNoise 加载失败:', error);
            this.rnnoiseReady = false;
        }
    }
    
    /**
     * 启用/禁用 RNNoise
     */
    setRnnoiseEnabled(enabled) {
        this.rnnoiseEnabled = enabled;
        if (enabled && !this.rnnoiseReady) {
            this.initRnnoise();
        }
        console.log(`🤖 RNNoise AI 降噪${enabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 应用 RNNoise 降噪
     * RNNoise 输入：Float32Array，值范围 -32768 到 32767 (16-bit PCM 范围)
     */
    applyRnnoise(samples, sampleRate) {
        if (!this.rnnoiseReady || !this.rnnoiseState || !this.rnnoiseBuffer) {
            return samples;
        }
        
        const frameSize = this.rnnoiseFrameSize;
        const scale = 32767;    // Float32 [-1,1] 转 Int16 范围
        const length = samples.length;
        
        // 使用环形缓冲处理
        for (let i = 0; i < length; i++) {
            // 转换到 Int16 范围
            this.rnnoiseBuffer[this.rnnoiseBufferPos++] = samples[i] * scale;
            
            // 当缓冲区满时处理
            if (this.rnnoiseBufferPos >= frameSize) {
                // RNNoise 处理（原地修改）
                this.rnnoiseState.processFrame(this.rnnoiseBuffer);
                
                // 把处理后的数据放回输出缓冲
                for (let j = 0; j < frameSize; j++) {
                    this.rnnoiseOutputBuffer.push(this.rnnoiseBuffer[j] / scale);
                }
                
                this.rnnoiseBufferPos = 0;
            }
        }
        
        // 从输出缓冲取数据写回 samples
        const available = Math.min(this.rnnoiseOutputBuffer.length, length);
        for (let i = 0; i < available; i++) {
            samples[i] = this.rnnoiseOutputBuffer.shift();
        }
        
        // 如果输出缓冲不够，保持原样（等待更多数据）
        // 这会引入少量延迟但保证连续性
        
        return samples;
    }
    
    /**
     * 销毁 RNNoise 实例（释放内存）
     */
    destroyRnnoise() {
        if (this.rnnoiseState) {
            this.rnnoiseState.destroy();
            this.rnnoiseState = null;
        }
        this.rnnoiseReady = false;
        console.log('🤖 RNNoise 已销毁');
    }

    /**
     * 初始化 FFT 降噪所需的数据结构
     */
    initSpectralDenoising() {
        this.fftSize = 2048;
        this.hopSize = this.fftSize / 4;  // 75% 重叠
        this.noiseProfile = null;  // 噪声频谱特征
        this.prevPhase = new Float32Array(this.fftSize);
        this.inputBuffer = new Float32Array(this.fftSize);
        this.outputBuffer = new Float32Array(this.fftSize * 2);
        this.outputReadPos = 0;
        this.outputWritePos = 0;
        this.inputWritePos = 0;
        
        // 汉宁窗
        this.window = new Float32Array(this.fftSize);
        for (let i = 0; i < this.fftSize; i++) {
            this.window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.fftSize - 1)));
        }
        
        // 降噪参数
        this.denoiseStrength = 1.0;  // 降噪强度 0-2
        this.spectralFloor = 0.002;  // 频谱底噪
        
        console.log('🔊 FFT 频谱降噪初始化完成');
    }
    
    /**
     * 收集噪声特征（采样时调用）
     */
    collectNoiseProfile(samples) {
        if (!this.noiseProfileFrames) {
            this.noiseProfileFrames = [];
        }
        
        // 计算当前帧的频谱
        const spectrum = this.computeSpectrum(samples);
        this.noiseProfileFrames.push(spectrum);
        
        this._noiseSampleCount++;
    }
    
    /**
     * 计算信号的幅度谱（简化 FFT）
     */
    computeSpectrum(samples) {
        const n = Math.min(samples.length, this.fftSize);
        const spectrum = new Float32Array(n / 2);
        
        // 简化 DFT（对于实时降噪够用）
        for (let k = 0; k < n / 2; k++) {
            let re = 0, im = 0;
            const freq = 2 * Math.PI * k / n;
            for (let i = 0; i < n; i++) {
                const w = this.window[i] || 1;
                re += samples[i] * w * Math.cos(freq * i);
                im -= samples[i] * w * Math.sin(freq * i);
            }
            spectrum[k] = Math.sqrt(re * re + im * im) / n;
        }
        
        return spectrum;
    }
    
    /**
     * 顶级频谱降噪算法
     * 基于 MMSE-STSA（最小均方短时谱幅度估计）
     */
    applySpectralDenoising(samples) {
        const length = samples.length;
        
        // 1. 噪声门限（快速初筛）
        if (this.noiseGateEnabled) {
            this.applyNoiseGate(samples);
        }
        
        // 2. 频谱减法降噪（如果有噪声特征）
        if (this.noiseProfile && this.denoiseStrength > 0) {
            this.applySpectralSubtraction(samples);
        }
        
        // 3. Wiener 滤波平滑
        if (this.wienerFilterEnabled) {
            this.applyWienerFilter(samples);
        }
        
        // 4. 软限幅
        this.applySoftLimiter(samples);
    }
    
    /**
     * 噪声门限
     */
    applyNoiseGate(samples) {
        const threshold = Math.pow(10, this.noiseGateThreshold / 20);
        const attack = 0.001 * this.audioContext.sampleRate;
        const release = 0.03 * this.audioContext.sampleRate;
        const hold = 0.015 * this.audioContext.sampleRate;
        
        let env = this._gateEnvelope || 0;
        let holdCount = this._gateHoldCount || 0;
        
        for (let i = 0; i < samples.length; i++) {
            const abs = Math.abs(samples[i]);
            
            // 包络跟随
            const coef = abs > env ? attack : release;
            env += (abs - env) / coef;
            
            // 门限
            if (env > threshold) {
                holdCount = hold;
            } else if (holdCount > 0) {
                holdCount--;
            } else {
                // 平滑衰减
                const gain = Math.pow(env / threshold, 2);
                samples[i] *= Math.max(0.01, gain);
            }
        }
        
        this._gateEnvelope = env;
        this._gateHoldCount = holdCount;
    }
    
    /**
     * 频谱减法（核心降噪）
     * 优化：更宽松的阈值保护人声，避免断断续续
     */
    applySpectralSubtraction(samples) {
        const blockSize = 512;  // 更大的块减少处理频率
        const numBlocks = Math.floor(samples.length / blockSize);
        
        for (let b = 0; b < numBlocks; b++) {
            const start = b * blockSize;
            const block = samples.slice(start, start + blockSize);
            
            // 计算 RMS
            let rms = 0;
            for (let i = 0; i < block.length; i++) {
                rms += block[i] * block[i];
            }
            rms = Math.sqrt(rms / block.length);
            
            // 与噪声特征比较
            const noiseRMS = this.noiseProfile.rms || 0.01;
            const snr = rms / noiseRMS;
            
            // 计算增益 - 更宽松的阈值保护人声
            let gain;
            if (snr > 1.5) {
                gain = 1;  // 信号明显大于噪声，完全保留
            } else if (snr > 0.8) {
                // 平滑过渡区间更大，避免突变
                gain = (snr - 0.8) / 0.7;
                // 使用更平缓的曲线
                gain = Math.sqrt(gain);
            } else {
                // 底噪保留更多，避免完全静音导致不自然
                gain = Math.max(this.spectralFloor, 0.15);
            }
            
            // 应用降噪强度（限制最大衰减）
            const attenuation = (1 - gain) * this.denoiseStrength * 0.7;  // 最多衰减70%
            gain = 1 - attenuation;
            
            // 平滑应用增益，避免块边界突变
            for (let i = 0; i < blockSize && start + i < samples.length; i++) {
                // 块内使用渐变过渡
                const pos = i / blockSize;
                const smoothGain = pos < 0.1 ? gain * (0.9 + pos) : 
                                   pos > 0.9 ? gain * (1.9 - pos) : gain;
                samples[start + i] *= smoothGain;
            }
        }
    }
    
    /**
     * Wiener 滤波器（帧间平滑，减少突变）
     * 注意：alpha 太低会产生回声，设为 0.85-0.95 比较合适
     */
    applyWienerFilter(samples) {
        // 只做帧间的轻微平滑，不做帧内平滑
        // 这样可以减少块状噪声，同时避免回声
        if (!this._wienerLastSample) {
            this._wienerLastSample = 0;
        }
        
        const alpha = 0.92;  // 高 alpha = 当前帧权重大，回声小
        
        // 只平滑帧的第一个样本与上一帧最后一个样本的过渡
        if (samples.length > 0) {
            samples[0] = alpha * samples[0] + (1 - alpha) * this._wienerLastSample;
            this._wienerLastSample = samples[samples.length - 1];
        }
    }
    
    /**
     * 软限幅（防爆音）
     */
    applySoftLimiter(samples) {
        for (let i = 0; i < samples.length; i++) {
            const x = samples[i];
            if (Math.abs(x) > 0.7) {
                samples[i] = Math.tanh(x * 2) * 0.85;
            }
        }
    }
    
    /**
     * 采样环境噪声
     */
    sampleNoiseFloor() {
        this.noiseProfileFrames = [];
        this._samplingNoise = true;
        this._noiseSampleCount = 0;
        console.log('🎤 开始采样环境噪声... 请保持安静3秒');
        
        setTimeout(() => {
            if (this.noiseProfileFrames && this.noiseProfileFrames.length > 0) {
                // 计算平均噪声特征
                const numFrames = this.noiseProfileFrames.length;
                let totalRMS = 0;
                
                for (const frame of this.noiseProfileFrames) {
                    let rms = 0;
                    for (let i = 0; i < frame.length; i++) {
                        rms += frame[i] * frame[i];
                    }
                    totalRMS += Math.sqrt(rms / frame.length);
                }
                
                this.noiseProfile = {
                    rms: totalRMS / numFrames,
                    frames: this.noiseProfileFrames
                };
                
                const dB = 20 * Math.log10(this.noiseProfile.rms + 0.0001);
                console.log(`✅ 噪声采样完成: ${dB.toFixed(1)} dB, ${numFrames} 帧`);
            }
            this._samplingNoise = false;
        }, 3000);
    }
    
    /**
     * 设置降噪强度
     */
    setDenoiseStrength(value) {
        this.denoiseStrength = Math.max(0, Math.min(2, value));
        console.log(`🔇 降噪强度: ${(this.denoiseStrength * 100).toFixed(0)}%`);
    }
    
    /**
     * 设置噪声门限阈值
     */
    setNoiseGateThreshold(dB) {
        this.noiseGateThreshold = dB;
        console.log(`🔇 噪声门限: ${dB} dB`);
    }
    
    /**
     * 启用/禁用噪声门限
     */
    setNoiseGateEnabled(enabled) {
        this.noiseGateEnabled = enabled;
        console.log(`🔇 噪声门限: ${enabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 启用/禁用 Wiener 滤波
     */
    setWienerFilterEnabled(enabled) {
        this.wienerFilterEnabled = enabled;
        console.log(`🔇 Wiener滤波: ${enabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 调度音频缓冲区播放（简单顺序播放）
     */
    scheduleAudioBuffer(buffer, pts = null) {
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioFilterInput);
        
        const currentTime = this.audioContext.currentTime;
        
        // 简单缓冲管理
        if (this.nextAudioTime < currentTime) {
            // 缓冲耗尽，重新开始
            this.nextAudioTime = currentTime + this.audioBufferAhead;
        } else if (this.nextAudioTime - currentTime > this.audioMaxBuffer) {
            // 缓冲过大，重置
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

export { WebGLVideoRenderer };