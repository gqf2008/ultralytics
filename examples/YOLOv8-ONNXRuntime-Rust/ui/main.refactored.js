/**
 * 主入口模块 - 重构版
 * 
 * 模块化结构:
 * - audioProcessor.js    - 音频降噪处理
 * - streamInfoManager.js - 流信息面板
 * - canvasTransform.js   - 画布缩放拖拽
 * - colorGrading.js      - 调色面板
 * - uiControls.js        - UI 控件事件
 * - llmInference.js      - LLM 推理
 * - zeroCopyRenderer.js  - 零拷贝渲染
 * - videoRenderer.js     - 视频渲染器 (本文件)
 */

import { invoke, Channel } from '@tauri-apps/api/core';
import { llmManager } from './llmInference.js';
import { ZeroCopyRenderer } from './zeroCopyRenderer.js';
import { StreamInfoManager } from './streamInfoManager.js';
import { CanvasTransform } from './canvasTransform.js';
import { ColorGrading } from './colorGrading.js';
import { 
    initAudioControls, 
    initControlPanel, 
    initHistoryControls, 
    initZoomControls,
    initLlmControls,
    createShowStatus 
} from './uiControls.js';
import { NOISE_PRESETS } from './audioProcessor.js';

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

// ==================== 全局实例 ====================

let streamInfoManager = null;
let canvasTransform = null;
let colorGrading = null;

// ==================== WebGL 视频渲染器 ====================
// 注意: 此类包含大量视频解码和音频处理逻辑,暂不拆分

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
        this.audioConfigured = false;
        
        // 音频缓冲管理
        this.audioBufferAhead = 0.1;
        this.audioMaxBuffer = 0.5;
        
        // 音视频同步
        this.audioClockBase = 0;
        this.audioClockStart = 0;
        this.audioClockReady = false;
        
        // 帧队列管理
        this.pendingFrames = [];
        this.maxPendingFrames = 2;
        this.lastFrameTime = 0;
        this.lastImageBitmap = null;
        
        // 编解码器信息
        this.currentCodec = null;
        this.extradata = null;
        this.decoderConfigured = false;
        this.decoderHasDescription = false;
        this.waitingForKeyframe = true;
        this.videoChunkCount = 0;
        
        // 调色滤镜
        this.colorFilter = 'brightness(1) contrast(1.2) saturate(1.3)';
        
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.initAudioContext();
    }
    
    // ========== 音频初始化 ==========
    
    initAudioContext() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('🔊 Audio Context initialized:', this.audioContext.sampleRate, 'Hz');
        
        // 降噪参数
        this.noiseGateThreshold = -35;
        this.noiseGateEnabled = true;
        this.wienerFilterEnabled = true;
        this.noiseProfile = null;
        this.denoiseStrength = 1.0;
        this.spectralFloor = 0.01;
        
        this.initSpectralDenoising();
        this.initFilterChain();
        
        // 音频上下文恢复
        document.addEventListener('click', () => {
            if (this.audioContext && this.audioContext.state === 'suspended') {
                this.audioContext.resume();
                console.log('🔊 Audio Context resumed');
            }
        }, { once: true });
    }
    
    initFilterChain() {
        // 高通滤波器 (双重)
        this.highpassFilter = this.audioContext.createBiquadFilter();
        this.highpassFilter.type = 'highpass';
        this.highpassFilter.frequency.value = 250;
        this.highpassFilter.Q.value = 1.0;
        
        this.highpassFilter2 = this.audioContext.createBiquadFilter();
        this.highpassFilter2.type = 'highpass';
        this.highpassFilter2.frequency.value = 250;
        this.highpassFilter2.Q.value = 1.0;
        
        // 低通滤波器 (双重)
        this.lowpassFilter = this.audioContext.createBiquadFilter();
        this.lowpassFilter.type = 'lowpass';
        this.lowpassFilter.frequency.value = 6000;
        this.lowpassFilter.Q.value = 0.7;
        
        this.lowpassFilter2 = this.audioContext.createBiquadFilter();
        this.lowpassFilter2.type = 'lowpass';
        this.lowpassFilter2.frequency.value = 6000;
        this.lowpassFilter2.Q.value = 0.7;
        
        // 陷波滤波器 (50Hz + 100Hz)
        this.notchFilter = this.audioContext.createBiquadFilter();
        this.notchFilter.type = 'notch';
        this.notchFilter.frequency.value = 50;
        this.notchFilter.Q.value = 30;
        
        this.notchFilter2 = this.audioContext.createBiquadFilter();
        this.notchFilter2.type = 'notch';
        this.notchFilter2.frequency.value = 100;
        this.notchFilter2.Q.value = 30;
        
        // 动态压缩器
        this.compressor = this.audioContext.createDynamicsCompressor();
        this.compressor.threshold.value = -35;
        this.compressor.knee.value = 5;
        this.compressor.ratio.value = 12;
        this.compressor.attack.value = 0.001;
        this.compressor.release.value = 0.05;
        
        // 增益节点
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = 1.5;
        
        // 连接滤波器链
        this.highpassFilter.connect(this.highpassFilter2);
        this.highpassFilter2.connect(this.notchFilter);
        this.notchFilter.connect(this.notchFilter2);
        this.notchFilter2.connect(this.lowpassFilter);
        this.lowpassFilter.connect(this.lowpassFilter2);
        this.lowpassFilter2.connect(this.compressor);
        this.compressor.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);
        
        this.audioFilterInput = this.highpassFilter;
        this.audioDirectOutput = this.gainNode;
        this.noiseReductionEnabled = true;
    }
    
    // ========== FFT 频谱降噪 ==========
    
    initSpectralDenoising() {
        this.fftSize = 2048;
        this.hopSize = this.fftSize / 4;
        this.prevPhase = new Float32Array(this.fftSize);
        this.inputBuffer = new Float32Array(this.fftSize);
        this.outputBuffer = new Float32Array(this.fftSize * 2);
        this.outputReadPos = 0;
        this.outputWritePos = 0;
        this.inputWritePos = 0;
        
        this.window = new Float32Array(this.fftSize);
        for (let i = 0; i < this.fftSize; i++) {
            this.window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.fftSize - 1)));
        }
        
        this._gateEnvelope = 0;
        this._gateHoldCount = 0;
        this._wienerPrev = null;
        
        console.log('🔊 FFT 频谱降噪初始化完成');
    }
    
    applySpectralDenoising(samples) {
        if (this.noiseGateEnabled) {
            this.applyNoiseGate(samples);
        }
        if (this.noiseProfile && this.denoiseStrength > 0) {
            this.applySpectralSubtraction(samples);
        }
        if (this.wienerFilterEnabled) {
            this.applyWienerFilter(samples);
        }
        this.applySoftLimiter(samples);
    }
    
    applyNoiseGate(samples) {
        const threshold = Math.pow(10, this.noiseGateThreshold / 20);
        const sampleRate = this.audioContext?.sampleRate || 48000;
        const attack = 0.001 * sampleRate;
        const release = 0.03 * sampleRate;
        const hold = 0.015 * sampleRate;
        
        let env = this._gateEnvelope;
        let holdCount = this._gateHoldCount;
        
        for (let i = 0; i < samples.length; i++) {
            const abs = Math.abs(samples[i]);
            const coef = abs > env ? attack : release;
            env += (abs - env) / coef;
            
            if (env > threshold) {
                holdCount = hold;
            } else if (holdCount > 0) {
                holdCount--;
            } else {
                const gain = Math.pow(env / threshold, 2);
                samples[i] *= Math.max(0.01, gain);
            }
        }
        
        this._gateEnvelope = env;
        this._gateHoldCount = holdCount;
    }
    
    applySpectralSubtraction(samples) {
        const blockSize = 256;
        const numBlocks = Math.floor(samples.length / blockSize);
        
        for (let b = 0; b < numBlocks; b++) {
            const start = b * blockSize;
            const block = samples.slice(start, start + blockSize);
            
            let rms = 0;
            for (let i = 0; i < block.length; i++) {
                rms += block[i] * block[i];
            }
            rms = Math.sqrt(rms / block.length);
            
            const noiseRMS = this.noiseProfile.rms || 0.01;
            const snr = rms / noiseRMS;
            
            let gain;
            if (snr > 2) {
                gain = 1;
            } else if (snr > 0.5) {
                gain = (snr - 0.5) / 1.5;
                gain = gain * gain;
            } else {
                gain = this.spectralFloor;
            }
            
            gain = 1 - (1 - gain) * this.denoiseStrength;
            
            for (let i = 0; i < blockSize && start + i < samples.length; i++) {
                samples[start + i] *= gain;
            }
        }
    }
    
    applyWienerFilter(samples) {
        if (!this._wienerPrev || this._wienerPrev.length !== samples.length) {
            this._wienerPrev = new Float32Array(samples.length);
        }
        
        const alpha = 0.3;
        for (let i = 0; i < samples.length; i++) {
            samples[i] = alpha * samples[i] + (1 - alpha) * this._wienerPrev[i];
            this._wienerPrev[i] = samples[i];
        }
    }
    
    applySoftLimiter(samples) {
        for (let i = 0; i < samples.length; i++) {
            const x = samples[i];
            if (Math.abs(x) > 0.7) {
                samples[i] = Math.tanh(x * 2) * 0.85;
            }
        }
    }
    
    collectNoiseProfile(samples) {
        if (!this.noiseProfileFrames) {
            this.noiseProfileFrames = [];
        }
        const spectrum = this.computeSpectrum(samples);
        this.noiseProfileFrames.push(spectrum);
        this._noiseSampleCount++;
    }
    
    computeSpectrum(samples) {
        const n = Math.min(samples.length, this.fftSize);
        const spectrum = new Float32Array(n / 2);
        
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
    
    sampleNoiseFloor() {
        this.noiseProfileFrames = [];
        this._samplingNoise = true;
        this._noiseSampleCount = 0;
        console.log('🎤 开始采样环境噪声... 请保持安静3秒');
        
        setTimeout(() => {
            if (this.noiseProfileFrames && this.noiseProfileFrames.length > 0) {
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
    
    // ========== 音频控制方法 ==========
    
    setVolume(gain) {
        this.audioGain = gain;
        if (this.gainNode) {
            this.gainNode.gain.value = gain;
            console.log(`🔊 音量设置: ${gain.toFixed(1)}x`);
        }
    }
    
    setNoiseReduction(enabled) {
        this.noiseReductionEnabled = enabled;
        this.audioFilterInput = enabled ? this.highpassFilter : this.gainNode;
        console.log(`🔇 降噪${enabled ? '启用' : '禁用'}`);
    }
    
    setHighpassFrequency(freq) {
        if (this.highpassFilter) {
            this.highpassFilter.frequency.value = freq;
            this.highpassFilter2.frequency.value = freq;
            console.log(`🔇 高通截止: ${freq}Hz`);
        }
    }
    
    setLowpassFrequency(freq) {
        if (this.lowpassFilter) {
            this.lowpassFilter.frequency.value = freq;
            this.lowpassFilter2.frequency.value = freq;
            console.log(`🔇 低通截止: ${freq}Hz`);
        }
    }
    
    setNotchFrequency(freq) {
        if (this.notchFilter) {
            if (freq > 0) {
                this.notchFilter.frequency.value = freq;
                this.notchFilter.Q.value = 30;
                this.notchFilter2.frequency.value = freq * 2;
                this.notchFilter2.Q.value = 30;
            } else {
                this.notchFilter.Q.value = 0.001;
                this.notchFilter2.Q.value = 0.001;
            }
        }
    }
    
    setDenoiseStrength(value) {
        this.denoiseStrength = Math.max(0, Math.min(2, value));
        console.log(`🔇 降噪强度: ${(this.denoiseStrength * 100).toFixed(0)}%`);
    }
    
    setNoiseGateThreshold(dB) {
        this.noiseGateThreshold = dB;
        console.log(`🔇 噪声门限: ${dB} dB`);
    }
    
    setNoiseGateEnabled(enabled) {
        this.noiseGateEnabled = enabled;
        console.log(`🔇 噪声门限: ${enabled ? '启用' : '禁用'}`);
    }
    
    setWienerFilterEnabled(enabled) {
        this.wienerFilterEnabled = enabled;
        console.log(`🔇 Wiener滤波: ${enabled ? '启用' : '禁用'}`);
    }
    
    applyNoisePreset(preset) {
        if (!preset) return;
        
        this.setNoiseReduction(preset.enabled);
        
        if (preset.enabled) {
            this.setHighpassFrequency(preset.highpass);
            this.setLowpassFrequency(preset.lowpass);
            
            if (preset.gate !== undefined) {
                this.setNoiseGateEnabled(preset.gate);
            }
            if (preset.gateThreshold !== undefined) {
                this.setNoiseGateThreshold(preset.gateThreshold);
            }
            if (preset.strength !== undefined) {
                this.setDenoiseStrength(preset.strength / 100);
            }
        }
        
        console.log(`🔇 预设: HP=${preset.highpass}Hz LP=${preset.lowpass}Hz`);
    }
    
    resetAudioBuffer() {
        this.nextAudioTime = this.audioContext ? this.audioContext.currentTime + this.audioBufferAhead : 0;
        this.audioClockBase = 0;
        this.audioClockStart = 0;
        this.audioClockReady = false;
        this._videoSyncBaseSet = false;
    }
    
    getAudioClockTime() {
        if (!this.audioClockReady || !this.audioContext) {
            return null;
        }
        const elapsed = (this.audioContext.currentTime - this.audioClockStart) * 1000;
        return this.audioClockBase + elapsed;
    }
    
    // ========== 视频/音频解码 (保留原有实现) ==========
    // 以下方法保持原有实现，详见原 main.js
    
    resizeCanvas() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        
        this.canvas.width = w;
        this.canvas.height = h;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        
        if (this.lastImageBitmap) {
            this.redrawLastFrame();
        } else if (!this.decoderConfigured && this.pendingFrames.length === 0) {
            this.ctx.fillStyle = '#1a1a1a';
            this.ctx.fillRect(0, 0, w, h);
            this.ctx.fillStyle = '#666';
            this.ctx.font = '20px monospace';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText('WAITING FOR STREAM...', w / 2, h / 2);
        }
    }
    
    redrawLastFrame() {
        if (!this.lastImageBitmap) return;
        
        const bitmap = this.lastImageBitmap;
        this.ctx.filter = this.colorFilter || 'contrast(1.2) saturate(1.3)';
        this.ctx.fillStyle = '#0a0a0f';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
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
        this.ctx.filter = 'none';
    }
    
    start() {
        this.isRunning = true;
    }
    
    stop() {
        this.isRunning = false;
        this.hwDecodeRunning = false;
        if (this.lastImageBitmap) {
            this.lastImageBitmap.close();
            this.lastImageBitmap = null;
        }
        this.audioConfigured = false;
        this.audioCodec = null;
        this.basePts = null;
        this.baseTime = null;
        this.resetAudioBuffer();
    }
    
    // 注意: initDecoder, handleVideoChunk, handleAudioChunk 等方法
    // 由于代码量大，保留在原 main.js 中的完整实现
    // 此文件仅作为模块化示例框架
}

// ==================== 初始化 ====================

const canvas = document.getElementById('canvas');
const renderer = new WebGLVideoRenderer(canvas);

// 创建状态显示函数
const showStatus = createShowStatus();

// DOM 加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    const overlayCanvas = document.getElementById('overlay-canvas');
    
    // 初始化画布变换
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
    
    // 流信息面板开关
    const streamInfoToggle = document.getElementById('stream-info-toggle');
    streamInfoToggle?.addEventListener('change', (e) => {
        const enabled = e.target.checked;
        if (streamInfoManager) {
            streamInfoManager.setEnabled(enabled);
            console.log(`📊 流信息面板: ${enabled ? '启用' : '禁用'}`);
        }
    });
    
    // 初始化 UI 控件
    initAudioControls(renderer);
    initControlPanel();
    initHistoryControls(showStatus);
    initZoomControls(canvasTransform);
    initLlmControls(showStatus);
    
    console.log('✅ 所有模块初始化完成');
});

// 暴露到全局方便调试
window.renderer = renderer;
window.llmManager = llmManager;

console.log('WebGL renderer initialized');
console.log('🔍 缩放功能: 滚轮缩放, 拖拽平移, 双击重置');
console.log('🤖 LLM 推理模块已加载');

// 导出供其他模块使用
export { renderer, streamInfoManager, canvasTransform, colorGrading, showStatus };
