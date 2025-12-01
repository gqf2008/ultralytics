/**
 * 音频播放器模块
 * 负责音频滤波器链、播放调度
 */

import { AudioDenoiser } from './audioDenoiser.js';

export class AudioPlayer {
    constructor() {
        this.audioContext = null;
        this.denoiser = null;
        
        // 音频参数
        this.sampleRate = 48000;
        this.channels = 1;
        this.gain = 1.0;
        
        // 播放状态
        this.nextPlayTime = 0;
        this.bufferAhead = 0.2;    // 预缓冲 200ms
        this.maxBuffer = 1.0;      // 最大缓冲 1000ms
        
        // 滤波器节点
        this.highpassFilter = null;
        this.highpassFilter2 = null;
        this.lowpassFilter = null;
        this.lowpassFilter2 = null;
        this.notchFilter = null;
        this.notchFilter2 = null;
        this.compressor = null;
        this.gainNode = null;
        
        // 滤波器链入口/出口
        this.filterInput = null;
        this.filterEnabled = true;
        
        this.init();
    }
    
    init() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log('🔊 AudioPlayer initialized:', this.audioContext.sampleRate, 'Hz');
        
        // 创建降噪器
        this.denoiser = new AudioDenoiser(this.audioContext);
        
        // 创建滤波器链
        this.createFilterChain();
        
        // 音频上下文可能被浏览器挂起
        document.addEventListener('click', () => {
            if (this.audioContext?.state === 'suspended') {
                this.audioContext.resume();
                console.log('🔊 AudioContext resumed');
            }
        }, { once: true });
    }
    
    createFilterChain() {
        const ctx = this.audioContext;
        
        // 高通滤波器：去除低频（默认250Hz）
        this.highpassFilter = ctx.createBiquadFilter();
        this.highpassFilter.type = 'highpass';
        this.highpassFilter.frequency.value = 250;
        this.highpassFilter.Q.value = 1.0;
        
        // 第二个高通：级联 = 24dB/oct
        this.highpassFilter2 = ctx.createBiquadFilter();
        this.highpassFilter2.type = 'highpass';
        this.highpassFilter2.frequency.value = 250;
        this.highpassFilter2.Q.value = 1.0;
        
        // 低通滤波器：去除高频（默认6kHz）
        this.lowpassFilter = ctx.createBiquadFilter();
        this.lowpassFilter.type = 'lowpass';
        this.lowpassFilter.frequency.value = 6000;
        this.lowpassFilter.Q.value = 0.7;
        
        // 第二个低通
        this.lowpassFilter2 = ctx.createBiquadFilter();
        this.lowpassFilter2.type = 'lowpass';
        this.lowpassFilter2.frequency.value = 6000;
        this.lowpassFilter2.Q.value = 0.7;
        
        // 陷波滤波器：消除电源工频干扰 (50Hz)
        this.notchFilter = ctx.createBiquadFilter();
        this.notchFilter.type = 'notch';
        this.notchFilter.frequency.value = 50;
        this.notchFilter.Q.value = 30;
        
        // 第二个陷波：消除谐波 (100Hz)
        this.notchFilter2 = ctx.createBiquadFilter();
        this.notchFilter2.type = 'notch';
        this.notchFilter2.frequency.value = 100;
        this.notchFilter2.Q.value = 30;
        
        // 动态压缩器
        this.compressor = ctx.createDynamicsCompressor();
        this.compressor.threshold.value = -30;
        this.compressor.knee.value = 20;
        this.compressor.ratio.value = 16;
        this.compressor.attack.value = 0.001;
        this.compressor.release.value = 0.1;
        
        // 增益节点
        this.gainNode = ctx.createGain();
        this.gainNode.gain.value = this.gain;
        
        // 连接滤波器链
        this.highpassFilter.connect(this.highpassFilter2);
        this.highpassFilter2.connect(this.notchFilter);
        this.notchFilter.connect(this.notchFilter2);
        this.notchFilter2.connect(this.lowpassFilter);
        this.lowpassFilter.connect(this.lowpassFilter2);
        this.lowpassFilter2.connect(this.compressor);
        this.compressor.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);
        
        // 保存滤波器链入口
        this.filterInput = this.highpassFilter;
        
        console.log('🔇 滤波器链创建完成: HP=250Hz LP=6kHz');
    }
    
    // ==================== 音量控制 ====================
    
    setVolume(value) {
        this.gain = value;
        if (this.gainNode) {
            this.gainNode.gain.value = value;
            console.log(`🔊 音量: ${value.toFixed(1)}x`);
        }
    }
    
    // ==================== 滤波器控制 ====================
    
    setFilterEnabled(enabled) {
        this.filterEnabled = enabled;
        console.log(`🔇 滤波器: ${enabled ? '启用' : '禁用'}`);
    }
    
    setHighpassFrequency(freq) {
        if (this.highpassFilter) {
            this.highpassFilter.frequency.value = freq;
            this.highpassFilter2.frequency.value = freq;
            console.log(`🔇 高通: ${freq}Hz`);
        }
    }
    
    setLowpassFrequency(freq) {
        if (this.lowpassFilter) {
            this.lowpassFilter.frequency.value = freq;
            this.lowpassFilter2.frequency.value = freq;
            console.log(`🔇 低通: ${freq}Hz`);
        }
    }
    
    setNotchFrequency(freq) {
        if (this.notchFilter) {
            if (freq > 0) {
                this.notchFilter.frequency.value = freq;
                this.notchFilter.Q.value = 30;
                this.notchFilter2.frequency.value = freq * 2;
                this.notchFilter2.Q.value = 30;
                console.log(`🔇 陷波: ${freq}Hz + ${freq * 2}Hz`);
            } else {
                this.notchFilter.Q.value = 0.001;
                this.notchFilter2.Q.value = 0.001;
                console.log(`🔇 陷波: 关闭`);
            }
        }
    }
    
    // ==================== 降噪代理方法 ====================
    
    setRnnoiseEnabled(enabled) {
        this.denoiser?.setRnnoiseEnabled(enabled);
    }
    
    setNoiseGateEnabled(enabled) {
        this.denoiser?.setNoiseGateEnabled(enabled);
    }
    
    setNoiseGateThreshold(dB) {
        this.denoiser?.setNoiseGateThreshold(dB);
    }
    
    setWienerFilterEnabled(enabled) {
        this.denoiser?.setWienerFilterEnabled(enabled);
    }
    
    setDenoiseStrength(strength) {
        this.denoiser?.setDenoiseStrength(strength);
    }
    
    sampleNoiseFloor() {
        this.denoiser?.sampleNoiseFloor();
    }
    
    applyNoisePreset(preset) {
        if (!preset) return;
        
        this.setFilterEnabled(preset.enabled);
        
        if (preset.enabled) {
            this.setHighpassFrequency(preset.highpass);
            this.setLowpassFrequency(preset.lowpass);
            this.denoiser?.applyPreset(preset);
        }
        
        console.log(`🔇 预设: HP=${preset.highpass}Hz LP=${preset.lowpass}Hz`);
    }
    
    // ==================== 播放方法 ====================
    
    /**
     * 播放 PCM 音频数据
     * @param {Float32Array} samples - 音频样本 (-1 到 1)
     * @param {number} sampleRate - 采样率
     * @param {number} channels - 声道数
     */
    play(samples, sampleRate = 48000, channels = 1) {
        if (!this.audioContext || samples.length === 0) return;
        
        // 确保音频上下文运行
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        
        // 应用降噪处理
        let processedSamples = this.denoiser ? this.denoiser.process(samples) : samples;
        
        // 软削波防止爆音
        processedSamples = this.applySoftClipping(processedSamples);
        
        // 创建音频缓冲
        const buffer = this.audioContext.createBuffer(channels, processedSamples.length, sampleRate);
        buffer.getChannelData(0).set(processedSamples);
        
        // 创建播放源
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        
        // 连接到滤波器链或直接输出
        if (this.filterEnabled && this.filterInput) {
            source.connect(this.filterInput);
        } else {
            source.connect(this.gainNode);
        }
        
        // 计算播放时间
        const now = this.audioContext.currentTime;
        if (this.nextPlayTime < now) {
            this.nextPlayTime = now + this.bufferAhead;
        }
        
        // 防止缓冲过大
        if (this.nextPlayTime > now + this.maxBuffer) {
            this.nextPlayTime = now + this.bufferAhead;
        }
        
        source.start(this.nextPlayTime);
        this.nextPlayTime += buffer.duration;
    }
    
    /**
     * 软削波，防止爆音
     */
    applySoftClipping(samples) {
        const output = new Float32Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
            // tanh 软削波
            output[i] = Math.tanh(samples[i] * 0.8);
        }
        return output;
    }
    
    /**
     * 重置播放缓冲
     */
    reset() {
        this.nextPlayTime = this.audioContext ? this.audioContext.currentTime + this.bufferAhead : 0;
        console.log('🔄 音频缓冲重置');
    }
    
    /**
     * 释放资源
     */
    destroy() {
        this.denoiser?.destroy();
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        console.log('🔊 AudioPlayer 已释放');
    }
}

export default AudioPlayer;
