/**
 * 音频降噪模块
 * 包含 RNNoise AI 降噪、Wiener 滤波、频谱降噪等算法
 */

import { Rnnoise } from '@shiguredo/rnnoise-wasm';

export class AudioDenoiser {
    constructor(audioContext) {
        this.audioContext = audioContext;
        
        // ========== 降噪参数 ==========
        this.noiseGateThreshold = -35;  // 噪声门限阈值 (dB)
        this.noiseGateEnabled = true;
        this.wienerFilterEnabled = false;  // 默认关闭，容易产生回声
        this.noiseProfile = null;  // 噪声频谱特征
        this.denoiseStrength = 1.0;  // 降噪强度 0-2
        this.spectralFloor = 0.01;  // 频谱底噪
        
        // ========== RNNoise AI 降噪 ==========
        this.rnnoiseEnabled = false;
        this.rnnoiseReady = false;
        this.rnnoiseInstance = null;
        this.rnnoiseState = null;
        this.rnnoiseFrameSize = 480;
        this.rnnoiseBuffer = null;
        this.rnnoiseBufferPos = 0;
        this.rnnoiseOutputBuffer = [];
        
        // ========== 噪声采样状态 ==========
        this._samplingNoise = false;
        this._noiseSampleCount = 0;
        this.noiseProfileFrames = null;
        
        // ========== Wiener 滤波状态 ==========
        this._wienerPrev = null;
        
        // 初始化
        this.initRnnoise();
        this.initSpectralDenoising();
    }
    
    // ==================== RNNoise ====================
    
    async initRnnoise() {
        try {
            console.log('🤖 正在加载 RNNoise WASM...');
            this.rnnoiseInstance = await Rnnoise.load();
            this.rnnoiseState = this.rnnoiseInstance.createDenoiseState();
            this.rnnoiseFrameSize = this.rnnoiseInstance.frameSize;
            this.rnnoiseBuffer = new Float32Array(this.rnnoiseFrameSize);
            this.rnnoiseBufferPos = 0;
            this.rnnoiseReady = true;
            console.log(`✅ RNNoise 加载完成，帧大小: ${this.rnnoiseFrameSize}`);
        } catch (e) {
            console.error('❌ RNNoise 加载失败:', e);
            this.rnnoiseReady = false;
        }
    }
    
    setRnnoiseEnabled(enabled) {
        this.rnnoiseEnabled = enabled && this.rnnoiseReady;
        console.log(`🤖 RNNoise: ${this.rnnoiseEnabled ? '启用' : '禁用'}`);
    }
    
    /**
     * RNNoise 处理音频数据
     * @param {Float32Array} samples - 输入音频样本 (-1 到 1)
     * @returns {Float32Array} 降噪后的音频
     */
    processWithRnnoise(samples) {
        if (!this.rnnoiseEnabled || !this.rnnoiseReady || !this.rnnoiseState) {
            return samples;
        }
        
        const output = new Float32Array(samples.length);
        let outPos = 0;
        
        // 先输出之前缓冲的结果
        while (this.rnnoiseOutputBuffer.length > 0 && outPos < output.length) {
            output[outPos++] = this.rnnoiseOutputBuffer.shift();
        }
        
        // 处理新样本
        for (let i = 0; i < samples.length && outPos < output.length; i++) {
            // 转换为 RNNoise 期望的范围 (-32768 到 32767)
            this.rnnoiseBuffer[this.rnnoiseBufferPos++] = samples[i] * 32768;
            
            if (this.rnnoiseBufferPos >= this.rnnoiseFrameSize) {
                // 处理一帧
                this.rnnoiseState.processFrame(this.rnnoiseBuffer);
                
                // 转换回 -1 到 1 范围并输出
                for (let j = 0; j < this.rnnoiseFrameSize; j++) {
                    const sample = this.rnnoiseBuffer[j] / 32768;
                    if (outPos < output.length) {
                        output[outPos++] = sample;
                    } else {
                        this.rnnoiseOutputBuffer.push(sample);
                    }
                }
                this.rnnoiseBufferPos = 0;
            }
        }
        
        return output;
    }
    
    // ==================== 频谱降噪 ====================
    
    initSpectralDenoising() {
        this.fftSize = 2048;
        this.hopSize = this.fftSize / 4;
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
    }
    
    /**
     * 采样环境噪声
     */
    sampleNoiseFloor() {
        this._samplingNoise = true;
        this._noiseSampleCount = 0;
        this.noiseProfileFrames = [];
        console.log('🎤 开始采样环境噪声... (3秒)');
        
        setTimeout(() => {
            this._samplingNoise = false;
            if (this.noiseProfileFrames && this.noiseProfileFrames.length > 0) {
                // 计算平均噪声频谱
                const len = this.noiseProfileFrames[0].length;
                this.noiseProfile = new Float32Array(len);
                for (const frame of this.noiseProfileFrames) {
                    for (let i = 0; i < len; i++) {
                        this.noiseProfile[i] += frame[i];
                    }
                }
                for (let i = 0; i < len; i++) {
                    this.noiseProfile[i] /= this.noiseProfileFrames.length;
                }
                console.log(`✅ 噪声采样完成，收集了 ${this.noiseProfileFrames.length} 帧`);
            }
            this.noiseProfileFrames = null;
        }, 3000);
    }
    
    // ==================== 噪声门限 ====================
    
    setNoiseGateEnabled(enabled) {
        this.noiseGateEnabled = enabled;
        console.log(`🔇 噪声门限: ${enabled ? '启用' : '禁用'}`);
    }
    
    setNoiseGateThreshold(dB) {
        this.noiseGateThreshold = dB;
        console.log(`🔇 噪声门限阈值: ${dB}dB`);
    }
    
    /**
     * 应用噪声门限
     */
    applyNoiseGate(samples) {
        if (!this.noiseGateEnabled) return samples;
        
        const threshold = Math.pow(10, this.noiseGateThreshold / 20);
        const attack = 0.001;
        const release = 0.05;
        const holdTime = 0.02;
        const holdSamples = Math.floor(holdTime * (this.audioContext?.sampleRate || 48000));
        
        const output = new Float32Array(samples.length);
        let envelope = this._gateEnvelope || 0;
        let holdCount = this._gateHoldCount || 0;
        
        for (let i = 0; i < samples.length; i++) {
            const level = Math.abs(samples[i]);
            
            if (level > threshold) {
                envelope = Math.min(1, envelope + attack);
                holdCount = holdSamples;
            } else if (holdCount > 0) {
                holdCount--;
            } else {
                envelope = Math.max(0, envelope - release);
            }
            
            output[i] = samples[i] * envelope;
        }
        
        this._gateEnvelope = envelope;
        this._gateHoldCount = holdCount;
        
        return output;
    }
    
    // ==================== Wiener 滤波 ====================
    
    setWienerFilterEnabled(enabled) {
        this.wienerFilterEnabled = enabled;
        console.log(`🔇 Wiener滤波: ${enabled ? '启用' : '禁用'}`);
    }
    
    setDenoiseStrength(strength) {
        this.denoiseStrength = Math.max(0, Math.min(2, strength));
        console.log(`🔇 降噪强度: ${(this.denoiseStrength * 100).toFixed(0)}%`);
    }
    
    /**
     * 应用 Wiener 滤波（简化版）
     */
    applyWienerFilter(samples) {
        if (!this.wienerFilterEnabled || !this.noiseProfile) return samples;
        
        const output = new Float32Array(samples.length);
        const alpha = 0.95;  // 平滑因子
        
        if (!this._wienerPrev || this._wienerPrev.length !== samples.length) {
            this._wienerPrev = new Float32Array(samples.length);
        }
        
        for (let i = 0; i < samples.length; i++) {
            // 简单的谱减法
            const noiseEst = this.noiseProfile[i % this.noiseProfile.length] * this.denoiseStrength;
            const signalPower = samples[i] * samples[i];
            const noisePower = noiseEst * noiseEst;
            
            let gain = Math.sqrt(Math.max(0, signalPower - noisePower) / (signalPower + 1e-10));
            gain = alpha * (this._wienerPrev[i] || gain) + (1 - alpha) * gain;
            gain = Math.max(this.spectralFloor, Math.min(1, gain));
            
            this._wienerPrev[i] = gain;
            output[i] = samples[i] * gain;
        }
        
        return output;
    }
    
    // ==================== 综合处理 ====================
    
    /**
     * 对音频数据进行完整的降噪处理
     * @param {Float32Array} samples - 输入音频样本
     * @returns {Float32Array} 处理后的音频
     */
    process(samples) {
        let output = samples;
        
        // 1. RNNoise AI 降噪（如果启用）
        if (this.rnnoiseEnabled) {
            output = this.processWithRnnoise(output);
        }
        
        // 2. 噪声门限
        if (this.noiseGateEnabled) {
            output = this.applyNoiseGate(output);
        }
        
        // 3. Wiener 滤波（如果启用且有噪声采样）
        if (this.wienerFilterEnabled && this.noiseProfile) {
            output = this.applyWienerFilter(output);
        }
        
        // 4. 采样噪声（如果正在采样）
        if (this._samplingNoise && this.noiseProfileFrames) {
            // 计算当前帧的频谱
            const spectrum = new Float32Array(samples.length);
            for (let i = 0; i < samples.length; i++) {
                spectrum[i] = Math.abs(samples[i]);
            }
            this.noiseProfileFrames.push(spectrum);
            this._noiseSampleCount++;
        }
        
        return output;
    }
    
    /**
     * 应用预设配置
     */
    applyPreset(preset) {
        if (!preset) return;
        
        if (preset.gate !== undefined) {
            this.setNoiseGateEnabled(preset.gate);
        }
        if (preset.gateThreshold !== undefined) {
            this.setNoiseGateThreshold(preset.gateThreshold);
        }
        if (preset.strength !== undefined) {
            this.setDenoiseStrength(preset.strength / 100);
        }
        
        console.log(`🔇 降噪预设应用: 门限=${preset.gateThreshold}dB 强度=${preset.strength}%`);
    }
    
    /**
     * 释放资源
     */
    destroy() {
        if (this.rnnoiseState) {
            this.rnnoiseState.destroy();
            this.rnnoiseState = null;
        }
        this.rnnoiseInstance = null;
        this.rnnoiseReady = false;
        console.log('🔇 AudioDenoiser 已释放');
    }
}

export default AudioDenoiser;
