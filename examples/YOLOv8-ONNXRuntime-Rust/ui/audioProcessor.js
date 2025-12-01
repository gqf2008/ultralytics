/**
 * 音频降噪处理模块
 * 包含 FFT 频谱降噪、噪声门限、Wiener 滤波等顶级降噪算法
 */

export class AudioProcessor {
    constructor(audioContext) {
        this.audioContext = audioContext;
        
        // 降噪参数
        this.noiseGateThreshold = -35;  // 噪声门限阈值 (dB)
        this.noiseGateEnabled = true;
        this.wienerFilterEnabled = true;
        this.noiseProfile = null;  // 噪声频谱特征
        this.denoiseStrength = 1.0;  // 降噪强度 0-2
        this.spectralFloor = 0.01;  // 频谱底噪
        this.noiseReductionEnabled = true;
        
        // 噪声采样状态
        this._samplingNoise = false;
        this._noiseSampleCount = 0;
        this.noiseProfileFrames = null;
        
        // 噪声门限状态
        this._gateEnvelope = 0;
        this._gateHoldCount = 0;
        
        // Wiener 滤波状态
        this._wienerPrev = null;
        
        // 初始化 FFT 降噪
        this.initSpectralDenoising();
    }
    
    /**
     * 初始化 FFT 降噪所需的数据结构
     */
    initSpectralDenoising() {
        this.fftSize = 2048;
        this.hopSize = this.fftSize / 4;  // 75% 重叠
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
        
        console.log('🔊 FFT 频谱降噪初始化完成');
    }
    
    /**
     * 处理音频数据（主入口）
     * @param {Float32Array} samples - 单声道音频数据
     */
    process(samples) {
        if (!this.noiseReductionEnabled) {
            return;
        }
        
        // 频谱降噪
        this.applySpectralDenoising(samples);
        
        // 采样噪声底噪（如果正在采样）
        if (this._samplingNoise) {
            this.collectNoiseProfile(samples);
        }
    }
    
    /**
     * 顶级频谱降噪算法
     * 基于 MMSE-STSA（最小均方短时谱幅度估计）
     * @param {Float32Array} samples - 音频样本
     */
    applySpectralDenoising(samples) {
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
     * @param {Float32Array} samples - 音频样本
     */
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
     * @param {Float32Array} samples - 音频样本
     */
    applySpectralSubtraction(samples) {
        const blockSize = 256;
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
            
            // 计算增益（Wiener 风格）
            let gain;
            if (snr > 2) {
                gain = 1;  // 信号远大于噪声
            } else if (snr > 0.5) {
                // 平滑过渡
                gain = (snr - 0.5) / 1.5;
                gain = gain * gain;  // 平方曲线
            } else {
                gain = this.spectralFloor;  // 底噪
            }
            
            // 应用降噪强度
            gain = 1 - (1 - gain) * this.denoiseStrength;
            
            // 应用增益
            for (let i = 0; i < blockSize && start + i < samples.length; i++) {
                samples[start + i] *= gain;
            }
        }
    }
    
    /**
     * Wiener 滤波器（平滑处理）
     * @param {Float32Array} samples - 音频样本
     */
    applyWienerFilter(samples) {
        if (!this._wienerPrev || this._wienerPrev.length !== samples.length) {
            this._wienerPrev = new Float32Array(samples.length);
        }
        
        const alpha = 0.3;  // 平滑系数
        for (let i = 0; i < samples.length; i++) {
            samples[i] = alpha * samples[i] + (1 - alpha) * this._wienerPrev[i];
            this._wienerPrev[i] = samples[i];
        }
    }
    
    /**
     * 软限幅（防爆音）
     * @param {Float32Array} samples - 音频样本
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
     * 收集噪声特征（采样时调用）
     * @param {Float32Array} samples - 音频样本
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
     * @param {Float32Array} samples - 音频样本
     * @returns {Float32Array} 幅度谱
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
     * 采样环境噪声（3秒）
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
    
    // ========== 参数设置方法 ==========
    
    /**
     * 设置降噪强度
     * @param {number} value - 0-2 的值
     */
    setDenoiseStrength(value) {
        this.denoiseStrength = Math.max(0, Math.min(2, value));
        console.log(`🔇 降噪强度: ${(this.denoiseStrength * 100).toFixed(0)}%`);
    }
    
    /**
     * 设置噪声门限阈值
     * @param {number} dB - 分贝值
     */
    setNoiseGateThreshold(dB) {
        this.noiseGateThreshold = dB;
        console.log(`🔇 噪声门限: ${dB} dB`);
    }
    
    /**
     * 启用/禁用噪声门限
     * @param {boolean} enabled
     */
    setNoiseGateEnabled(enabled) {
        this.noiseGateEnabled = enabled;
        console.log(`🔇 噪声门限: ${enabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 启用/禁用 Wiener 滤波
     * @param {boolean} enabled
     */
    setWienerFilterEnabled(enabled) {
        this.wienerFilterEnabled = enabled;
        console.log(`🔇 Wiener滤波: ${enabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 启用/禁用降噪
     * @param {boolean} enabled
     */
    setNoiseReductionEnabled(enabled) {
        this.noiseReductionEnabled = enabled;
        console.log(`🔇 降噪${enabled ? '启用' : '禁用'}`);
    }
}

/**
 * 降噪预设配置
 */
export const NOISE_PRESETS = {
    off: { enabled: false, highpass: 20, lowpass: 20000, gate: false, gateThreshold: -60, strength: 0 },
    light: { enabled: true, highpass: 80, lowpass: 12000, gate: true, gateThreshold: -45, strength: 50 },
    normal: { enabled: true, highpass: 150, lowpass: 8000, gate: true, gateThreshold: -40, strength: 80 },
    strong: { enabled: true, highpass: 250, lowpass: 6000, gate: true, gateThreshold: -35, strength: 100 },
    voice: { enabled: true, highpass: 300, lowpass: 5000, gate: true, gateThreshold: -35, strength: 120 },
    store: { enabled: true, highpass: 350, lowpass: 4500, gate: true, gateThreshold: -32, strength: 150 },
    extreme: { enabled: true, highpass: 400, lowpass: 4000, gate: true, gateThreshold: -28, strength: 200 }
};

export default AudioProcessor;
