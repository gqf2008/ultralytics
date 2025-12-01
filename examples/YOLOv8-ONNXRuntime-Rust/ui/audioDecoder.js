/**
 * 音频解码器模块
 * 支持 AAC、PCM (A-law/μ-law)、MP3 等格式解码
 */

export class AudioDecoder {
    constructor(onDecodedCallback) {
        this.decoder = null;
        this.codec = null;
        this.sampleRate = 48000;
        this.channels = 1;
        this.configured = false;
        this.description = null;
        
        // 解码后的回调
        this.onDecoded = onDecodedCallback;
        
        // PCM G.711 解码表
        this.alawTable = this.buildAlawTable();
        this.ulawTable = this.buildUlawTable();
    }
    
    /**
     * 初始化/配置解码器
     * @param {string} codec - 编码格式 ('aac', 'pcm_alaw', 'pcm_mulaw', 'mp3')
     * @param {number} sampleRate - 采样率
     * @param {number} channels - 声道数
     * @param {Uint8Array} description - 解码器配置数据 (AAC 的 ASC)
     */
    init(codec, sampleRate = 48000, channels = 1, description = null) {
        // 检查是否需要重新配置
        if (this.configured && 
            this.codec === codec && 
            this.sampleRate === sampleRate && 
            this.channels === channels) {
            return;
        }
        
        this.codec = codec;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.description = description;
        
        // PCM 格式不需要 WebCodecs 解码器
        if (codec === 'pcm_alaw' || codec === 'pcm_mulaw') {
            this.configured = true;
            console.log(`🎵 PCM 解码器配置: ${codec} ${sampleRate}Hz ${channels}ch`);
            return;
        }
        
        // 关闭之前的解码器
        if (this.decoder && this.decoder.state !== 'closed') {
            this.decoder.close();
        }
        
        // 创建 WebCodecs 音频解码器
        if (!('AudioDecoder' in window)) {
            console.error('❌ WebCodecs AudioDecoder 不支持');
            return;
        }
        
        this.decoder = new window.AudioDecoder({
            output: (audioData) => this.handleDecodedData(audioData),
            error: (e) => console.error('❌ 音频解码错误:', e.message)
        });
        
        // 配置解码器
        const config = {
            codec: this.getWebCodecsCodec(codec),
            sampleRate: sampleRate,
            numberOfChannels: channels
        };
        
        if (description && description.length > 0) {
            config.description = description;
        }
        
        try {
            this.decoder.configure(config);
            this.configured = true;
            console.log(`🎵 音频解码器配置: ${codec} -> ${config.codec} ${sampleRate}Hz ${channels}ch`);
        } catch (e) {
            console.error('❌ 音频解码器配置失败:', e);
            this.configured = false;
        }
    }
    
    /**
     * 转换为 WebCodecs 支持的 codec 字符串
     */
    getWebCodecsCodec(codec) {
        const mapping = {
            'aac': 'mp4a.40.2',
            'mp3': 'mp3',
            'opus': 'opus',
            'vorbis': 'vorbis',
            'flac': 'flac'
        };
        return mapping[codec] || codec;
    }
    
    /**
     * 解码音频数据
     * @param {Uint8Array} data - 编码的音频数据
     * @param {number} timestamp - 时间戳 (微秒)
     */
    decode(data, timestamp = 0) {
        if (!this.configured) {
            console.warn('⚠️ 音频解码器未配置');
            return;
        }
        
        // PCM G.711 直接解码
        if (this.codec === 'pcm_alaw') {
            const samples = this.decodeAlaw(data);
            this.onDecoded?.(samples, this.sampleRate, this.channels, timestamp);
            return;
        }
        
        if (this.codec === 'pcm_mulaw') {
            const samples = this.decodeUlaw(data);
            this.onDecoded?.(samples, this.sampleRate, this.channels, timestamp);
            return;
        }
        
        // WebCodecs 解码
        if (this.decoder && this.decoder.state === 'configured') {
            const chunk = new EncodedAudioChunk({
                type: 'key',
                timestamp: timestamp,
                data: data
            });
            this.decoder.decode(chunk);
        }
    }
    
    /**
     * 处理 WebCodecs 解码后的数据
     */
    handleDecodedData(audioData) {
        const samples = new Float32Array(audioData.numberOfFrames * audioData.numberOfChannels);
        audioData.copyTo(samples, { planeIndex: 0 });
        
        this.onDecoded?.(samples, audioData.sampleRate, audioData.numberOfChannels, audioData.timestamp);
        
        audioData.close();
    }
    
    // ==================== G.711 解码 ====================
    
    buildAlawTable() {
        const table = new Int16Array(256);
        for (let i = 0; i < 256; i++) {
            let val = i ^ 0x55;
            let t = (val & 0x0F) << 4;
            let seg = (val & 0x70) >> 4;
            
            if (seg > 0) {
                t += 0x100;
                t <<= (seg - 1);
            } else {
                t += 8;
            }
            
            table[i] = (val & 0x80) ? t : -t;
        }
        return table;
    }
    
    buildUlawTable() {
        const table = new Int16Array(256);
        const BIAS = 33;
        
        for (let i = 0; i < 256; i++) {
            let val = ~i;
            let t = ((val & 0x0F) << 3) + BIAS;
            t <<= (val & 0x70) >> 4;
            
            table[i] = (val & 0x80) ? (BIAS - t) : (t - BIAS);
        }
        return table;
    }
    
    decodeAlaw(data) {
        const samples = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
            samples[i] = this.alawTable[data[i]] / 32768.0;
        }
        return samples;
    }
    
    decodeUlaw(data) {
        const samples = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) {
            samples[i] = this.ulawTable[data[i]] / 32768.0;
        }
        return samples;
    }
    
    // ==================== 状态管理 ====================
    
    reset() {
        if (this.decoder && this.decoder.state === 'configured') {
            this.decoder.reset();
        }
        this.configured = false;
        console.log('🔄 音频解码器重置');
    }
    
    destroy() {
        if (this.decoder && this.decoder.state !== 'closed') {
            this.decoder.close();
        }
        this.decoder = null;
        this.configured = false;
        console.log('🎵 AudioDecoder 已释放');
    }
}

export default AudioDecoder;
