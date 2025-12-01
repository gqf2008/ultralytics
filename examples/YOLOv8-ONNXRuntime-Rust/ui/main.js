/**
 * RTSP/HTTP-FLV 视频监控 - 主入口文件
 * 模块化重构版本
 */

import { invoke, Channel } from '@tauri-apps/api/core';
import { setupLogging } from './logger.js';
import { StreamInfoManager } from './streamInfo.js';
import { WebGLVideoRenderer, setStreamInfoManager } from './videoRenderer.js';
import { CanvasTransform } from './canvasTransformNew.js';
import { ColorGrading } from './colorGradingNew.js';
import { llmManager, defaultLlmConfig } from './llmInference.js';

// ==================== 初始化 ====================

// 设置日志
setupLogging();

// 前端加载完成后显示窗口
invoke('show_window').catch(console.error);

// 全局实例
let streamInfoManager = null;
let renderer = null;
let canvasTransform = null;
let colorGrading = null;

// 创建渲染器
const canvas = document.getElementById('canvas');
renderer = new WebGLVideoRenderer(canvas);

// ==================== DOM 元素 ====================

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

// ==================== 工具函数 ====================

function showStatus(message, isError = false, duration = 5000) {
    statusDiv.textContent = message;
    statusDiv.classList.add('show');
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
    }, isError ? 8000 : duration);
}

// ==================== 历史记录 ====================

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

// ==================== 拖拽功能 ====================

let isDragging = false;
let currentX, currentY, initialX, initialY;
let xOffset = 0, yOffset = 0;

function dragStart(e) {
    if (e.target.closest('#toggle-btn') || e.target.closest('#history-toggle')) return;
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

function dragEnd() {
    isDragging = false;
    controlPanel.style.transition = 'background 0.3s ease, border-color 0.3s ease';
}

// ==================== 流媒体控制 ====================

async function startStream() {
    const url = rtspInput.value.trim();
    if (!url) {
        showStatus('❌ 请输入 RTSP 地址');
        return;
    }
    
    try {
        startBtn.disabled = true;
        showStatus('🚀 正在启动 (WebCodecs 解码)...');
        
        const onData = new Channel();
        let packetCount = 0;
        let lastMessageTime = Date.now();
        
        // 心跳检测：每5秒检查是否还在收到数据
        const heartbeatInterval = setInterval(() => {
            const elapsed = Date.now() - lastMessageTime;
            if (elapsed > 10000) {
                console.warn(`⚠️ 已 ${(elapsed / 1000).toFixed(1)}s 未收到数据`);
            }
        }, 5000);
        
        // 保存 interval 以便停止时清除
        window._heartbeatInterval = heartbeatInterval;
        
        onData.onmessage = (response) => {
            try {
                lastMessageTime = Date.now();
                
                if (packetCount < 5) {
                    console.log(`📨 收到数据 #${packetCount}:`, typeof response);
                }
                packetCount++;
                
                // 解析 JSON 消息
                let message = null;
                if (typeof response === 'string') {
                    try { message = JSON.parse(response); } catch (e) { }
                } else if (typeof response === 'object' && response !== null && !(response instanceof ArrayBuffer)) {
                    message = response;
                }
                
                if (message && message.type) {
                    handleJsonMessage(message);
                } else if (response instanceof ArrayBuffer) {
                    handleBinaryPacket(new Uint8Array(response));
                }
            } catch (err) {
                console.error('❌ onmessage 处理异常:', err);
            }
        };
        
        renderer.start();
        if (streamInfoManager) streamInfoManager.reset();
        
        await invoke('start_stream', { url, onData });
        showStatus('✅ 监控已启动 (WebCodecs)');
        
        try {
            await invoke('add_stream_history', { url });
        } catch (e) {
            console.warn('保存历史记录失败:', e);
        }
        
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        rtspInput.disabled = true;
    } catch (err) {
        console.error('启动失败:', err);
        showStatus('❌ 启动失败: ' + err, true);
        startBtn.disabled = false;
    }
}

function handleJsonMessage(message) {
    console.log('📦 收到消息:', message.type);
    
    switch (message.type) {
        case 'video_config': {
            const { codec, width, height, extradata } = message;
            console.log(`🎬 [Video Config] ${codec.toUpperCase()} ${width}x${height}`);
            const extradataBytes = extradata?.length > 0 ? new Uint8Array(extradata) : null;
            renderer.initDecoder(codec, width, height, extradataBytes);
            break;
        }
        case 'audio_config': {
            const { audio_codec, audio_sample_rate, audio_channels, description } = message;
            console.log(`🎵 [Audio Config] ${audio_codec} ${audio_sample_rate}Hz ${audio_channels}ch`);
            const frontendCodec = convertAudioCodec(audio_codec);
            const descBuffer = description?.length > 0 ? new Uint8Array(description) : null;
            renderer.initAudioDecoder(frontendCodec, audio_sample_rate, audio_channels, descBuffer);
            renderer.audioConfigured = true;
            if (streamInfoManager) {
                streamInfoManager.updateAudioInfo(audio_codec, audio_sample_rate, audio_channels);
            }
            break;
        }
        case 'stream_info':
            if (streamInfoManager) {
                streamInfoManager.updateStreamInfo(message);
                streamInfoManager.show();
            }
            break;
        case 'error':
            console.error('❌ 后端错误:', message.error);
            showStatus(`❌ ${message.error}`, true);
            if (streamInfoManager) streamInfoManager.showError(message.error);
            break;
    }
}

function handleBinaryPacket(data) {
    if (data.length < 2) return;
    
    const packetType = data[0];
    
    try {
        if (packetType === 1) {
            // 视频包
            if (data.length < 30) return;
            const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
            const isKeyframe = data[1] === 1;
            const pts = Number(view.getBigInt64(2, true));
            const dataLen = view.getUint32(26, true);
            const encodedData = data.slice(30, 30 + dataLen);
            
            renderer.handleVideoChunk(encodedData, isKeyframe, pts);
            if (streamInfoManager) streamInfoManager.addPacket(encodedData.length, isKeyframe);
        } else if (packetType === 2) {
            // 音频包
            const codecLen = data[1];
            let pos = 2;
            const codec = new TextDecoder().decode(data.slice(pos, pos + codecLen));
            pos += codecLen;
            
            const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
            const sampleRate = view.getUint32(pos, true); pos += 4;
            const channels = data[pos]; pos += 1;
            const pts = Number(view.getBigInt64(pos, true)); pos += 8;
            const audioDataLen = view.getUint32(pos, true); pos += 4;
            const audioData = data.slice(pos, pos + audioDataLen);
            
            const frontendCodec = convertAudioCodec(codec);
            renderer.initAudioDecoder(frontendCodec, sampleRate, channels);
            renderer.handleAudioChunk(audioData, pts);
        }
    } catch (e) {
        console.error('❌ 处理数据包出错:', e);
    }
}

function convertAudioCodec(codec) {
    const mapping = {
        'mp4a.40.2': 'aac',
        'pcm_alaw': 'pcm_alaw',
        'pcm_mulaw': 'pcm_mulaw',
        'mp3': 'mp3'
    };
    return mapping[codec] || codec;
}

async function stopStream() {
    try {
        // 清除心跳检测
        if (window._heartbeatInterval) {
            clearInterval(window._heartbeatInterval);
            window._heartbeatInterval = null;
        }
        
        renderer.stop();
        await invoke('stop_stream');
        if (streamInfoManager) streamInfoManager.hide();
        
        stopBtn.classList.add('hidden');
        startBtn.classList.remove('hidden');
        startBtn.disabled = false;
        rtspInput.disabled = false;
        showStatus('⏹ 监控已停止');
    } catch (e) {
        console.error('停止失败:', e);
        showStatus('❌ 停止失败: ' + e, true);
    }
}

// ==================== 降噪控制 ====================

const NOISE_PRESETS = {
    off: { enabled: false, highpass: 20, lowpass: 20000, gate: false, gateThreshold: -60, strength: 0 },
    light: { enabled: true, highpass: 80, lowpass: 12000, gate: true, gateThreshold: -50, strength: 30 },
    normal: { enabled: true, highpass: 120, lowpass: 10000, gate: true, gateThreshold: -45, strength: 50 },
    strong: { enabled: true, highpass: 180, lowpass: 8000, gate: true, gateThreshold: -40, strength: 70 },
    voice: { enabled: true, highpass: 200, lowpass: 7000, gate: true, gateThreshold: -38, strength: 80 },
    store: { enabled: true, highpass: 250, lowpass: 6000, gate: true, gateThreshold: -35, strength: 100 },
    extreme: { enabled: true, highpass: 300, lowpass: 5000, gate: true, gateThreshold: -30, strength: 150 }
};

function setupNoiseControls() {
    const noisePreset = document.getElementById('noise-preset');
    const highpassSlider = document.getElementById('highpass-slider');
    const highpassValue = document.getElementById('highpass-value');
    const lowpassSlider = document.getElementById('lowpass-slider');
    const lowpassValue = document.getElementById('lowpass-value');
    const noiseGateToggle = document.getElementById('noise-gate-toggle');
    const noiseGateSlider = document.getElementById('noise-gate-slider');
    const noiseGateValue = document.getElementById('noise-gate-value');
    const sampleNoiseBtn = document.getElementById('sample-noise-btn');
    const denoiseStrengthSlider = document.getElementById('denoise-strength-slider');
    const denoiseStrengthValue = document.getElementById('denoise-strength-value');
    const wienerToggle = document.getElementById('wiener-toggle');
    const noiseStatus = document.getElementById('noise-status');
    const rnnoiseToggle = document.getElementById('rnnoise-toggle');
    const rnnoiseStatus = document.getElementById('rnnoise-status');
    
    // RNNoise 开关
    rnnoiseToggle?.addEventListener('change', (e) => {
        if (renderer) {
            renderer.setRnnoiseEnabled(e.target.checked);
            if (rnnoiseStatus) {
                rnnoiseStatus.textContent = e.target.checked 
                    ? '🤖 RNNoise AI 降噪已启用'
                    : '🤖 深度学习降噪 (无需采样，效果更自然)';
                rnnoiseStatus.style.color = e.target.checked 
                    ? 'rgba(100,255,150,0.9)'
                    : 'rgba(100,200,255,0.8)';
            }
        }
    });
    
    // 降噪预设
    noisePreset?.addEventListener('change', (e) => {
        const preset = NOISE_PRESETS[e.target.value];
        if (renderer && preset) {
            renderer.applyNoisePreset(preset);
            if (highpassSlider) { highpassSlider.value = preset.highpass; highpassValue.textContent = preset.highpass + 'Hz'; }
            if (lowpassSlider) { lowpassSlider.value = preset.lowpass; lowpassValue.textContent = (preset.lowpass / 1000).toFixed(1) + 'kHz'; }
            if (noiseGateToggle) noiseGateToggle.checked = preset.gate;
            if (noiseGateSlider) { noiseGateSlider.value = preset.gateThreshold; noiseGateValue.textContent = preset.gateThreshold + 'dB'; }
            if (denoiseStrengthSlider) { denoiseStrengthSlider.value = preset.strength; denoiseStrengthValue.textContent = preset.strength + '%'; }
        }
    });
    
    // 降噪强度
    denoiseStrengthSlider?.addEventListener('input', (e) => {
        const strength = parseInt(e.target.value);
        if (renderer) renderer.setDenoiseStrength(strength / 100);
        if (denoiseStrengthValue) denoiseStrengthValue.textContent = strength + '%';
    });
    
    // Wiener 滤波
    wienerToggle?.addEventListener('change', (e) => {
        if (renderer) renderer.setWienerFilterEnabled(e.target.checked);
    });
    
    // 噪声门限
    noiseGateToggle?.addEventListener('change', (e) => {
        if (renderer) renderer.setNoiseGateEnabled(e.target.checked);
    });
    
    noiseGateSlider?.addEventListener('input', (e) => {
        const dB = parseInt(e.target.value);
        if (renderer) renderer.setNoiseGateThreshold(dB);
        if (noiseGateValue) noiseGateValue.textContent = dB + 'dB';
    });
    
    // 采样噪声
    sampleNoiseBtn?.addEventListener('click', () => {
        if (renderer) {
            renderer.sampleNoiseFloor();
            sampleNoiseBtn.textContent = '🎤 采样中... 请安静';
            sampleNoiseBtn.disabled = true;
            if (noiseStatus) {
                noiseStatus.textContent = '⏳ 正在采样环境噪声...';
                noiseStatus.style.color = 'rgba(251,191,36,0.9)';
            }
            setTimeout(() => {
                sampleNoiseBtn.textContent = '🎤 采样环境噪声 (必须！)';
                sampleNoiseBtn.disabled = false;
                if (noiseStatus) {
                    noiseStatus.textContent = '✅ 噪声采样完成！降噪已激活';
                    noiseStatus.style.color = 'rgba(74,222,128,0.9)';
                }
            }, 3500);
        }
    });
    
    // 高通/低通
    highpassSlider?.addEventListener('input', (e) => {
        const freq = parseInt(e.target.value);
        if (renderer) renderer.setHighpassFrequency(freq);
        if (highpassValue) highpassValue.textContent = freq + 'Hz';
    });
    
    lowpassSlider?.addEventListener('input', (e) => {
        const freq = parseInt(e.target.value);
        if (renderer) renderer.setLowpassFrequency(freq);
        if (lowpassValue) lowpassValue.textContent = (freq / 1000).toFixed(1) + 'kHz';
    });
    
    // 陷波滤波
    document.getElementById('notch-select')?.addEventListener('change', (e) => {
        if (renderer) renderer.setNotchFrequency(parseInt(e.target.value));
    });
}

// ==================== LLM 推理 ====================

function setupLlmControls() {
    const llmOutput = document.getElementById('llm-output');
    const llmOutputContainer = document.getElementById('llm-output-container');
    const startLlmBtn = document.getElementById('start-llm-btn');
    const stopLlmBtn = document.getElementById('stop-llm-btn');
    const llmConfigToggle = document.getElementById('llm-config-toggle');
    const llmConfigPanel = document.getElementById('llm-config-panel');
    const llmClearOutput = document.getElementById('llm-clear-output');
    
    function appendLlmOutput(text) {
        if (llmOutput) {
            llmOutput.textContent += text;
            llmOutput.scrollTop = llmOutput.scrollHeight;
        }
    }
    
    // 配置面板切换
    llmConfigToggle?.addEventListener('click', () => {
        llmConfigPanel?.classList.toggle('hidden');
        llmConfigToggle.textContent = llmConfigPanel?.classList.contains('hidden') ? '配置' : '收起';
    });
    
    // 清空输出
    llmClearOutput?.addEventListener('click', () => {
        if (llmOutput) llmOutput.textContent = '';
    });
    
    // 设置回调
    llmManager.onFrameInfo = (chunk) => {
        appendLlmOutput(`\n\n━━━ 📷 帧 #${chunk.frame_id} (${new Date().toLocaleTimeString()}) ━━━\n`);
    };
    llmManager.onChunk = (chunk) => appendLlmOutput(chunk.content);
    llmManager.onEnd = () => appendLlmOutput('\n');
    llmManager.onError = (chunk) => {
        appendLlmOutput(`\n❌ 错误: ${chunk.content}\n`);
        showStatus('❌ LLM 推理错误: ' + chunk.content, true);
    };
    
    // 启动推理
    startLlmBtn?.addEventListener('click', async () => {
        const url = rtspInput.value.trim();
        if (!url) {
            showStatus('❌ 请先输入 RTSP 地址');
            return;
        }
        
        llmManager.setConfig({
            model: document.getElementById('llm-model')?.value || 'llava',
            api_url: document.getElementById('llm-api-url')?.value || 'http://localhost:11434/api/chat',
            system_prompt: document.getElementById('llm-system-prompt')?.value || '',
            user_prompt: document.getElementById('llm-user-prompt')?.value || '描述这张监控画面中的场景和活动',
            frame_interval: parseFloat(document.getElementById('llm-frame-interval')?.value) || 2.0,
            temperature: parseFloat(document.getElementById('llm-temperature')?.value) || 0.7
        });
        
        try {
            startLlmBtn.disabled = true;
            llmOutputContainer?.classList.remove('hidden');
            if (llmOutput) llmOutput.textContent = '🚀 启动 LLM 推理...\n';
            
            await llmManager.start(url);
            
            startLlmBtn.classList.add('hidden');
            stopLlmBtn.classList.remove('hidden');
            startLlmBtn.disabled = false;
            showStatus('🤖 LLM 推理已启动');
            appendLlmOutput('✅ 已连接，等待视频帧...\n');
        } catch (e) {
            console.error('启动 LLM 推理失败:', e);
            showStatus('❌ LLM 推理启动失败: ' + e, true);
            startLlmBtn.disabled = false;
        }
    });
    
    // 停止推理
    stopLlmBtn?.addEventListener('click', async () => {
        try {
            await llmManager.stop();
            stopLlmBtn.classList.add('hidden');
            startLlmBtn.classList.remove('hidden');
            showStatus('⏹ LLM 推理已停止');
            appendLlmOutput('\n\n🛑 LLM 推理已停止\n');
        } catch (e) {
            console.error('停止 LLM 推理失败:', e);
            showStatus('❌ 停止失败: ' + e, true);
        }
    });
}

// ==================== 事件绑定 ====================

function setupEventListeners() {
    // 音量控制
    volumeSlider?.addEventListener('input', (e) => {
        const gain = parseFloat(e.target.value);
        if (renderer) renderer.setVolume(gain);
        if (volumeValue) volumeValue.textContent = gain.toFixed(1) + 'x';
    });
    
    // 面板折叠
    let isPanelCollapsed = false;
    toggleBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        isPanelCollapsed = !isPanelCollapsed;
        controlPanel?.classList.toggle('collapsed', isPanelCollapsed);
    });
    
    // 面板拖拽
    panelHeader?.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);
    
    // 历史记录
    historyToggle?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (historyDropdown.classList.contains('show')) {
            historyDropdown.classList.remove('show');
        } else {
            loadHistory();
            historyDropdown.classList.add('show');
        }
    });
    
    clearHistory?.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm('确定要清空所有历史记录吗?')) {
            await invoke('clear_stream_history');
            showStatus('✅ 历史记录已清空');
            historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,255,255,0.4);">暂无历史记录</div>';
        }
    });
    
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.input-wrapper')) {
            historyDropdown?.classList.remove('show');
        }
    });
    
    // 开始/停止
    startBtn?.addEventListener('click', startStream);
    stopBtn?.addEventListener('click', stopStream);
    
    // 缩放按钮
    document.getElementById('zoom-reset')?.addEventListener('click', () => canvasTransform?.resetView());
    document.getElementById('zoom-fit')?.addEventListener('click', () => canvasTransform?.fitToWindow());
    
    // 流信息面板
    document.getElementById('stream-info-toggle')?.addEventListener('change', (e) => {
        streamInfoManager?.setEnabled(e.target.checked);
    });
}

// ==================== DOMContentLoaded ====================

document.addEventListener('DOMContentLoaded', () => {
    // 初始化流信息管理器
    streamInfoManager = new StreamInfoManager();
    // 传递给渲染器模块
    setStreamInfoManager(streamInfoManager);
    window.streamInfoManager = streamInfoManager;
    console.log('📊 StreamInfoManager 已初始化');
    
    // 初始化画布变换
    const overlayCanvas = document.getElementById('overlay-canvas');
    if (canvas) {
        canvasTransform = new CanvasTransform(canvas, overlayCanvas);
        window.canvasTransform = canvasTransform;
    }
    
    // 初始化调色面板
    colorGrading = new ColorGrading(renderer);
    console.log('🎨 ColorGrading 已初始化');
    
    // 设置事件监听
    setupEventListeners();
    setupNoiseControls();
    setupLlmControls();
});

// ==================== 全局暴露 ====================

window.renderer = renderer;
window.llmManager = llmManager;

console.log('🎬 视频监控系统已加载');
console.log('🔍 缩放功能: 滚轮缩放, 拖拽平移, 双击重置');
