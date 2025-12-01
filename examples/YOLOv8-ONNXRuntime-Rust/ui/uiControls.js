/**
 * UI 控件模块
 * 集中管理所有 UI 事件绑定和用户交互
 */

import { invoke, Channel } from '@tauri-apps/api/core';
import { llmManager } from './llmInference.js';
import { NOISE_PRESETS } from './audioProcessor.js';

/**
 * 初始化音频控件
 * @param {Object} renderer - 渲染器实例
 */
export function initAudioControls(renderer) {
    // 音量控制
    const volumeSlider = document.getElementById('volume-slider');
    const volumeValue = document.getElementById('volume-value');
    
    volumeSlider?.addEventListener('input', (e) => {
        const gain = parseFloat(e.target.value);
        if (renderer) {
            renderer.setVolume(gain);
        }
        if (volumeValue) volumeValue.textContent = gain.toFixed(1) + 'x';
        console.log(`🔊 [Volume] Gain set to ${gain.toFixed(1)}x`);
    });
    
    // 降噪预设
    const noisePreset = document.getElementById('noise-preset');
    const highpassSlider = document.getElementById('highpass-slider');
    const highpassValue = document.getElementById('highpass-value');
    const lowpassSlider = document.getElementById('lowpass-slider');
    const lowpassValue = document.getElementById('lowpass-value');
    const noiseGateToggle = document.getElementById('noise-gate-toggle');
    const noiseGateSlider = document.getElementById('noise-gate-slider');
    const noiseGateValue = document.getElementById('noise-gate-value');
    const denoiseStrengthSlider = document.getElementById('denoise-strength-slider');
    const denoiseStrengthValue = document.getElementById('denoise-strength-value');
    
    noisePreset?.addEventListener('change', (e) => {
        const preset = NOISE_PRESETS[e.target.value];
        if (renderer && preset) {
            renderer.applyNoisePreset(preset);
            // 同步更新所有滑块
            if (highpassSlider) {
                highpassSlider.value = preset.highpass;
                if (highpassValue) highpassValue.textContent = preset.highpass + 'Hz';
            }
            if (lowpassSlider) {
                lowpassSlider.value = preset.lowpass;
                if (lowpassValue) lowpassValue.textContent = (preset.lowpass / 1000).toFixed(1) + 'kHz';
            }
            if (noiseGateToggle) {
                noiseGateToggle.checked = preset.gate;
            }
            if (noiseGateSlider) {
                noiseGateSlider.value = preset.gateThreshold;
                if (noiseGateValue) noiseGateValue.textContent = preset.gateThreshold + 'dB';
            }
            if (denoiseStrengthSlider) {
                denoiseStrengthSlider.value = preset.strength;
                if (denoiseStrengthValue) denoiseStrengthValue.textContent = preset.strength + '%';
            }
        }
        console.log(`🔇 [Preset] ${e.target.value}`);
    });
    
    // 降噪强度
    denoiseStrengthSlider?.addEventListener('input', (e) => {
        const strength = parseInt(e.target.value);
        if (renderer) {
            renderer.setDenoiseStrength(strength / 100);
        }
        if (denoiseStrengthValue) denoiseStrengthValue.textContent = strength + '%';
    });
    
    // Wiener 滤波开关
    const wienerToggle = document.getElementById('wiener-toggle');
    wienerToggle?.addEventListener('change', (e) => {
        if (renderer) {
            renderer.setWienerFilterEnabled(e.target.checked);
        }
    });
    
    // 噪声门限开关
    noiseGateToggle?.addEventListener('change', (e) => {
        if (renderer) {
            renderer.setNoiseGateEnabled(e.target.checked);
        }
    });
    
    // 噪声门限阈值
    noiseGateSlider?.addEventListener('input', (e) => {
        const dB = parseInt(e.target.value);
        if (renderer) {
            renderer.setNoiseGateThreshold(dB);
        }
        if (noiseGateValue) noiseGateValue.textContent = dB + 'dB';
    });
    
    // 采样环境噪声
    const sampleNoiseBtn = document.getElementById('sample-noise-btn');
    const noiseStatus = document.getElementById('noise-status');
    
    sampleNoiseBtn?.addEventListener('click', () => {
        if (renderer) {
            renderer.sampleNoiseFloor();
            sampleNoiseBtn.textContent = '🎤 采样中... 请安静';
            sampleNoiseBtn.disabled = true;
            sampleNoiseBtn.style.background = 'rgba(251,146,60,0.5)';
            if (noiseStatus) {
                noiseStatus.textContent = '⏳ 正在采样环境噪声...';
                noiseStatus.style.color = 'rgba(251,191,36,0.9)';
            }
            setTimeout(() => {
                sampleNoiseBtn.textContent = '🎤 采样环境噪声 (必须！)';
                sampleNoiseBtn.disabled = false;
                sampleNoiseBtn.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
                if (noiseStatus) {
                    noiseStatus.textContent = '✅ 噪声采样完成！降噪已激活';
                    noiseStatus.style.color = 'rgba(74,222,128,0.9)';
                }
            }, 3500);
        }
    });
    
    // 高通滤波器滑块
    highpassSlider?.addEventListener('input', (e) => {
        const freq = parseInt(e.target.value);
        if (renderer) {
            renderer.setHighpassFrequency(freq);
        }
        if (highpassValue) highpassValue.textContent = freq + 'Hz';
    });
    
    // 低通滤波器滑块
    lowpassSlider?.addEventListener('input', (e) => {
        const freq = parseInt(e.target.value);
        if (renderer) {
            renderer.setLowpassFrequency(freq);
        }
        if (lowpassValue) lowpassValue.textContent = (freq / 1000).toFixed(1) + 'kHz';
    });
    
    // 陷波滤波器
    const notchSelect = document.getElementById('notch-select');
    notchSelect?.addEventListener('change', (e) => {
        const freq = parseInt(e.target.value);
        if (renderer) {
            renderer.setNotchFrequency(freq);
        }
    });
}

/**
 * 初始化控制面板（折叠/拖拽）
 */
export function initControlPanel() {
    const controlPanel = document.getElementById('control-panel');
    const panelHeader = document.getElementById('panel-header');
    const toggleBtn = document.getElementById('toggle-btn');
    
    let isPanelCollapsed = false;
    let isDragging = false;
    let currentX, currentY, initialX, initialY;
    let xOffset = 0, yOffset = 0;
    
    // 折叠/展开
    toggleBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        isPanelCollapsed = !isPanelCollapsed;
        if (isPanelCollapsed) {
            controlPanel?.classList.add('collapsed');
        } else {
            controlPanel?.classList.remove('collapsed');
        }
    });
    
    // 拖拽开始
    function dragStart(e) {
        if (e.target.closest('#toggle-btn') || e.target.closest('#history-toggle')) {
            return;
        }
        if (e.target === panelHeader || e.target.closest('#panel-header')) {
            initialX = e.clientX - xOffset;
            initialY = e.clientY - yOffset;
            isDragging = true;
            if (controlPanel) controlPanel.style.transition = 'none';
        }
    }
    
    // 拖拽中
    function drag(e) {
        if (isDragging && controlPanel) {
            e.preventDefault();
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;
            xOffset = currentX;
            yOffset = currentY;
            controlPanel.style.transform = `translate(${currentX}px, ${currentY}px)`;
        }
    }
    
    // 拖拽结束
    function dragEnd() {
        isDragging = false;
        if (controlPanel) {
            controlPanel.style.transition = 'background 0.3s ease, border-color 0.3s ease';
        }
    }
    
    panelHeader?.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);
}

/**
 * 初始化历史记录功能
 * @param {Function} showStatus - 状态显示函数
 */
export function initHistoryControls(showStatus) {
    const rtspInput = document.getElementById('rtsp-url');
    const historyToggle = document.getElementById('history-toggle');
    const clearHistory = document.getElementById('clear-history');
    const historyDropdown = document.getElementById('history-dropdown');
    
    // 加载历史记录
    async function loadHistory() {
        try {
            const urls = await invoke('get_stream_history');
            
            if (historyDropdown) {
                historyDropdown.innerHTML = '';
                
                if (urls.length === 0) {
                    historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,255,255,0.4); cursor: default;">暂无历史记录</div>';
                } else {
                    urls.forEach(url => {
                        const item = document.createElement('div');
                        item.className = 'history-item';
                        item.textContent = url;
                        item.addEventListener('click', () => {
                            if (rtspInput) rtspInput.value = url;
                            historyDropdown.classList.remove('show');
                        });
                        historyDropdown.appendChild(item);
                    });
                }
            }
        } catch (err) {
            console.error('加载历史记录失败:', err);
            if (historyDropdown) {
                historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,100,100,0.8); cursor: default;">加载失败</div>';
            }
        }
    }
    
    // 切换下拉列表
    historyToggle?.addEventListener('click', (e) => {
        e.stopPropagation();
        if (historyDropdown?.classList.contains('show')) {
            historyDropdown.classList.remove('show');
        } else {
            loadHistory();
            historyDropdown?.classList.add('show');
        }
    });
    
    // 清空历史记录
    clearHistory?.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm('确定要清空所有历史记录吗?')) {
            try {
                await invoke('clear_stream_history');
                showStatus('✅ 历史记录已清空');
                if (historyDropdown) {
                    historyDropdown.innerHTML = '<div class="history-item" style="color: rgba(255,255,255,0.4); cursor: default;">暂无历史记录</div>';
                }
            } catch (err) {
                console.error('清空历史记录失败:', err);
                showStatus('❌ 清空失败');
            }
        }
    });
    
    // 点击外部关闭下拉列表
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.input-wrapper')) {
            historyDropdown?.classList.remove('show');
        }
    });
}

/**
 * 初始化缩放按钮
 * @param {Object} canvasTransform - 画布变换实例
 */
export function initZoomControls(canvasTransform) {
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
}

/**
 * 初始化 LLM 推理控件
 * @param {Function} showStatus - 状态显示函数
 */
export function initLlmControls(showStatus) {
    const rtspInput = document.getElementById('rtsp-url');
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
    
    // 追加输出文本
    function appendLlmOutput(text) {
        if (llmOutput) {
            llmOutput.textContent += text;
            llmOutput.scrollTop = llmOutput.scrollHeight;
        }
    }
    
    // 配置面板切换
    llmConfigToggle?.addEventListener('click', () => {
        llmConfigPanel?.classList.toggle('hidden');
        if (llmConfigToggle) {
            llmConfigToggle.textContent = llmConfigPanel?.classList.contains('hidden') ? '配置' : '收起';
        }
    });
    
    // 清空输出
    llmClearOutput?.addEventListener('click', () => {
        if (llmOutput) llmOutput.textContent = '';
    });
    
    // 设置 LLM 回调
    llmManager.onFrameInfo = (chunk) => {
        appendLlmOutput(`\n\n━━━ 📷 帧 #${chunk.frame_id} (${new Date().toLocaleTimeString()}) ━━━\n`);
    };
    
    llmManager.onChunk = (chunk, fullText) => {
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
        const url = rtspInput?.value.trim();
        if (!url) {
            showStatus('❌ 请先输入 RTSP 地址');
            return;
        }
        
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
            
            llmOutputContainer?.classList.remove('hidden');
            if (llmOutput) llmOutput.textContent = '🚀 启动 LLM 推理...\n';
            
            await llmManager.start(url);
            
            startLlmBtn.classList.add('hidden');
            stopLlmBtn?.classList.remove('hidden');
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
            startLlmBtn?.classList.remove('hidden');
            if (startLlmBtn) {
                startLlmBtn.disabled = false;
                startLlmBtn.innerHTML = '<span>🤖</span> 开启LLM推理';
            }
            showStatus('⏹ LLM 推理已停止');
            appendLlmOutput('\n\n🛑 LLM 推理已停止\n');
        } catch (e) {
            console.error('停止 LLM 推理失败:', e);
            showStatus('❌ 停止失败: ' + e);
        }
    });
    
    console.log('🤖 LLM 推理模块已加载');
}

/**
 * 创建状态显示函数
 * @returns {Function}
 */
export function createShowStatus() {
    const statusDiv = document.getElementById('status');
    
    return function showStatus(message, isError = false, duration = 5000) {
        if (!statusDiv) return;
        
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
    };
}

export default {
    initAudioControls,
    initControlPanel,
    initHistoryControls,
    initZoomControls,
    initLlmControls,
    createShowStatus
};
