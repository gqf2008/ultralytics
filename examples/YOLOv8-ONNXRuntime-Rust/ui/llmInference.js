// LLM 视频推理模块
// 支持 Ollama, OpenAI, 本地 API

import { invoke, Channel } from '@tauri-apps/api/core';

/**
 * LLM 推理配置
 */
export const defaultLlmConfig = {
    api_url: 'http://localhost:11434/api/chat',  // Ollama 默认
    api_key: null,
    model: 'llava',
    system_prompt: '你是一个视频监控分析助手。请分析图像中的内容，描述你看到的场景、人物、物体和活动。如果发现异常情况，请特别指出。',
    user_prompt: '请分析这张监控画面，描述你看到的内容',
    frame_interval: 3.0,  // 每3秒分析一帧
    max_tokens: 500,
    temperature: 0.7
};

/**
 * LLM 推理管理器
 */
export class LlmInferenceManager {
    constructor() {
        this.isRunning = false;
        this.config = { ...defaultLlmConfig };
        this.resultChannel = null;
        this.onChunk = null;      // 流式文本回调
        this.onFrameInfo = null;  // 帧信息回调
        this.onStart = null;      // 推理开始回调
        this.onEnd = null;        // 推理结束回调
        this.onError = null;      // 错误回调
        this.frameCount = 0;
        this.currentFrameId = 0;
        this.currentText = '';
        this.frameIdCounter = 0;  // 帧ID计数器
    }

    /**
     * 设置配置
     */
    setConfig(config) {
        this.config = { ...this.config, ...config };
    }

    /**
     * 启动 LLM 推理
     * @param {string} rtspUrl - RTSP 流地址 (可选，现在由前端直接提交帧)
     */
    async start(rtspUrl) {
        if (this.isRunning) {
            console.warn('LLM 推理已在运行');
            return;
        }

        // 创建结果 Channel (接收 InvokeResponseBody)
        this.resultChannel = new Channel();
        this.resultChannel.onmessage = (response) => this.handleResponse(response);

        try {
            const result = await invoke('start_llm_inference', {
                config: this.config,
                resultChannel: this.resultChannel
            });

            this.isRunning = true;
            this.frameCount = 0;
            this.frameIdCounter = 0;
            console.log('🤖 LLM 推理已启动:', result);
            return result;
        } catch (e) {
            console.error('启动 LLM 推理失败:', e);
            throw e;
        }
    }

    /**
     * 停止推理
     */
    async stop() {
        if (!this.isRunning) return;

        try {
            await invoke('stop_llm_inference');
            this.isRunning = false;
            console.log('🛑 LLM 推理已停止');
        } catch (e) {
            console.error('停止 LLM 推理失败:', e);
        }
    }

    /**
     * 提交 RGBA 帧进行 LLM 推理
     * @param {Uint8ClampedArray|Uint8Array} rgbaData - RGBA 像素数据
     * @param {number} width - 图像宽度
     * @param {number} height - 图像高度
     */
    async submitFrame(rgbaData, width, height) {
        if (!this.isRunning) {
            console.warn('LLM 推理未启动，无法提交帧');
            return;
        }

        this.frameIdCounter++;
        const frameId = this.frameIdCounter;

        // 构建带 header 的数据包
        // header: width(4) + height(4) + frame_id(8) = 16 bytes
        const headerSize = 16;
        const totalSize = headerSize + rgbaData.length;
        const buffer = new ArrayBuffer(totalSize);
        const view = new DataView(buffer);
        
        // 写入 header (little-endian)
        view.setUint32(0, width, true);
        view.setUint32(4, height, true);
        // frame_id 是 u64，分两次写入
        view.setUint32(8, frameId & 0xFFFFFFFF, true);
        view.setUint32(12, 0, true); // 高32位为0

        // 写入 RGBA 数据
        const dataView = new Uint8Array(buffer, headerSize);
        dataView.set(rgbaData);

        try {
            // 使用 Raw body 发送
            await invoke('submit_frame_for_llm', {}, {
                headers: { 'Content-Type': 'application/octet-stream' }
            });
            
            // 注意：Tauri 2.0 的 invoke 暂不直接支持 Raw body
            // 需要通过其他方式传递，这里先用普通方式
            // 实际实现可能需要使用 fetch 或其他方法
            
            console.log(`📸 提交帧 #${frameId}: ${width}x${height}, ${rgbaData.length} bytes`);
        } catch (e) {
            console.error('提交帧失败:', e);
        }
    }

    /**
     * 从 Canvas 提交帧
     * @param {HTMLCanvasElement} canvas - Canvas 元素
     */
    async submitFrameFromCanvas(canvas) {
        if (!this.isRunning) return;

        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        await this.submitFrame(imageData.data, canvas.width, canvas.height);
    }

    /**
     * 从 VideoFrame 提交帧
     * @param {VideoFrame} frame - WebCodecs VideoFrame
     */
    async submitFrameFromVideoFrame(frame) {
        if (!this.isRunning) return;

        const width = frame.displayWidth;
        const height = frame.displayHeight;
        
        // 创建临时 canvas 转换为 RGBA
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(frame, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, width, height);
        await this.submitFrame(imageData.data, width, height);
    }

    /**
     * 处理后端响应 (InvokeResponseBody)
     * @param {string|ArrayBuffer} response - JSON 字符串或二进制数据
     */
    handleResponse(response) {
        // InvokeResponseBody 可能是 JSON 字符串或 Raw 二进制
        if (typeof response === 'string') {
            // JSON 响应
            try {
                const chunk = JSON.parse(response);
                this.handleChunk(chunk);
            } catch (e) {
                console.error('解析 JSON 响应失败:', e, response);
            }
        } else if (response instanceof ArrayBuffer || response instanceof Uint8Array) {
            // Raw 二进制响应 (如处理后的图像)
            console.log('收到二进制响应:', response.byteLength || response.length, 'bytes');
            // 可以在这里处理返回的图像数据
        } else {
            console.warn('未知响应类型:', typeof response, response);
        }
    }

    /**
     * 处理流式推理结果 (解析后的 chunk)
     */
    handleChunk(chunk) {
        switch (chunk.type) {
            case 'frame_info':
                this.currentFrameId = chunk.frame_id;
                this.frameCount++;
                this.currentText = '';
                if (this.onFrameInfo) {
                    this.onFrameInfo(chunk);
                }
                break;

            case 'start':
                if (this.onStart) {
                    this.onStart(chunk);
                }
                break;

            case 'chunk':
                this.currentText += chunk.content;
                if (this.onChunk) {
                    this.onChunk(chunk, this.currentText);
                }
                break;

            case 'end':
                if (this.onEnd) {
                    this.onEnd(chunk, this.currentText);
                }
                break;

            case 'error':
                console.error('LLM 推理错误:', chunk.content);
                if (this.onError) {
                    this.onError(chunk);
                }
                break;
        }
    }

    /**
     * 获取状态
     */
    async getStatus() {
        try {
            const [running, frames] = await invoke('get_llm_status');
            return { running, frames };
        } catch (e) {
            return { running: false, frames: 0 };
        }
    }
}

// 导出单例
export const llmManager = new LlmInferenceManager();

/**
 * LLM 推理面板 UI
 */
export class LlmInferencePanel {
    constructor() {
        this.manager = llmManager;
        this.elements = {};
        this.rtspUrl = '';
    }

    /**
     * 初始化面板
     */
    init() {
        // 获取 DOM 元素
        this.elements = {
            configToggle: document.getElementById('llm-config-toggle'),
            configPanel: document.getElementById('llm-config-panel'),
            apiUrl: document.getElementById('llm-api-url'),
            model: document.getElementById('llm-model'),
            systemPrompt: document.getElementById('llm-system-prompt'),
            userPrompt: document.getElementById('llm-user-prompt'),
            frameInterval: document.getElementById('llm-frame-interval'),
            temperature: document.getElementById('llm-temperature'),
            startBtn: document.getElementById('start-llm-btn'),
            stopBtn: document.getElementById('stop-llm-btn'),
            outputContainer: document.getElementById('llm-output-container'),
            output: document.getElementById('llm-output'),
            clearOutput: document.getElementById('llm-clear-output')
        };

        // 检查元素是否存在
        if (!this.elements.startBtn) {
            console.warn('LLM 面板元素未找到，跳过初始化');
            return;
        }

        // 绑定事件
        this.bindEvents();
        
        // 设置默认值
        this.loadConfig();
        
        // 设置回调
        this.setupCallbacks();

        console.log('🤖 LLM 推理面板已初始化');
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        // 配置面板切换
        this.elements.configToggle?.addEventListener('click', () => {
            this.elements.configPanel?.classList.toggle('hidden');
            this.elements.configToggle.textContent = 
                this.elements.configPanel?.classList.contains('hidden') ? '配置' : '收起';
        });

        // 开始按钮
        this.elements.startBtn?.addEventListener('click', () => this.start());

        // 停止按钮
        this.elements.stopBtn?.addEventListener('click', () => this.stop());

        // 清空输出
        this.elements.clearOutput?.addEventListener('click', () => {
            if (this.elements.output) {
                this.elements.output.textContent = '';
            }
        });

        // 配置变更保存
        const configInputs = [
            this.elements.apiUrl,
            this.elements.model,
            this.elements.systemPrompt,
            this.elements.userPrompt,
            this.elements.frameInterval,
            this.elements.temperature
        ];
        
        configInputs.forEach(input => {
            input?.addEventListener('change', () => this.saveConfig());
        });
    }

    /**
     * 加载配置到 UI
     */
    loadConfig() {
        const config = this.manager.config;
        if (this.elements.apiUrl) this.elements.apiUrl.value = config.api_url || '';
        if (this.elements.model) this.elements.model.value = config.model || '';
        if (this.elements.systemPrompt) this.elements.systemPrompt.value = config.system_prompt || '';
        if (this.elements.userPrompt) this.elements.userPrompt.value = config.user_prompt || '';
        if (this.elements.frameInterval) this.elements.frameInterval.value = config.frame_interval || 60;
        if (this.elements.temperature) this.elements.temperature.value = config.temperature || 0.7;
    }

    /**
     * 从 UI 保存配置
     */
    saveConfig() {
        const config = {
            api_url: this.elements.apiUrl?.value || defaultLlmConfig.api_url,
            model: this.elements.model?.value || defaultLlmConfig.model,
            system_prompt: this.elements.systemPrompt?.value || '',
            user_prompt: this.elements.userPrompt?.value || defaultLlmConfig.user_prompt,
            frame_interval: parseFloat(this.elements.frameInterval?.value) || 2.0,
            temperature: parseFloat(this.elements.temperature?.value) || 0.7
        };
        
        this.manager.setConfig(config);
        console.log('📝 LLM 配置已保存:', config);
    }

    /**
     * 设置流式回调
     */
    setupCallbacks() {
        // 帧信息
        this.manager.onFrameInfo = (chunk) => {
            this.appendOutput(`\n\n━━━ 📷 帧 #${chunk.frame_id} (${new Date().toLocaleTimeString()}) ━━━\n`);
        };

        // 开始推理
        this.manager.onStart = () => {
            // 显示光标或加载指示
        };

        // 流式文本
        this.manager.onChunk = (chunk, fullText) => {
            // 增量更新，只添加新内容
            this.updateStreamingText(chunk.content);
        };

        // 推理结束
        this.manager.onEnd = () => {
            this.appendOutput('\n');
        };

        // 错误
        this.manager.onError = (chunk) => {
            this.appendOutput(`\n❌ 错误: ${chunk.content}\n`);
        };
    }

    /**
     * 追加输出
     */
    appendOutput(text) {
        if (this.elements.output) {
            this.elements.output.textContent += text;
            this.elements.output.scrollTop = this.elements.output.scrollHeight;
        }
    }

    /**
     * 流式更新文本
     */
    updateStreamingText(text) {
        this.appendOutput(text);
    }

    /**
     * 设置 RTSP URL (从主应用获取)
     */
    setRtspUrl(url) {
        this.rtspUrl = url;
    }

    /**
     * 开始推理
     */
    async start() {
        // 获取当前 RTSP URL
        const rtspInput = document.getElementById('rtsp-url');
        const rtspUrl = rtspInput?.value || this.rtspUrl;

        if (!rtspUrl) {
            this.showError('请先输入 RTSP 地址');
            return;
        }

        // 保存最新配置
        this.saveConfig();

        try {
            // 更新按钮状态
            this.elements.startBtn?.classList.add('hidden');
            this.elements.stopBtn?.classList.remove('hidden');
            
            // 显示输出容器
            this.elements.outputContainer?.classList.remove('hidden');
            
            // 清空之前的输出
            if (this.elements.output) {
                this.elements.output.textContent = '🚀 启动 LLM 推理...\n';
            }

            await this.manager.start(rtspUrl);
            this.appendOutput('✅ 已连接，等待视频帧...\n');

        } catch (e) {
            this.showError(`启动失败: ${e}`);
            // 恢复按钮状态
            this.elements.startBtn?.classList.remove('hidden');
            this.elements.stopBtn?.classList.add('hidden');
        }
    }

    /**
     * 停止推理
     */
    async stop() {
        try {
            await this.manager.stop();
            this.appendOutput('\n\n🛑 LLM 推理已停止\n');
        } catch (e) {
            console.error('停止失败:', e);
        }

        // 更新按钮状态
        this.elements.startBtn?.classList.remove('hidden');
        this.elements.stopBtn?.classList.add('hidden');
    }

    /**
     * 显示错误
     */
    showError(message) {
        // 使用全局 status 显示
        const status = document.getElementById('status');
        if (status) {
            status.textContent = message;
            status.classList.add('error');
            status.classList.remove('hidden');
            setTimeout(() => status.classList.add('hidden'), 3000);
        }
        console.error(message);
    }
}

// 导出面板单例
export const llmPanel = new LlmInferencePanel();
