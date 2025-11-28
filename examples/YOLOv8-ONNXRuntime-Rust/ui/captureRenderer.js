/**
 * 摄像头/桌面采集渲染器
 * 跨平台支持：Windows (dshow/gdigrab), macOS (avfoundation), Linux (v4l2/x11grab)
 */

const { invoke } = window.__TAURI__.core;

export class CaptureRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.running = false;
        this.animationId = null;
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fps = 0;
        
        // 统计信息
        this.stats = {
            fps: 0,
            frameCount: 0,
            droppedFrames: 0,
            latencyMs: 0
        };
        
        // 事件回调
        this.onFrame = null;
        this.onStats = null;
        this.onError = null;
    }
    
    /**
     * 列出所有可用的采集设备
     * @returns {Promise<Array<{id: string, name: string, device_type: string}>>}
     */
    async listDevices() {
        try {
            return await invoke('list_capture_devices');
        } catch (e) {
            console.error('列出设备失败:', e);
            return [];
        }
    }
    
    /**
     * 启动采集
     * @param {string} deviceId - 设备 ID
     * @param {string} deviceType - 设备类型: 'camera', 'screen', 'window'
     * @param {number} width - 分辨率宽度
     * @param {number} height - 分辨率高度
     * @param {number} fps - 帧率
     */
    async start(deviceId, deviceType, width = 1280, height = 720, fps = 30) {
        if (this.running) {
            await this.stop();
        }
        
        try {
            console.log(`🎥 启动采集: ${deviceId} (${deviceType}) @ ${width}x${height} ${fps}fps`);
            
            await invoke('start_capture', {
                deviceId,
                deviceType,
                width,
                height,
                fps
            });
            
            this.running = true;
            this.canvas.width = width;
            this.canvas.height = height;
            this.lastFrameTime = performance.now();
            this.frameCount = 0;
            
            // 开始渲染循环
            this.renderLoop();
            
        } catch (e) {
            console.error('启动采集失败:', e);
            if (this.onError) this.onError(e);
            throw e;
        }
    }
    
    /**
     * 停止采集
     */
    async stop() {
        this.running = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        try {
            await invoke('stop_capture');
            console.log('🛑 采集已停止');
        } catch (e) {
            console.error('停止采集失败:', e);
        }
    }
    
    /**
     * 渲染循环
     */
    async renderLoop() {
        if (!this.running) return;
        
        try {
            // 获取帧数据 (RGBA 格式)
            const response = await invoke('get_capture_frame');
            
            if (response && response.length > 16) {
                // 解析 header
                const headerView = new DataView(response.buffer);
                const timestamp = Number(headerView.getBigUint64(0, true));
                const width = headerView.getUint32(8, true);
                const height = headerView.getUint32(12, true);
                
                // 提取 RGBA 数据
                const rgbaData = new Uint8ClampedArray(response.buffer, 16);
                
                // 创建 ImageData 并渲染
                if (rgbaData.length === width * height * 4) {
                    const imageData = new ImageData(rgbaData, width, height);
                    
                    // 调整 canvas 尺寸
                    if (this.canvas.width !== width || this.canvas.height !== height) {
                        this.canvas.width = width;
                        this.canvas.height = height;
                    }
                    
                    this.ctx.putImageData(imageData, 0, 0);
                    
                    // 更新统计
                    this.frameCount++;
                    const now = performance.now();
                    if (now - this.lastFrameTime >= 1000) {
                        this.fps = this.frameCount / ((now - this.lastFrameTime) / 1000);
                        this.frameCount = 0;
                        this.lastFrameTime = now;
                        
                        // 获取后端统计
                        this.updateStats();
                    }
                    
                    // 帧回调
                    if (this.onFrame) {
                        this.onFrame({ width, height, timestamp });
                    }
                }
            }
        } catch (e) {
            // 忽略 "没有可用的帧" 错误
            if (!e.includes?.('没有可用的帧')) {
                console.error('获取帧失败:', e);
            }
        }
        
        // 继续渲染循环
        this.animationId = requestAnimationFrame(() => this.renderLoop());
    }
    
    /**
     * 更新统计信息
     */
    async updateStats() {
        try {
            this.stats = await invoke('get_capture_stats');
            this.stats.frontendFps = this.fps;
            
            if (this.onStats) {
                this.onStats(this.stats);
            }
        } catch (e) {
            // 忽略
        }
    }
    
    /**
     * 检查是否正在运行
     * @returns {Promise<boolean>}
     */
    async isRunning() {
        try {
            return await invoke('is_capture_running');
        } catch (e) {
            return false;
        }
    }
    
    /**
     * 获取当前帧用于检测
     * @returns {Uint8ClampedArray|null} RGBA 数据
     */
    getCurrentFrame() {
        if (!this.canvas.width || !this.canvas.height) return null;
        
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        return {
            data: imageData.data,
            width: this.canvas.width,
            height: this.canvas.height
        };
    }
}

/**
 * 创建设备选择 UI
 * @param {string} containerId - 容器元素 ID
 * @param {CaptureRenderer} renderer - 渲染器实例
 */
export function createDeviceSelector(containerId, renderer) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="capture-controls">
            <div class="device-select-group">
                <label>📷 采集设备:</label>
                <select id="device-select">
                    <option value="">-- 选择设备 --</option>
                </select>
                <button id="refresh-devices" title="刷新设备列表">🔄</button>
            </div>
            
            <div class="resolution-group">
                <label>📐 分辨率:</label>
                <select id="resolution-select">
                    <option value="640x480">640x480 (VGA)</option>
                    <option value="1280x720" selected>1280x720 (720p)</option>
                    <option value="1920x1080">1920x1080 (1080p)</option>
                    <option value="2560x1440">2560x1440 (2K)</option>
                </select>
            </div>
            
            <div class="fps-group">
                <label>🎬 帧率:</label>
                <select id="fps-select">
                    <option value="15">15 fps</option>
                    <option value="24">24 fps</option>
                    <option value="30" selected>30 fps</option>
                    <option value="60">60 fps</option>
                </select>
            </div>
            
            <div class="button-group">
                <button id="start-capture" class="primary">▶️ 开始采集</button>
                <button id="stop-capture" class="danger" disabled>⏹️ 停止采集</button>
            </div>
            
            <div class="stats-display" id="capture-stats">
                <span>FPS: --</span>
                <span>帧数: --</span>
            </div>
        </div>
    `;
    
    // 获取元素
    const deviceSelect = document.getElementById('device-select');
    const refreshBtn = document.getElementById('refresh-devices');
    const resolutionSelect = document.getElementById('resolution-select');
    const fpsSelect = document.getElementById('fps-select');
    const startBtn = document.getElementById('start-capture');
    const stopBtn = document.getElementById('stop-capture');
    const statsDisplay = document.getElementById('capture-stats');
    
    // 刷新设备列表
    async function refreshDevices() {
        deviceSelect.innerHTML = '<option value="">-- 加载中... --</option>';
        
        const devices = await renderer.listDevices();
        
        deviceSelect.innerHTML = '<option value="">-- 选择设备 --</option>';
        
        // 分组显示
        const cameras = devices.filter(d => d.device_type === 'camera');
        const screens = devices.filter(d => d.device_type === 'screen');
        const windows = devices.filter(d => d.device_type === 'window');
        
        if (cameras.length > 0) {
            const group = document.createElement('optgroup');
            group.label = '📷 摄像头';
            cameras.forEach(d => {
                const option = document.createElement('option');
                option.value = JSON.stringify({ id: d.id, type: d.device_type });
                option.textContent = d.name;
                group.appendChild(option);
            });
            deviceSelect.appendChild(group);
        }
        
        if (screens.length > 0) {
            const group = document.createElement('optgroup');
            group.label = '🖥️ 屏幕';
            screens.forEach(d => {
                const option = document.createElement('option');
                option.value = JSON.stringify({ id: d.id, type: d.device_type });
                option.textContent = d.name;
                group.appendChild(option);
            });
            deviceSelect.appendChild(group);
        }
        
        if (windows.length > 0) {
            const group = document.createElement('optgroup');
            group.label = '🪟 窗口';
            windows.forEach(d => {
                const option = document.createElement('option');
                option.value = JSON.stringify({ id: d.id, type: d.device_type });
                option.textContent = d.name;
                group.appendChild(option);
            });
            deviceSelect.appendChild(group);
        }
    }
    
    // 开始采集
    async function startCapture() {
        const deviceValue = deviceSelect.value;
        if (!deviceValue) {
            alert('请选择一个采集设备');
            return;
        }
        
        const device = JSON.parse(deviceValue);
        const [width, height] = resolutionSelect.value.split('x').map(Number);
        const fps = parseInt(fpsSelect.value);
        
        try {
            startBtn.disabled = true;
            startBtn.textContent = '⏳ 启动中...';
            
            await renderer.start(device.id, device.type, width, height, fps);
            
            startBtn.disabled = true;
            startBtn.textContent = '▶️ 开始采集';
            stopBtn.disabled = false;
            deviceSelect.disabled = true;
            resolutionSelect.disabled = true;
            fpsSelect.disabled = true;
            
        } catch (e) {
            startBtn.disabled = false;
            startBtn.textContent = '▶️ 开始采集';
            alert('启动采集失败: ' + e);
        }
    }
    
    // 停止采集
    async function stopCapture() {
        await renderer.stop();
        
        startBtn.disabled = false;
        stopBtn.disabled = true;
        deviceSelect.disabled = false;
        resolutionSelect.disabled = false;
        fpsSelect.disabled = false;
    }
    
    // 更新统计显示
    renderer.onStats = (stats) => {
        statsDisplay.innerHTML = `
            <span>FPS: ${stats.frontendFps?.toFixed(1) || '--'}</span>
            <span>帧数: ${stats.frame_count || 0}</span>
        `;
    };
    
    // 绑定事件
    refreshBtn.addEventListener('click', refreshDevices);
    startBtn.addEventListener('click', startCapture);
    stopBtn.addEventListener('click', stopCapture);
    
    // 初始加载设备列表
    refreshDevices();
}

// 添加样式
const style = document.createElement('style');
style.textContent = `
.capture-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 8px;
    align-items: center;
}

.capture-controls label {
    font-size: 12px;
    color: #aaa;
    margin-right: 4px;
}

.capture-controls select {
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid #444;
    background: #222;
    color: #fff;
    font-size: 13px;
    min-width: 150px;
}

.capture-controls button {
    padding: 8px 16px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
}

.capture-controls button.primary {
    background: #4CAF50;
    color: white;
}

.capture-controls button.primary:hover:not(:disabled) {
    background: #45a049;
}

.capture-controls button.danger {
    background: #f44336;
    color: white;
}

.capture-controls button.danger:hover:not(:disabled) {
    background: #da190b;
}

.capture-controls button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.capture-controls button#refresh-devices {
    padding: 6px 10px;
    background: #333;
    color: #fff;
}

.device-select-group,
.resolution-group,
.fps-group,
.button-group {
    display: flex;
    align-items: center;
    gap: 6px;
}

.stats-display {
    display: flex;
    gap: 16px;
    padding: 6px 12px;
    background: rgba(0, 0, 0, 0.4);
    border-radius: 4px;
    font-size: 12px;
    color: #0f0;
    font-family: monospace;
}
`;
document.head.appendChild(style);

export default CaptureRenderer;
