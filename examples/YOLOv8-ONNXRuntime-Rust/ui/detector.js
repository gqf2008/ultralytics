/**
 * 视频帧检测管理器
 * 负责从视频帧提取、缩放并发送给 Rust 后端进行 YOLO 检测
 */

import { invoke, Channel } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export class FrameDetector {
    constructor(videoRenderer, detectionCanvas) {
        this.videoRenderer = videoRenderer;
        this.detectionCanvas = detectionCanvas || document.createElement('canvas');
        this.detectionCtx = this.detectionCanvas.getContext('2d', { 
            willReadFrequently: true // 优化性能
        });
        
        // YOLO 输入尺寸 - 640x640 提高检测精度
        // Raw Request 优化后 1.6MB/帧 IPC 开销可接受
        this.targetWidth = 640;
        this.targetHeight = 640;
        
        // 设置离屏 Canvas 尺寸
        this.detectionCanvas.width = this.targetWidth;
        this.detectionCanvas.height = this.targetHeight;
        
        // 检测状态
        this.isDetecting = false;
        // Raw Request + Channel 优化后可以提高帧率
        this.detectionFps = 25; // 目标检测帧率
        this.lastDetectTime = 0;
        this.detectFrameCount = 0;
        this.detectInterval = null;
        this.rafId = null; // requestAnimationFrame ID
        this.isProcessing = false; // 防止重叠请求
        this.resultUnlisten = null; // 事件监听取消函数
        this.pendingSend = false; // 是否有待发送的帧
        
        // 检测结果缓存
        this.lastDetections = [];
        
        // 模型配置
        this.modelName = 'yolov8n';
        this.trackerName = 'bytetrack';
        
        console.log(`🎯 [Detector] 初始化完成 - 目标尺寸: ${this.targetWidth}x${this.targetHeight}`);
    }
    
    /**
     * 启动检测器
     */
    async startDetector(modelName = 'yolov8n', trackerName = 'bytetrack') {
        if (this.isDetecting) {
            console.warn('⚠️ 检测器已在运行');
            return { message: '检测器已在运行', input_width: this.targetWidth, input_height: this.targetHeight };
        }
        
        this.modelName = modelName;
        this.trackerName = trackerName;
        
        try {
            console.log(`🚀 [Detector] 启动检测器: ${modelName}, tracker: ${trackerName}`);
            
            // 创建结果 Channel (Rust → 前端)
            const resultChannel = new Channel();
            resultChannel.onmessage = (result) => {
                this.lastDetections = result.boxes;
                
                // 触发自定义事件通知 UI 更新
                window.dispatchEvent(new CustomEvent('yolo-detection', { 
                    detail: result 
                }));
                
                this.detectFrameCount++;
                const now = performance.now();
                if (now - this.lastDetectTime >= 2000) {
                    const actualFps = this.detectFrameCount / 2;
                    this.detectFrameCount = 0;
                    this.lastDetectTime = now;
                    console.log(`📊 [Detector] 检测FPS: ${actualFps.toFixed(1)} | 推理: ${result.inference_time_ms.toFixed(1)}ms`);
                }
            };
            
            const result = await invoke('start_detector', { 
                model: modelName,
                tracker: trackerName,
                resultChannel: resultChannel
            });
            console.log('✅ 检测器已启动:', result);
            
            // 使用后端返回的模型输入尺寸
            if (result.input_width && result.input_height) {
                this.setTargetSize(result.input_width, result.input_height);
                console.log(`📐 [Detector] 使用模型输入尺寸: ${result.input_width}x${result.input_height}`);
            }
            
            this.isDetecting = true;
            this.startFrameExtraction();
            
            return result;
        } catch (err) {
            console.error('❌ [Detector] 启动失败:', err);
            throw err;
        }
    }
    
    /**
     * 设置检测结果监听 (已移到 Channel.onmessage)
     */
    async setupResultListener() {
        // 现在使用 Channel 接收结果，不再需要 listen
        console.log('ℹ️ [Detector] 结果监听已改用 Channel');
    }
    
    /**
     * 停止检测器
     */
    async stopDetector() {
        if (!this.isDetecting) {
            console.warn('⚠️ 检测器未运行');
            return;
        }
        
        try {
            console.log('🛑 [Detector] 停止检测器');
            this.isDetecting = false;
            
            // 停止帧提取
            if (this.detectInterval) {
                clearInterval(this.detectInterval);
                this.detectInterval = null;
            }
            if (this.rafId) {
                cancelAnimationFrame(this.rafId);
                this.rafId = null;
            }
            
            // 取消事件监听
            if (this.resultUnlisten) {
                this.resultUnlisten();
                this.resultUnlisten = null;
            }
            
            const result = await invoke('stop_detector');
            console.log('✅', result);
            
            this.lastDetections = [];
            return result;
        } catch (err) {
            console.error('❌ [Detector] 停止失败:', err);
            throw err;
        }
    }
    
    /**
     * 启动帧提取循环 (使用 RAF 跟随视频帧率)
     */
    startFrameExtraction() {
        const minInterval = 1000 / this.detectionFps; // 最小间隔
        
        console.log(`⏱️ [Detector] 启动帧提取 - 目标FPS: ${this.detectionFps}`);
        
        const loop = () => {
            if (!this.isDetecting) return;
            
            const now = performance.now();
            // 控制最大检测频率
            if (now - this.lastDetectTime >= minInterval) {
                this.extractAndDetectFrame();
                this.lastDetectTime = now;
            }
            
            this.rafId = requestAnimationFrame(loop);
        };
        
        this.rafId = requestAnimationFrame(loop);
    }
    
    /**
     * 从渲染器的 Canvas 提取帧并发送检测
     */
    extractAndDetectFrame() {
        if (!this.isDetecting || this.isProcessing) return;
        
        const sourceCanvas = this.videoRenderer.canvas;
        
        // 检查视频源是否有数据
        if (!sourceCanvas || sourceCanvas.width === 0 || sourceCanvas.height === 0) {
            return;
        }
        
        this.isProcessing = true;
        
        try {
            // 1. 将源 Canvas 缩放绘制到检测 Canvas
            this.detectionCtx.drawImage(
                sourceCanvas,
                0, 0, sourceCanvas.width, sourceCanvas.height,
                0, 0, this.targetWidth, this.targetHeight
            );
            
            // 2. 提取 RGBA 像素数据
            const imageData = this.detectionCtx.getImageData(
                0, 0, 
                this.targetWidth, 
                this.targetHeight
            );
            
            // 3. 使用 Uint8Array 直接传输 (Tauri 2.0 Raw Request)
            const rgbaData = new Uint8Array(imageData.data.buffer);
            
            // 4. 使用 Raw Request 传输 - 避免 JSON 序列化 400KB 数据
            // Tauri 2.0 支持直接传 ArrayBuffer/Uint8Array 作为 payload
            invoke('detect_frame', rgbaData, {
                headers: {
                    'X-Width': String(this.targetWidth),
                    'X-Height': String(this.targetHeight)
                }
            }).catch(() => {});
            
        } catch (err) {
            console.error('❌ [Detector] 帧提取失败:', err);
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * 监听检测结果
     */
    listenDetectionResults() {
        listen('detection-result', (event) => {
            const result = event.payload;
            this.lastDetections = result.boxes;
            
            // 触发自定义事件通知 UI 更新
            window.dispatchEvent(new CustomEvent('yolo-detection', { 
                detail: result 
            }));
            
            // 打印检测摘要 (降低频率)
            if (Math.random() < 0.1) { // 10% 采样
                console.log(
                    `🎯 [Detection] 检测到 ${result.boxes.length} 个目标 | ` +
                    `推理耗时: ${result.inference_time_ms.toFixed(1)}ms | ` +
                    `FPS: ${result.detect_fps.toFixed(1)}`
                );
            }
        });
    }
    
    /**
     * 设置检测帧率
     */
    setDetectionFps(fps) {
        this.detectionFps = Math.max(1, Math.min(60, fps)); // 限制 1-60 FPS
        
        if (this.isDetecting) {
            // 重启定时器
            if (this.detectInterval) {
                clearInterval(this.detectInterval);
            }
            this.startFrameExtraction();
        }
        
        console.log(`⚙️ [Detector] 检测FPS 设置为: ${this.detectionFps}`);
    }
    
    /**
     * 设置目标尺寸
     */
    setTargetSize(width, height) {
        this.targetWidth = width;
        this.targetHeight = height;
        this.detectionCanvas.width = width;
        this.detectionCanvas.height = height;
        
        console.log(`⚙️ [Detector] 目标尺寸 设置为: ${width}x${height}`);
    }
    
    /**
     * 获取最新检测结果
     */
    getLastDetections() {
        return this.lastDetections;
    }
}
