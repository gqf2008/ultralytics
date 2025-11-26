/**
 * 视频帧检测管理器
 * 负责从视频帧提取、缩放并发送给 Rust 后端进行 YOLO 检测
 */

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

export class FrameDetector {
    constructor(videoRenderer, detectionCanvas) {
        this.videoRenderer = videoRenderer;
        this.detectionCanvas = detectionCanvas || document.createElement('canvas');
        this.detectionCtx = this.detectionCanvas.getContext('2d', { 
            willReadFrequently: true // 优化性能
        });
        
        // YOLO 输入尺寸 (可配置)
        this.targetWidth = 640;
        this.targetHeight = 640;
        
        // 设置离屏 Canvas 尺寸
        this.detectionCanvas.width = this.targetWidth;
        this.detectionCanvas.height = this.targetHeight;
        
        // 检测状态
        this.isDetecting = false;
        this.detectionFps = 30; // 检测帧率 (可调低降低 CPU 占用)
        this.lastDetectTime = 0;
        this.detectFrameCount = 0;
        this.detectInterval = null;
        
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
            return;
        }
        
        this.modelName = modelName;
        this.trackerName = trackerName;
        
        try {
            console.log(`🚀 [Detector] 启动检测器: ${modelName}, tracker: ${trackerName}`);
            const result = await invoke('start_detector', { 
                model: modelName,
                tracker: trackerName 
            });
            console.log('✅', result.message);
            
            // 使用后端返回的输入尺寸
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
     * 启动帧提取定时器
     */
    startFrameExtraction() {
        const intervalMs = 1000 / this.detectionFps;
        
        console.log(`⏱️ [Detector] 启动帧提取 - 目标FPS: ${this.detectionFps} (间隔: ${intervalMs.toFixed(1)}ms)`);
        
        this.detectInterval = setInterval(() => {
            this.extractAndDetectFrame();
        }, intervalMs);
    }
    
    /**
     * 从渲染器的 Canvas 提取帧并发送检测
     */
    async extractAndDetectFrame() {
        if (!this.isDetecting) return;
        
        const sourceCanvas = this.videoRenderer.canvas;
        
        // 检查视频源是否有数据
        if (!sourceCanvas || sourceCanvas.width === 0 || sourceCanvas.height === 0) {
            return;
        }
        
        try {
            const now = performance.now();
            
            // 1. 将源 Canvas 缩放绘制到检测 Canvas (640x640)
            this.detectionCtx.drawImage(
                sourceCanvas,
                0, 0, sourceCanvas.width, sourceCanvas.height,  // 源区域
                0, 0, this.targetWidth, this.targetHeight        // 目标区域(缩放)
            );
            
            // 2. 提取 RGBA 像素数据
            const imageData = this.detectionCtx.getImageData(
                0, 0, 
                this.targetWidth, 
                this.targetHeight
            );
            
            const rgbaData = imageData.data; // Uint8ClampedArray
            
            // 3. 转换为普通 Uint8Array (Tauri 需要)
            const uint8Array = new Uint8Array(rgbaData);
            
            // 4. 发送到 Rust 后端检测并获取结果
            const result = await invoke('detect_frame', {
                rgbaData: Array.from(uint8Array), // Tauri 需要 Array
                width: this.targetWidth,
                height: this.targetHeight
            });
            
            // 5. 处理检测结果
            if (result && result.boxes) {
                this.lastDetections = result.boxes;
                
                // 触发自定义事件通知 UI 更新
                window.dispatchEvent(new CustomEvent('yolo-detection', { 
                    detail: result 
                }));
                
                // 打印检测摘要 (有检测结果时打印)
                if (result.boxes.length > 0) {
                    console.log(
                        `🎯 [Detection] 检测到 ${result.boxes.length} 个目标 | ` +
                        `推理耗时: ${result.inference_time_ms.toFixed(1)}ms | ` +
                        `FPS: ${result.detect_fps.toFixed(1)}`
                    );
                }
            }
            
            this.detectFrameCount++;
            
            // 统计检测FPS
            if (now - this.lastDetectTime >= 1000) {
                const actualFps = this.detectFrameCount;
                this.detectFrameCount = 0;
                this.lastDetectTime = now;
                
                // 每秒打印一次统计
                console.log(`📊 [Detector] 检测FPS: ${actualFps} | 数据量: ${(uint8Array.length / 1024).toFixed(1)}KB`);
            }
            
        } catch (err) {
            console.error('❌ [Detector] 帧检测失败:', err);
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
