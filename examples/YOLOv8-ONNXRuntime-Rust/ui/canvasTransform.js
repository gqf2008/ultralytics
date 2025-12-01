/**
 * 画布变换模块
 * 处理视频画布的缩放、平移和拖拽功能
 */

export class CanvasTransform {
    constructor(canvas, overlayCanvas) {
        this.canvas = canvas;
        this.overlayCanvas = overlayCanvas;
        this.scale = 1;
        this.minScale = 0.1;
        this.maxScale = 10;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        this.container = document.getElementById('canvas-container');
        
        this.initEvents();
        this.updateTransform();
    }
    
    initEvents() {
        // 滚轮缩放
        this.container.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const rect = this.container.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // 以鼠标位置为中心缩放
            const imgX = (mouseX - this.offsetX) / this.scale;
            const imgY = (mouseY - this.offsetY) / this.scale;
            
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * delta));
            
            this.offsetX = mouseX - imgX * newScale;
            this.offsetY = mouseY - imgY * newScale;
            this.scale = newScale;
            
            this.updateTransform();
            this.updateZoomDisplay();
        }, { passive: false });
        
        // 鼠标按下开始拖拽
        this.container.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;  // 只响应左键
            if (e.target.closest('#control-panel')) return;  // 忽略控制面板区域
            
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.container.style.cursor = 'grabbing';
        });
        
        // 鼠标移动拖拽
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            this.offsetX += deltaX;
            this.offsetY += deltaY;
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.updateTransform();
        });
        
        // 鼠标松开结束拖拽
        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.container.style.cursor = 'grab';
            }
        });
        
        // 双击重置视图
        this.container.addEventListener('dblclick', (e) => {
            if (e.target.closest('#control-panel')) return;
            this.resetView();
        });
        
        this.container.style.cursor = 'grab';
    }
    
    /**
     * 应用变换到画布
     */
    updateTransform() {
        const transform = `translate(${this.offsetX}px, ${this.offsetY}px) scale(${this.scale})`;
        this.canvas.style.transform = transform;
        this.canvas.style.transformOrigin = '0 0';
        if (this.overlayCanvas) {
            this.overlayCanvas.style.transform = transform;
            this.overlayCanvas.style.transformOrigin = '0 0';
        }
    }
    
    /**
     * 更新缩放百分比显示
     */
    updateZoomDisplay() {
        const zoomEl = document.getElementById('zoom-level');
        if (zoomEl) {
            zoomEl.textContent = `${Math.round(this.scale * 100)}%`;
        }
    }
    
    /**
     * 重置视图到原始状态
     */
    resetView() {
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.updateTransform();
        this.updateZoomDisplay();
        console.log('🔄 视图已重置');
    }
    
    /**
     * 适应窗口大小
     */
    fitToWindow() {
        const containerRect = this.container.getBoundingClientRect();
        const canvasWidth = this.canvas.width || containerRect.width;
        const canvasHeight = this.canvas.height || containerRect.height;
        
        const scaleX = containerRect.width / canvasWidth;
        const scaleY = containerRect.height / canvasHeight;
        this.scale = Math.min(scaleX, scaleY, 1);
        
        this.offsetX = (containerRect.width - canvasWidth * this.scale) / 2;
        this.offsetY = (containerRect.height - canvasHeight * this.scale) / 2;
        
        this.updateTransform();
        this.updateZoomDisplay();
    }
    
    /**
     * 设置缩放级别
     * @param {number} scale - 缩放比例
     */
    setScale(scale) {
        this.scale = Math.max(this.minScale, Math.min(this.maxScale, scale));
        this.updateTransform();
        this.updateZoomDisplay();
    }
    
    /**
     * 获取当前缩放级别
     * @returns {number}
     */
    getScale() {
        return this.scale;
    }
}

export default CanvasTransform;
