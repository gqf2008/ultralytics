/**
 * 调色面板 - 亮度/对比度/饱和度控制
 */

class ColorGrading {
    constructor(renderer) {
        this.renderer = renderer;
        
        // 默认值
        this.defaults = {
            brightness: 1,
            contrast: 1.2,
            saturate: 1.3,
            hue: 0,
            blur: 0
        };
        
        // 当前值
        this.values = { ...this.defaults };
        
        // 预设
        this.presets = {
            vivid: { brightness: 1.05, contrast: 1.3, saturate: 1.6, hue: 0, blur: 0 },
            soft: { brightness: 1.1, contrast: 0.95, saturate: 0.9, hue: 0, blur: 0.5 }
        };
        
        this.setupEventListeners();
        this.updateFilter();
    }
    
    setupEventListeners() {
        // 展开/收起按钮
        const toggleBtn = document.getElementById('color-config-toggle');
        const panel = document.getElementById('color-config-panel');
        toggleBtn?.addEventListener('click', () => {
            panel?.classList.toggle('hidden');
            toggleBtn.textContent = panel?.classList.contains('hidden') ? '展开' : '收起';
        });
        
        // 滑块事件
        this.bindSlider('brightness', '%', 100);
        this.bindSlider('contrast', '%', 100);
        this.bindSlider('saturate', '%', 100);
        this.bindSlider('hue', '°', 1);
        this.bindSlider('blur', 'px', 1);
        
        // 重置按钮
        document.getElementById('color-reset-btn')?.addEventListener('click', () => {
            this.applyPreset(this.defaults);
        });
        
        // 预设按钮
        document.getElementById('color-preset-vivid')?.addEventListener('click', () => {
            this.applyPreset(this.presets.vivid);
        });
        document.getElementById('color-preset-soft')?.addEventListener('click', () => {
            this.applyPreset(this.presets.soft);
        });
    }
    
    bindSlider(name, suffix, multiplier) {
        const slider = document.getElementById(`${name}-slider`);
        const valueEl = document.getElementById(`${name}-value`);
        
        slider?.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            this.values[name] = val;
            
            if (valueEl) {
                if (suffix === '%') {
                    valueEl.textContent = Math.round(val * multiplier) + suffix;
                } else {
                    valueEl.textContent = val + suffix;
                }
            }
            
            this.updateFilter();
        });
    }
    
    applyPreset(preset) {
        this.values = { ...preset };
        
        // 更新滑块
        const sliders = ['brightness', 'contrast', 'saturate', 'hue', 'blur'];
        sliders.forEach(name => {
            const slider = document.getElementById(`${name}-slider`);
            const valueEl = document.getElementById(`${name}-value`);
            if (slider) slider.value = this.values[name];
            if (valueEl) {
                if (name === 'hue') {
                    valueEl.textContent = this.values[name] + '°';
                } else if (name === 'blur') {
                    valueEl.textContent = this.values[name] + 'px';
                } else {
                    valueEl.textContent = Math.round(this.values[name] * 100) + '%';
                }
            }
        });
        
        this.updateFilter();
        console.log('🎨 应用预设:', preset);
    }
    
    updateFilter() {
        const { brightness, contrast, saturate, hue, blur } = this.values;
        
        // 构建 CSS filter 字符串
        let filter = `brightness(${brightness}) contrast(${contrast}) saturate(${saturate})`;
        
        if (hue !== 0) {
            filter += ` hue-rotate(${hue}deg)`;
        }
        if (blur > 0) {
            filter += ` blur(${blur}px)`;
        }
        
        // 保存到渲染器
        if (this.renderer) {
            this.renderer.colorFilter = filter;
        }
        
        console.log('🎨 滤镜:', filter);
    }
    
    getFilter() {
        return this.renderer?.colorFilter || 'none';
    }
}

export { ColorGrading };
