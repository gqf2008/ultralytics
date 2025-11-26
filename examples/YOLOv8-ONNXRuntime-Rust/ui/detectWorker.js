/**
 * 检测 Worker - 在后台线程处理帧数据，避免阻塞主线程
 */

self.onmessage = async (e) => {
    const { type, data } = e.data;
    
    if (type === 'detect') {
        // 直接转发数据，不做任何处理
        // Worker 主要用于隔离 IPC 调用
        self.postMessage({ type: 'frame', data: data });
    }
};
