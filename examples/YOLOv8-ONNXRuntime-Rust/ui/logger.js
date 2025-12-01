/**
 * 日志工具 - 将前端日志发送到后端
 */
import { invoke } from '@tauri-apps/api/core';

const originalLog = console.log;
const originalError = console.error;
const originalWarn = console.warn;

async function sendLog(level, args) {
    const message = args.map(arg => 
        typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join(' ');
    try {
        await invoke('frontend_log', { level, message });
    } catch (e) { }
}

export function setupLogging() {
    console.log = (...args) => {
        originalLog(...args);
        sendLog('INFO', args);
    };

    console.error = (...args) => {
        originalError(...args);
        sendLog('ERROR', args);
    };

    console.warn = (...args) => {
        originalWarn(...args);
        sendLog('WARN', args);
    };

    window.addEventListener('error', (event) => {
        console.error('Global Error:', event.message, 'at', event.filename, ':', event.lineno);
    });

    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled Rejection:', event.reason);
    });
}
