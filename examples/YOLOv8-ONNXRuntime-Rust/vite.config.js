import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: './ui',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    target: 'esnext',
    minify: 'esbuild',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'ui/index.html'),
        regionSelector: resolve(__dirname, 'ui/region-selector.html'),
        recordingIndicator: resolve(__dirname, 'ui/recording-indicator.html'),
      },
      output: {
        manualChunks: undefined,
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  clearScreen: false,
  envPrefix: ['VITE_', 'TAURI_'],
});
