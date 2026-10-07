// vite.config.js
// Purpose: let Vite bundle CesiumJS - vite-plugin-cesium copies Cesium's workers, assets
//          and widgets and tells Cesium where to find them (CESIUM_BASE_URL)
import { defineConfig } from 'vite';
import cesium from 'vite-plugin-cesium';

export default defineConfig({
  plugins: [cesium()],
});
