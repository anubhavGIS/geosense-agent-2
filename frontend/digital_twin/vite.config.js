// vite.config.js
// Purpose: let Vite bundle CesiumJS - vite-plugin-cesium copies Cesium's workers, assets
//          and widgets and tells Cesium where to find them (CESIUM_BASE_URL)
// Phase 5, Exercise 3: a build whose env file sets VITE_ASSET_BASE_URL (.env.production -
//          the GitHub Pages build) reads the imagery tiles and the GeoJSON layers from the
//          Cloud Storage bucket, so public/ (223 MB) is not copied into dist/; only the
//          favicon, which index.html links, is written there
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig, loadEnv } from 'vite';
import cesium from 'vite-plugin-cesium';

const here = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig(({ mode }) => {
  const fromBucket = Boolean(loadEnv(mode, here, 'VITE_').VITE_ASSET_BASE_URL);
  const faviconOnly = {
    name: 'geosense-favicon-only',
    apply: 'build',
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'favicon.svg',
                      source: readFileSync(new URL('./public/favicon.svg', import.meta.url)) });
    },
  };
  return {
    plugins: [cesium(), fromBucket && faviconOnly],
    build: { copyPublicDir: !fromBucket },
  };
});
