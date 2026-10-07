// main.js
// Purpose: GeoSense 3D digital twin - a CesiumJS globe with world terrain and satellite
//          imagery, the flood-risk zones of the Kolkata study area (PostGIS flood_zones),
//          the LiDAR buildings of the Dupont Circle tile extruded by their height, an NDVI
//          overlay of 2023 and Sentinel-2 imagery of four years behind a year selector,
//          with buttons to the NDVI-change hotspots of Phase II for comparing years
// Exercise 5: polls the MCP server for the agent's latest result and flies the camera to
//          it (800 m) whenever a new result arrives, with a marker and the similar sites
// Open http://localhost:5173 after `npm run dev`; ?buildings=cylinders shows the
// blueprint's cylinder stand-in instead of the extruded footprints.
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import './style.css';

Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_TOKEN;

const PLACES = {
  kolkata: { lon: 88.36, lat: 22.56, height: 15000 },          // the study area
  dupont: { lon: -77.0434, lat: 38.9097, height: 1500 },       // the LiDAR tile
  loss: { lon: 88.3022, lat: 22.6462, height: 4000 },     // Phase II NDVI loss, 2018-23
  gain: { lon: 88.4079, lat: 22.5189, height: 4000 },     // Phase II NDVI gain, 2018-23
};
const STUDY_AREA = Cesium.Rectangle.fromDegrees(88.30, 22.45, 88.42, 22.67);
const YEARS = [2018, 2021, 2023, 2025];
const FLOOD_COLOURS = {
  high: Cesium.Color.RED.withAlpha(0.45),
  medium: Cesium.Color.ORANGE.withAlpha(0.35),
  low: Cesium.Color.YELLOW.withAlpha(0.25),
};
const HEIGHT_CLASSES = [                                      // upper limit (m), colour
  [10, '#c6dbef', '< 10 m'],
  [20, '#6baed6', '10-20 m'],
  [30, '#2171b5', '20-30 m'],
  [Infinity, '#08306b', '>= 30 m'],
];
const BUILDING_MODE = new URLSearchParams(window.location.search).get('buildings')
  || 'extruded';

const viewer = new Cesium.Viewer('cesiumContainer', {
  terrain: Cesium.Terrain.fromWorldTerrain(),
  baseLayerPicker: true,
  timeline: true,
  animation: true,
});

function flyTo(place, duration = 2) {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(place.lon, place.lat, place.height),
    duration,
  });
}

// any point from the browser console, e.g. flyToDegrees(88.3022, 22.6462, 4000)
window.flyToDegrees = (lon, lat, height = 4000) => flyTo({ lon, lat, height });

const loaded = [];                   // layers that loaded, for the status line

function status(text, ok = false) {
  const line = document.getElementById('status');
  line.textContent = text;
  line.style.color = ok ? '#9d9' : '#f9b';      // green: information, pink: error
}

function showLoaded() {
  status(`loaded: ${loaded.join(', ')}`, true);
}

// ---- Sentinel-2 imagery of four years and the NDVI overlay (z/x/y PNG tiles) ----
function tileLayer(folder, alpha = 1.0) {
  const layer = viewer.imageryLayers.addImageryProvider(
    new Cesium.UrlTemplateImageryProvider({
      url: `/imagery/${folder}/{z}/{x}/{y}.png`,
      rectangle: STUDY_AREA,
      minimumLevel: 10,
      maximumLevel: 15,
    }),
  );
  layer.alpha = alpha;
  layer.show = false;
  return layer;
}

const imageryLayers = {};
YEARS.forEach((year) => {
  imageryLayers[year] = tileLayer(year);
});
const ndviLayer = tileLayer('ndvi_2023', 0.8);       // added last, so drawn on top
let currentYear = YEARS[YEARS.length - 1];           // most recent year by default

function showYear(year) {
  currentYear = year;
  const on = document.getElementById('layer-imagery').checked;
  YEARS.forEach((y) => {
    imageryLayers[y].show = on && y === year;
  });
  document.querySelectorAll('#years button').forEach((b) => {
    b.classList.toggle('active', Number(b.dataset.year) === year);
  });
}
window.showYear = showYear;      // callable from the console or another page

// ---- flood-risk zones (exported from PostGIS as flood_risk.geojson) ----
async function loadFloodZones() {
  const source = await Cesium.GeoJsonDataSource.load('/flood_risk.geojson', {
    clampToGround: true,
    fill: Cesium.Color.RED.withAlpha(0.4),
    stroke: Cesium.Color.RED,
  });
  source.entities.values.forEach((entity) => {
    const level = entity.properties.risk_level?.getValue();
    entity.name = `Flood risk: ${level}`;
    if (entity.polygon) {
      entity.polygon.material = FLOOD_COLOURS[level] ?? Cesium.Color.RED.withAlpha(0.4);
    }
  });
  viewer.dataSources.add(source);
  return source;
}

// ---- LiDAR buildings: footprints extruded by height_m (or the cylinder stand-in) ----
function heightColour(height) {
  const [, css] = HEIGHT_CLASSES.find(([limit]) => height < limit);
  return Cesium.Color.fromCssColorString(css);
}

async function loadBuildings() {
  const source = await Cesium.GeoJsonDataSource.load('/lidar_buildings.geojson');
  const now = Cesium.JulianDate.now();
  source.entities.values.forEach((entity) => {
    const height = entity.properties.height_m.getValue();
    entity.name = `LiDAR building cell: ${height.toFixed(1)} m`;
    if (BUILDING_MODE === 'cylinders') {
      // blueprint stand-in: a 15 m cylinder at the centre of the cell, height_m tall
      const positions = entity.polygon.hierarchy.getValue(now).positions;
      const centre = Cesium.Cartographic.fromCartesian(
        Cesium.BoundingSphere.fromPoints(positions).center);
      entity.position = Cesium.Cartesian3.fromRadians(centre.longitude, centre.latitude,
        height / 2);
      entity.cylinder = new Cesium.CylinderGraphics({
        length: height,
        topRadius: 15,
        bottomRadius: 15,
        material: Cesium.Color.SLATEGRAY,
        heightReference: Cesium.HeightReference.RELATIVE_TO_GROUND,
      });
      entity.polygon = undefined;
    } else {
      entity.polygon.material = heightColour(height);
      entity.polygon.outline = false;
      entity.polygon.height = 0;
      entity.polygon.heightReference = Cesium.HeightReference.CLAMP_TO_GROUND;
      entity.polygon.extrudedHeight = height;
      entity.polygon.extrudedHeightReference = Cesium.HeightReference.RELATIVE_TO_GROUND;
    }
  });
  viewer.dataSources.add(source);
  return source;
}

// ---- control panel ----
function buildLegend() {
  const items = [
    ...Object.entries(FLOOD_COLOURS).map(([level, colour]) => [
      colour.withAlpha(0.8).toCssColorString(), `flood risk ${level}`]),
    ...HEIGHT_CLASSES.map(([, css, text]) => [css, `building ${text}`]),
  ];
  document.getElementById('legend').innerHTML = items.map(([css, text]) =>
    `<div class="item"><span class="swatch" style="background:${css}"></span>`
    + `${text}</div>`)
    .join('');
}

function buildYearButtons() {
  document.getElementById('years').innerHTML = YEARS.map((y) =>
    `<button data-year="${y}">${y}</button>`).join('');
  document.querySelectorAll('#years button').forEach((b) => {
    b.addEventListener('click', () => showYear(Number(b.dataset.year)));
  });
}

async function loadLayer(loader, checkboxId, name) {
  try {
    const source = await loader();
    const box = document.getElementById(checkboxId);
    source.show = box.checked;
    box.addEventListener('change', () => { source.show = box.checked; });
    loaded.push(`${name} (${source.entities.values.length.toLocaleString()})`);
    showLoaded();
    console.info(`${name}: ${source.entities.values.length} entities loaded`);
    return source;
  } catch (error) {
    status(`${name} not loaded: ${error.message ?? error}`);
    console.error(`${name} not loaded`, error);
    return null;
  }
}

buildLegend();
buildYearButtons();
document.querySelectorAll('[data-fly]').forEach((b) => {
  b.addEventListener('click', () => flyTo(PLACES[b.dataset.fly]));
});
document.getElementById('layer-imagery')
  .addEventListener('change', () => showYear(currentYear));
const ndviBox = document.getElementById('layer-ndvi');
ndviBox.addEventListener('change', () => { ndviLayer.show = ndviBox.checked; });
showYear(currentYear);
flyTo(PLACES.kolkata, 0);

const [floodZones, buildings] = await Promise.all([          // both load in parallel
  loadLayer(loadFloodZones, 'layer-flood', 'flood_risk.geojson'),
  loadLayer(loadBuildings, 'layer-buildings', 'lidar_buildings.geojson'),
]);

// ---- the agent's latest result (Exercise 5): fly there when a new one arrives ----
const SERVER_URL = import.meta.env.VITE_MCP_SERVER_URL || 'http://localhost:8000';
const agentLayer = new Cesium.CustomDataSource('agent result');
viewer.dataSources.add(agentLayer);
let lastResultId = null;

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text ?? '';
  return div.innerHTML;
}

function marker(lon, lat, text, colour, size) {
  return {
    position: Cesium.Cartesian3.fromDegrees(lon, lat),
    point: {
      pixelSize: size,
      color: colour,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 2,
      heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text,
      font: '14px sans-serif',
      pixelOffset: new Cesium.Cartesian2(0, -20),
      showBackground: true,
      heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  };
}

function showAgentResult(result) {
  agentLayer.entities.removeAll();
  agentLayer.entities.add({
    name: `Agent: ${result.question}`,
    description: `<p><b>${escapeHtml(result.question)}</b></p>`
      + `<p>${escapeHtml(result.answer)}</p><p>Tools: ${result.tools.join(', ')}</p>`,
    ...marker(result.longitude, result.latitude, 'agent', Cesium.Color.YELLOW, 14),
  });
  (result.sites || []).forEach((site) => agentLayer.entities.add({
    name: `Similar site #${site.rank} (score ${site.opportunity_score})`,
    ...marker(site.longitude, site.latitude, `#${site.rank}`, Cesium.Color.CYAN, 10),
  }));
  document.getElementById('agent').innerHTML = `<b>${escapeHtml(result.question)}</b>`
    + `<br>${result.latitude.toFixed(4)}, ${result.longitude.toFixed(4)} - `
    + `${escapeHtml(result.tools.join(', '))} (${result.received.slice(11)})`;
}

function flyToResult(result) {
  // 800 m above the place the agent discussed; when similar sites were found, 4 x the
  // distance of the farthest one (up to 5 km -> 20 km), so every #rank marker is in view
  const sites = result.sites || [];
  const here = Cesium.Cartographic.fromDegrees(result.longitude, result.latitude);
  const geodesic = new Cesium.EllipsoidGeodesic();
  const reach = Math.max(0, ...sites.map((site) => {
    const there = Cesium.Cartographic.fromDegrees(site.longitude, site.latitude);
    geodesic.setEndPoints(here, there);
    return geodesic.surfaceDistance;
  }));
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(result.longitude, result.latitude,
      Math.max(800, 4 * reach)),
  });
}

async function pollAgentResult() {
  try {
    const response = await fetch(`${SERVER_URL}/agent/latest_result`);
    const result = await response.json();
    showLoaded();                                 // clears an earlier fetch error
    if (result.id === undefined || result.id === lastResultId) {
      return;                                     // no result yet, or nothing new
    }
    lastResultId = result.id;
    showAgentResult(result);
    flyToResult(result);
  } catch (error) {
    status(`agent result not reachable: ${error.message}`);    // server down, or CORS
  }
}

setInterval(pollAgentResult, 3000);
pollAgentResult();

export { viewer, floodZones, buildings, flyTo };
