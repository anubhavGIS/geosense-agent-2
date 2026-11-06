# Portfolio screenshots (Task D.1 / D.2)

| File | What it shows | Phase |
|---|---|---|
| `architecture_diagram.png` | The five layers of the system — user interface, agent, MCP server, analysis modules, data and cloud — with where each part runs since Phase 5 and the delivery pipeline; exported from `../architecture_diagram.drawio` with draw.io | 5 |
| `01_live_demo.png` | The live demo app on Streamlit Community Cloud: the question, the agent's answer, the tools it called on Cloud Run with their times, and the 3D twin below it | 4–5 |
| `02_shap_site_score.png` | SHAP importance of the Random Forest site scorer: distance to the nearest hospital first, road distance second, flood risk third | 1 |
| `03_land_cover.jpg` | Four Sentinel-2 chips of Kolkata: true colour, the rule-based labels, the U-Net prediction and the fine-tuned Prithvi-100M prediction (urban, vegetation, water, bare land, agriculture) | 2 |
| `04_lidar_point_cloud.jpg` | The classified USGS 3DEP point cloud of Dupont Circle, Washington, D.C.: ground in brown, every other point coloured by its height above ground, and an east–west section | 3 |
| `05_street_scenes.jpg` | Three Mapillary photographs of the Park Street area segmented by SegFormer (road, sidewalk, building, vegetation, sky, car) with the share of each class | 3 |
| `06_twin_buildings.jpg` | The CesiumJS digital twin at Dupont Circle: 2,898 LiDAR building cells extruded by their height | 4 |
| `07_twin_time_slider.jpg` | The twin's Sentinel-2 years at the vegetation-loss hotspot on the Howrah side, north-west of central Kolkata: 2018 against 2025 | 4 |
| `08_twin_agent_result.jpg` | The twin after an answer about 22.5450, 88.3600: the agent's site (yellow) and the five most similar candidate sites (cyan, ranked) over the flood-risk zones | 4 |
