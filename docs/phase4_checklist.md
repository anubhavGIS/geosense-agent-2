# Phase 4 Completion Checklist (Part H.1) — with evidence

Project: GeoSense Agent 2.0 · Study area: Kolkata (88.30–88.42 E, 22.45–22.67 N) · LiDAR tile: Dupont Circle, Washington, D.C. · Environment: conda `geosense` (Python 3.11.15, Windows 11, RTX 5060 Laptop GPU), Node.js 24.21.0 / npm 11.19.0 · Language model: Gemini API free tier (`google_genai:gemini-3.5-flash-lite`) through LangChain; the agent is model-agnostic (`AGENT_MODEL`).
Record references (Ex. n) point to the Phase IV lab records, Exercises 1–5.

## Software Installation

| Done | Item | Evidence |
|---|---|---|
| ☑ | Language-model API key generated and stored in `.env` | Google AI Studio project `geosense-agent-phase4`, key stored as `GOOGLE_API_KEY`; `ANTHROPIC_API_KEY` is the optional Claude switch; keys are printed only masked by `test_llm_connection.py` (Ex. 1) |
| ☑ | MCP SDK installed — `import mcp` works | mcp 2.3.0; the server publishes the seven tools on `/mcp` (protocol 2026-07-28) and `mcp_client_demo.py` lists and calls them through the SDK (Ex. 1, Ex. 2) |
| ☑ | FastAPI and Uvicorn installed — server starts without error | fastapi 0.142.2, uvicorn 0.54.0; "Uvicorn running on http://0.0.0.0:8000" (Ex. 2) |
| ☑ | Postman installed and able to send test requests | collection `GeoSense MCP Server` — 23 requests, 71 assertions, Runner 71/71 in 24.6 s; exported to `docs/postman/` (Ex. 2) |
| ☑ | LangGraph and the LangChain model integrations installed | langgraph 1.2.12, langchain 1.4.3, langchain-google-genai 4.4.0, langchain-anthropic 1.7.5 (Ex. 1) |
| ☑ | Node.js 18+ installed — `node --version` and `npm --version` both work | v24.21.0 / 11.19.0 (Ex. 1) |
| ☑ | Cesium Ion account created and access token stored in `.env` | community account, default token (assets:read, geocode) as `CESIUM_ION_TOKEN`; `make_twin_env.py` copies it into `frontend/digital_twin/.env` as `VITE_CESIUM_ION_TOKEN` (Ex. 1, Ex. 4) |
| ☑ | Streamlit installed — `streamlit hello` opens a browser tab | streamlit 1.65.0 (Ex. 1) |

## Environment Setup

| Done | Item | Evidence |
|---|---|---|
| ☑ | Phase 4 folder structure created | `src/phase4_agent/` (+ `tools/`), `frontend/digital_twin/{public,src}`, `outputs/agent_logs/` (Ex. 1) |
| ☑ | CesiumJS Vite project scaffolded with vite-plugin-cesium installed | create-vite 9.2.1 vanilla template; `package.json`: vite 8.3.2, cesium 1.146.0, vite-plugin-cesium 1.2.23 (Ex. 1) |
| ☑ | `.env` files updated with the model key, CESIUM_ION_TOKEN, MCP_SERVER_URL | 19 keys in the project `.env` (`GOOGLE_API_KEY`, `CESIUM_ION_TOKEN`, `MCP_SERVER_URL=http://localhost:8000`, `AGENT_MODEL`); `frontend/digital_twin/.env` holds only `VITE_CESIUM_ION_TOKEN` (Ex. 1, Ex. 4) |
| ☑ | `.gitignore` updated to exclude node_modules/, dist/ and the frontend `.env` | plus `outputs/agent_logs/`, `data/phase4_cache/`, `data/twin/` and `frontend/digital_twin/public/imagery/`; `git status` shows no token or key (Ex. 1, Ex. 5) |

## Data Layer Preparation

| Done | Item | Evidence |
|---|---|---|
| ☑ | Flood risk GeoJSON exported from PostGIS and saved to `frontend/digital_twin/public/` | `build_flood_zones.py` builds `flood_zones` (MERIT Hydro HAND + JRC Global Surface Water, 30 m): high 197.9 km² in 464 polygons, medium 85.0 km² in 839, low 15.5 km² in 240; the Section D.2 `ST_AsGeoJSON` query in DBeaver → `flood_risk.geojson` (786 KB), valid in QGIS; Park Street lies in a high zone (Ex. 4) |
| ☑ | LiDAR building GeoJSON exported and saved to `frontend/digital_twin/public/` | `export_lidar_geojson.py`: 2,898 ten-metre cells with ≥ 50 % building points, EPSG:32618 → WGS 84, heights 2.9–51.3 m; the first version (grid indices as coordinates) put 331 points in the Gulf of Guinea — kept as the record's before/after (Ex. 4) |
| ☑ | At least one historical imagery layer prepared for the time slider | four years — 2018, 2021, 2023, 2025 — plus an NDVI 2023 overlay, 399 tiles each at zoom 10–15 (`prepare_imagery.py --tile`) (Ex. 4) |

## MCP Server

| Done | Item | Evidence |
|---|---|---|
| ☑ | `mcp_server.py` runs without error — Uvicorn starts on port 8000 | banner lists the seven tools, torch 2.11.0+cu128 on the RTX 5060, REST / docs / MCP addresses and the CORS origins (Ex. 2, Ex. 5) |
| ☑ | All 7 tools visible and testable at http://localhost:8000/docs | seven POST endpoints under the tags "Phase I–III modules" and "Context and search tools", each executed from the page (Ex. 2) |
| ☑ | Each tool tested individually in Postman and returns a correct JSON result | 23 requests incl. coverage checks (Chennai → `available: false`) and the 422 cases; `call_all_tools.py` 17 calls, 0 unexpected (Ex. 2) |

## GeoAI Agent

| Done | Item | Evidence |
|---|---|---|
| ☑ | `geoai_agent.py` runs — `ask_agent()` returns a readable, evidence-based answer | e.g. "not safe from flooding; classified as being in a high flood zone … 1,767 mm against 1,651 mm (+7.0 %)" with the tool named for every number; transcripts in `outputs/agent_logs/` (Ex. 3, Ex. 5) |
| ☑ | Agent correctly selects at least one relevant tool for a clear test question | six test questions × three prompt styles: described prompt 6 of 6 exact, basic 5 of 6, vague 1 of 6 (`evaluate_tool_selection.py`) (Ex. 3) |
| ☑ | Agent remembers earlier context within the same `thread_id` (memory works) | `memory_test.py`: "What about its flood risk?" answers about the demo point without memory and about 22.553, 88.354 ("carried over from the previous question") with memory; the same in Streamlit for "What about its walkability?" (Ex. 3, Ex. 5) |

## 3D Digital Twin

| Done | Item | Evidence |
|---|---|---|
| ☑ | CesiumJS globe loads with real world terrain and satellite base imagery | `Cesium.Terrain.fromWorldTerrain()` + Bing imagery through Cesium ion; the manual's `createWorldTerrain` call does not exist in 1.146 and was the first troubleshooting record (Ex. 4) |
| ☑ | Flood risk and LiDAR building layers render without error | 1,543 flood polygons clamped to the ground, 2,898 extruded buildings coloured by height class; console clean apart from Cesium's one-time outline notice; status line "loaded: flood_risk.geojson (1,543), lidar_buildings.geojson (2,898)" (Ex. 4) |
| ☑ | Time slider or year selector switches between historical imagery layers | buttons 2018 / 2021 / 2023 / 2025 and the NDVI 2023 toggle; the loss and gain hotspots of Phase 2 as fly-to buttons (Ex. 4) |
| ☑ | Camera flies to a location automatically when the agent produces a result | `POST /agent/latest_result` after every answer, the twin polls `GET` every 3 s and flies only when the id changes; markers for the agent's point and the similar sites (Ex. 5) |

## Streamlit Demo

| Done | Item | Evidence |
|---|---|---|
| ☑ | `app.py` runs — `streamlit run` opens a working demo page | http://localhost:8501, wide layout, title "GeoSense Agent 2.0 — Conversational Geospatial Intelligence" (Ex. 5) |
| ☑ | Question box triggers the agent and displays its answer | `st.success` answer, caption with location, source, turn, tokens, seconds and "sent to the twin", table of the tools used, "Earlier questions" expander, "New conversation" button (Ex. 5) |
| ☑ | 3D twin is embedded and visible inside the Streamlit page | `st.iframe('http://localhost:5173', height=600)`; the embedded twin flies to the answer's location (Ex. 5) |

## Core Testing

| Done | Item | Evidence |
|---|---|---|
| ☑ | All 4 unit tests pass — `pytest tests/ -v` shows PASSED (with MCP server running) | `tests/test_phase4_agent.py`: the four tests of Section G.1 + every tool 200 with `available` (7), three 422 payloads, the latest-result round trip and a mocked-model test = 16 passed in 54 s; the whole folder 32 passed in 68 s (Ex. 5) |

## GitHub Portfolio

| Done | Item | Evidence |
|---|---|---|
| ☑ | All Phase 4 code committed to the existing geosense-agent-2 repository | `src/phase4_agent/` (13 scripts + `tools/`), `frontend/digital_twin/` (without node_modules, `.env` and the tile pyramids), `tests/test_phase4_agent.py`, `docs/postman/`, example transcripts (Ex. 5) |
| ☑ | README.md updated with a Phase 4 section describing the agent and digital twin | architecture table, the seven tools, the agent and its tool-selection measurements, an example conversation, the data layers, the twin, how to start the three services (Ex. 5) |
| ☑ | A screenshot or short GIF of the 3D twin added to the README | `outputs/digital_twin/twin_dupont_buildings.jpg`, `twin_agent_similar_sites.jpg`, `twin_timeslider_2018_2025.jpg`, `streamlit_demo.jpg` (Ex. 5) |
| ☑ | At least 5 meaningful Phase 4 commits with descriptive messages | Add: / Improve: / Complete: commits for the setup, the server, the real tools, the agent, the data layers, the twin, the integration and the documentation (see `git log`) |

## Common errors of Section G.2 — cause and fix in my own words

| Error | What it means | Fix |
|---|---|---|
| "blocked by CORS policy" in the browser console | the twin (origin `http://localhost:5173`) asked the server (origin `http://localhost:8000`) for `/agent/latest_result`; the server answered 200 but without an `Access-Control-Allow-Origin` header for that origin, so the browser discarded the response — the request itself succeeded, which is why the server log still shows 200 | add the exact origin, port included, to `CORSMiddleware` (`allow_origins`) and restart; reproduced with `CORS_ORIGINS=http://localhost:8501` (Streamlit only) and fixed by restoring both origins (Ex. 5) |
| wrong or empty tool selection | the model saw only tool names: "Which of these tools would you use?" produced prose with no names (1 of 6 right), and the bare list added `get_demographics` to a clinic question | describe every tool — what it measures, from which data, where it applies — and ask for a fixed reply format (JSON); 6 of 6 afterwards; keep only names that are in the catalogue (Ex. 3) |
| blank or grey globe, 401 in the console | the Cesium ion token is wrong or missing, so World Terrain and the Bing imagery (both ion assets) fail with 401; local layers still draw, floating over a black globe | correct `VITE_CESIUM_ION_TOKEN` in `frontend/digital_twin/.env` (the `VITE_` prefix is what exposes it to the browser) and restart `npm run dev` — Vite reads `.env` at start-up; reproduced with `make_twin_env.py --break` (Ex. 4) |
| invalid API key (401 UNAUTHENTICATED / AuthenticationError) | the model provider rejected the key: altered, expired or the wrong variable name | restore the key in `.env` and start a new Python process — the key is read once at import; reproduced with `test_llm_connection.py --bad-key` (Ex. 1) |

## Other errors met and fixed in practice

| Error | Cause | Fix |
|---|---|---|
| `OSError: [WinError 127] … torch\lib\shm.dll` at server start | `import torch` came after site_scorer's imports (shap, scikit-learn, geopandas) had loaded another version of a native library torch depends on | import torch first in `mcp_server.py` (Ex. 2) |
| `ModuleNotFoundError: No module named 'site_scorer'` | the server was started from `src/phase4_agent`; the `sys.path.append('src/…')` lines are relative to the project root | start every script from the project root (Ex. 2) |
| `429 RESOURCE_EXHAUSTED … limit: 20` and `503 UNAVAILABLE` from the model | the free tier of `gemini-3.8-flash` allows 20 requests a day per project; the model is sometimes overloaded | `AGENT_MODEL=google_genai:gemini-3.5-flash-lite` (500 a day); `ask_model()` retries a 503 or per-minute 429 once and turns anything else into a fallback answer built from the raw tool facts (Ex. 3) |
| `httpx.ConnectError` in `execute_tools` | the MCP server was not running | `call_tool()` returns an error entry ("MCP server not reachable …") that the answer reports; restart the server (Ex. 3) |
| HTTP 422 from `lidar_profile` | the agent sent a point to a tool that expects a bounding box | `tool_payload()` builds a ± 0.0015° box around the point for that tool (Ex. 3) |
| "walkability 1.7 out of 100" in an answer | the model guessed a scale | the synthesis prompt now states every scale (scores 0–10, drought 0–1, shares in %) (Ex. 3) |
| a follow-up about Park Street answered for Boston | the model copied the place name from the earlier turn without the city and the geocoder's answer outranked the remembered location | a place is geocoded only when its words appear in the question itself; otherwise the previous location is carried over; Nominatim is biased towards the last location (Ex. 5) |
| `Cesium.createWorldTerrain is not a function` | removed from CesiumJS; 1.146 has `Terrain.fromWorldTerrain()` / `createWorldTerrainAsync()` | `terrain: Cesium.Terrain.fromWorldTerrain()` in the Viewer options (Ex. 4) |
| the twin's buildings stayed hidden after a reload | Chrome restores the previous tick state of check boxes on reload | `autocomplete="off"` on the layer check boxes; a status line now reports what loaded (Ex. 4) |
| `Please replace st.components.v1.iframe with st.iframe` | Streamlit 1.65 deprecates the components call | `st.iframe(...)` (Ex. 5) |

## Section I in practice

- Naming: `mcp_server.py`, `geoai_agent.py`, `export_lidar_geojson.py`, `build_flood_zones.py`, `prepare_imagery.py`, `make_twin_env.py`, `app.py`; `main.js`, `vite.config.js`; `flood_risk.geojson`, `lidar_buildings.geojson`; folders `phase4_agent`, `digital_twin`, `agent_logs`.
- Daily routine: Anaconda Prompt → `conda activate geosense` → project root → MCP server, then `npm run dev`, then Streamlit; every tool tested alone (`/docs`, Postman, `call_all_tools.py`) before the agent used it.
- Backup: code on GitHub (never `.env`, `node_modules` or the tile pyramids), the project folder on the laptop, the lab records with every script version, screenshot and note under `lab_records/`.
- Five-step troubleshooting on a real error (the Boston follow-up): read the caption — location "Park Street" geocoded with Nominatim, 42.356 N; find where it comes from — `resolve_location()` prefers a geocoded place over the previous turn; reproduce — the selection prompt returned `"place": "Park Street"` for a question that names no place; fix — geocode only a place written in the question, bias the geocoder towards the last location; verify — the follow-up now reads "carried over from the previous question" and returns Park Street's walkability.
