# Cloud secrets checklist — GeoSense Agent 2.0, Phase 5 (Part D.3)

Every value the deployed system needs at runtime, where it is created, where it lives on the laptop and where it will live in the cloud. Values are never written here, in a Dockerfile, in docker-compose.yml or in any committed file. Server-side secrets go into Cloud Run environment variables or Secret Manager; the frontend's go into Streamlit Community Cloud's Secrets; the pipeline's into GitHub Actions secrets. The only value that reaches a browser by design is the Cesium Ion token, which is a client-side token limited to reading assets and geocoding and is restricted to the twin's domain in the Cesium Ion dashboard.

| # | Name | Used by | Created in | On the laptop | In the cloud (Exercise 3–4) | Kind |
|---|------|---------|------------|---------------|-----------------------------|------|
| 1 | GOOGLE_API_KEY | GeoAI agent (runs inside the Streamlit frontend) | Google AI Studio, project geosense-agent-phase4 (Phase 4; regenerated after Phase 4) | .env | Streamlit Community Cloud → Secrets | server-side secret |
| 2 | AGENT_MODEL | GeoAI agent | setting, not a secret (google_genai:gemini-3.5-flash-lite) | .env | Streamlit Secrets | setting |
| 3 | MCP_SERVER_URL | Streamlit app; the twin's build (VITE_MCP_SERVER_URL) | — | .env (http://localhost:8000) | Streamlit Secrets; twin build argument (the Cloud Run URL) | setting |
| 4 | DB_HOST, DB_PORT, DB_NAME, DB_USER | MCP server (site scorer, similar sites, hospitals, flood zones) | Phase 1 PostGIS; the cloud Postgres of Exercise 3 | .env | Cloud Run environment variables | setting |
| 5 | DB_PASSWORD | MCP server | Phase 1 (laptop) / Exercise 3 (cloud database) | .env | Cloud Run secret | server-side secret |
| 6 | DB_SSLMODE | MCP server | new in Phase 5 (require for the cloud database, absent locally) | — | Cloud Run environment variable | setting |
| 7 | MAPILLARY_TOKEN | streetview_scene tool | mapillary.com developer dashboard (Phase 3) | .env | Cloud Run secret | server-side secret |
| 8 | GEE_PROJECT | get_climate_risk, get_demographics (Earth Engine) | Earth Engine registration (Phase 2) | .env | Cloud Run environment variable | setting |
| 9 | GEE_SERVICE_ACCOUNT_KEY | Earth Engine sign-in without a browser | Google Cloud service account registered for Earth Engine (Exercise 3) | not stored (the laptop uses the interactive sign-in) | Cloud Run secret (JSON) | server-side secret |
| 10 | CORS_ORIGINS | MCP server | Phase 4 (localhost defaults) | not set | Cloud Run environment variable with the live frontend origins | setting |
| 11 | CESIUM_ION_TOKEN → VITE_CESIUM_ION_TOKEN | digital twin (browser) | Cesium Ion (Phase 4) | .env → frontend/digital_twin/.env | build-time variable of the twin (GitHub Actions secret for the Pages build) | client-side token, public by design |
| 12 | GCP_SA_KEY, GCP_PROJECT_ID | GitHub Actions deploy job | service account github-actions-deployer (Exercise 4) | never kept (key.json deleted after upload) | GitHub repository secrets | CI secret |
| 13 | OPENTOPOGRAPHY_API_KEY, SAM_CHECKPOINT_PATH, STUDY_AREA_BBOX_LIDAR | Phase 3 scripts only | Phase 3 | .env | not deployed | — |
| 14 | ANTHROPIC_API_KEY | optional model switch | — | commented out | not deployed | — |

Checks before deploying: `.env` and `frontend/digital_twin/.env` are git-ignored (`git check-ignore -v`); `deploy/key.json` and `.streamlit/secrets.toml` are in the Phase 5 .gitignore block; `check_deploy_toolchain.py` lists every name above as "set" without showing a value.
