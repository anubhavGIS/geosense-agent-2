# Phase 5 — deliverables checklist (Part H.1) with evidence

The final checklist of GeoSense Agent 2.0: cloud deployment, CI/CD, portfolio and job launch.
"Ex n" is the exercise of Phase 5 in which the item was done (Ex 1 toolchain, Ex 2 containers,
Ex 3 Google Cloud deployment, Ex 4 CI/CD and load test, Ex 5 portfolio and release).

## Software installation

| Item | Status | Evidence |
|---|---|---|
| Docker Desktop installed and running — `docker run hello-world` succeeds | Done (Ex 1) | Docker Desktop 4.98.0, engine 29.9.2, Compose v5.6.1 on WSL 2; hello-world ran; `check_deploy_toolchain.py` 35 checks passed |
| Google Cloud SDK installed — `gcloud --version` shows a version | Done (Ex 1) | Google Cloud SDK 592.0.0; default region us-central1 |
| Billing enabled; Cloud Run and Artifact Registry APIs enabled | Done (Ex 1) | project `geosense-agent-2`, free-trial billing with a ₹500/month budget alert; run, artifactregistry, cloudbuild APIs on; Docker repository `geosense-repo` (us-central1) |
| Streamlit Community Cloud connected to the GitHub repository | Done (Ex 1, Ex 3) | workspace linked to `anubhavGIS/geosense-agent-2`; the app deploys from `main` |
| OBS Studio or Loom installed for the demo video | Done (Ex 1) | OBS Studio 33.0.1 (NVENC H.264); used for the Ex 5 recording |

## Containerisation

| Item | Status | Evidence |
|---|---|---|
| `Dockerfile.backend` builds | Done (Ex 2) | `geosense-backend:local`, 953 MB compressed, built in 262.5 s; CPU PyTorch, models and data inside |
| `Dockerfile.frontend` builds | Done (Ex 2) | `geosense-frontend:local`, 217 MB compressed, 59.3 s |
| `docker-compose.yml` runs the full stack locally | Done (Ex 2) | PostGIS 18-3.6 restored from the project dump, backend and Streamlit; smoke test of the seven tools and 16 agent tests passed against the containers |
| Requirements trimmed to the packages needed | Done (Ex 2) | `deploy/requirements-backend.in` (29 packages) and `requirements-frontend.in` (10), pinned to the versions the laptop runs |

## Cloud deployment

| Item | Status | Evidence |
|---|---|---|
| Backend image pushed to Artifact Registry | Done (Ex 3) | `us-central1-docker.pkg.dev/geosense-agent-2/geosense-repo/backend:v1`, `:v2`; CI images `backend:<commit>` since Ex 4 |
| Backend on Cloud Run with a public URL, reachable at `/docs` | Done (Ex 3) | https://geosense-backend-917306622260.us-central1.run.app/docs — 2 GiB, 2 vCPU, 0–3 instances; Postman/Newman 71 of 71 tests passed against it |
| Frontend deployed with a public URL | Done (Ex 3) | Streamlit app https://geosense-agent.streamlit.app; digital twin https://anubhavgis.github.io/geosense-agent-2/ |
| Secrets configured in the cloud environment | Done (Ex 3, Ex 4) | Secret Manager: database password, Mapillary token, Earth Engine service-account key; Streamlit secrets: server URL, Gemini key, model, twin URL; GitHub secrets: deployer key, project id, Cesium token; none in the repository |

## CI/CD pipeline

| Item | Status | Evidence |
|---|---|---|
| `.github/workflows/deploy.yml` created and committed | Done (Ex 4) | commit f6b941a; job `test` (13 offline tests) → job `deploy` (build on the released image, push, deploy, `/health`); `twin.yml` publishes the twin |
| GCP service account with the correct IAM role | Done (Ex 4) | `github-actions-deployer`: Cloud Run Developer, Service Account User on the runtime account, Artifact Registry Writer on `geosense-repo` |
| `GCP_SA_KEY` and `GCP_PROJECT_ID` as GitHub Actions secrets | Done (Ex 4) | both set (plus `CESIUM_ION_TOKEN`); the local key file deleted afterwards |
| A test push triggers the test job and the deploy job | Done (Ex 4) | run #1: test 1 min 59 s, deploy 2 min 48 s → revision 00006-nkm; run #2 with a deliberately failing test: deploy skipped; run #3 after the revert → 00007-4wt |

## Portfolio content

| Item | Status | Evidence |
|---|---|---|
| Architecture diagram in `docs/portfolio/screenshots/` | Done (Ex 5) | `architecture_diagram.png`, exported with draw.io from `docs/portfolio/architecture_diagram.drawio` |
| 5–8 screenshots from the outputs of Phases 1–4 | Done (Ex 5) | eight images with captions in `docs/portfolio/screenshots/` |
| 5-minute demo video recorded and uploaded | Done (Ex 5) | a 3-minute captioned walkthrough of the live system, YouTube (unlisted): https://youtu.be/Ovwtpb2bGgA; the five-minute live-demo script: `docs/portfolio/demo_script.md` |
| README rewritten with live demo link, diagram, tech stack and screenshots | Done (Ex 5) | `README.md` with workflow, licence and Python badges; the detailed phase documentation moved to `docs/project_phases.md` |
| 3-part blog series written and published | Done (Ex 5) | three articles of about 750–900 words, published in the repository and linked from the README: `docs/portfolio/blog/part1_geoai.md`, `part2_mcp.md`, `part3_digital_twin.md` |

## Job search launch

| Item | Status | Evidence |
|---|---|---|
| LinkedIn portfolio post with the demo GIF and hashtags | Prepared | the text with the six hashtags and the GIF are ready (`docs/portfolio/linkedin_post.md`, `docs/portfolio/demo.gif`); kept for the start of the applications — the GIF is shown in the README |
| Connected with at least 20 geospatial professionals | Planned | contacts are added with the post, with a short personal note |
| Upwork and Arc.dev profiles | Prepared | profile texts written; accounts to be opened |
| CV updated with GeoSense Agent 2.0 as the flagship project | Prepared | the project entry is written |
| Cover letter template for geospatial / GeoAI roles | Done (Ex 5) | template written |
| At least 10 targeted applications | Planned | a tracker for ten applications is ready |

## Testing

| Item | Status | Evidence |
|---|---|---|
| Load test against the live deployment — 0 errors, average under 5 s | Done (Ex 4) | `tests/test_load.py`: 20 sequential site scores at Park Street, 0 errors, average 3.44 s, 95th percentile 4.52 s; a burst of 10 at once queued (average 16.4 s) because the server runs one analysis at a time |
| v1.0.0 Git tag created and pushed | Done (Ex 5) | annotated tag `v1.0.0`, release https://github.com/anubhavGIS/geosense-agent-2/releases/tag/v1.0.0 |

## Beyond the checklist

- The digital twin on GitHub Pages, built and published by its own workflow; its 1,995 imagery tiles in
  a public Cloud Storage bucket with CORS.
- PostGIS on Neon (PostgreSQL 18.6, PostGIS 3.6.4) restored from the laptop's dump; Earth Engine
  through a service account; the MCP endpoint `/mcp` reachable on Cloud Run.
- A cold start measured (71 s) and a rollback practised: traffic to revision 00005-mv9 and back to the
  latest; the steps are in `docs/redeploy_from_scratch.md`.
