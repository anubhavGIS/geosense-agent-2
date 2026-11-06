# Redeploying from Scratch

GeoSense Agent 2.0 runs on five platforms: the backend on **Cloud Run** (image in
**Artifact Registry**, secrets in **Secret Manager**, twin assets in **Cloud Storage**), the
database on **Neon** (PostgreSQL 18 + PostGIS), the digital twin on **GitHub Pages** and the
Streamlit app on **Streamlit Community Cloud**. Every step below is a script of this
repository; run them from the project root in the `geosense` environment (Windows prompt).
No secret is typed on a command line: the scripts read `.env` and pass values on standard
input or to the clipboard.

## 0. What must exist first

- A Google Cloud project with billing, the Cloud Run, Artifact Registry and Cloud Build APIs,
  and a Docker repository `geosense-repo` in `us-central1` (Exercise 1).
- `.env` with `CLOUD_DATABASE_URL` (Neon, direct host), `GEE_PROJECT`, `MAPILLARY_TOKEN`,
  `CESIUM_ION_TOKEN` and the study-area box; the database dump `deploy/geosense_db.sql.gz`.
- The Earth Engine service-account key in `%USERPROFILE%\.config\geosense\gee-sa-key.json`
  (create a new key if the old one was deleted).

## 1. Database

    python deploy\restore_cloud_db.py              (PostGIS enabled, five tables, counts checked)

## 2. Secrets, service account and settings

    python deploy\cloud_setup.py prepare           (APIs, runtime identity, 3 secrets, env file)

## 3. Backend image and service

    deploy\deploy_backend.bat v1                   (docker build, docker push, gcloud run deploy)
    python deploy\cloud_setup.py envfile --cors https://anubhavgis.github.io
    gcloud run services update geosense-backend --project=geosense-agent-2 --region=us-central1 --env-vars-file=deploy\cloudrun.env.yaml

The same in four commands, for one image:

    1. Build:  docker build -f deploy/Dockerfile.backend -t <image-tag> .
    2. Push:   docker push <image-tag>
    3. Deploy: gcloud run deploy geosense-backend --image <image-tag> --region us-central1
    4. Verify: curl https://<service-url>/docs

## 4. The twin's assets and the twin

    deploy\upload_twin_assets.bat                  (public bucket, CORS, 1,995 tiles, 2 layers)
    python deploy\cloud_setup.py twin-env          (.env.production: backend and bucket URLs)

Then push to `main`: the workflow `twin.yml` builds and publishes the twin (repository
Settings, Pages, Source: GitHub Actions; secret `CESIUM_ION_TOKEN`).

## 5. Streamlit app

share.streamlit.io, Create app: repository `anubhavGIS/geosense-agent-2`, branch `main`,
main file `src/phase4_agent/app.py`, Python 3.11, secrets `MCP_SERVER_URL`,
`GOOGLE_API_KEY`, `AGENT_MODEL`, `TWIN_URL`.

## 6. The pipeline

    python deploy\ci_setup.py account              (github-actions-deployer, 3 roles, key)
    python deploy\ci_setup.py clip key             (paste as GCP_SA_KEY; GCP_PROJECT_ID too)
    python deploy\ci_setup.py delete-key

From then on every push to `main` that changes the code runs `deploy.yml`: the offline
tests, then `backend:<commit sha>` built on the released image and deployed. A change of
the requirements, the models or the data needs a new full image from step 3
(`deploy_backend.bat v3`) and `BASE_TAG: v3` in `.github/workflows/deploy.yml`.

## 7. Verify

    python deploy\cloud_check.py                   (/health twice, Earth Engine, CORS)
    python deploy\smoke_test.py --url <service-url>
    python tests\test_load.py                      (20 sequential calls, a burst of 10)
    python deploy\cloud_setup.py status            (revisions, traffic, secrets, bucket)

## Rolling back

    gcloud run revisions list --service=geosense-backend --project=geosense-agent-2 --region=us-central1
    gcloud run services update-traffic geosense-backend --project=geosense-agent-2 --region=us-central1 --to-revisions=<revision>=100
    gcloud run services update-traffic geosense-backend --project=geosense-agent-2 --region=us-central1 --to-latest

`/health` names the revision that answers. After a rollback the traffic stays on that
revision until `--to-latest` is given again.
