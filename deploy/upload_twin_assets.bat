@echo off
rem upload_twin_assets.bat - Phase 5, Exercise 3 (Task E.3.3): the digital twin's imagery tiles
rem   (5 layers x 399 PNG tiles, about 221 MB) and its two GeoJSON layers go to a public
rem   Cloud Storage bucket; the GitHub Pages build of the twin loads them from there
rem   (VITE_ASSET_BASE_URL), so the 221 MB never enter the repository.
rem   Run from the project root:   deploy\upload_twin_assets.bat
rem   Safe to run again: the bucket is created only once, the files are copied over.
setlocal
set PROJECT_ID=geosense-agent-2
set REGION=us-central1
set BUCKET=gs://geosense-agent-2-twin
set TWIN=frontend\digital_twin\public

if not exist %TWIN%\imagery (
  echo %TWIN%\imagery not found - run this from the project root
  exit /b 1
)

echo [1/5] bucket %BUCKET% in %REGION%
call gcloud storage buckets describe %BUCKET% --project=%PROJECT_ID% --format="value(name)" >nul 2>&1
if errorlevel 1 (
  call gcloud storage buckets create %BUCKET% --project=%PROJECT_ID% --location=%REGION% --uniform-bucket-level-access --no-public-access-prevention
  if errorlevel 1 goto :fail
) else (
  echo       exists
)

echo [2/5] public read access ^(allUsers = Storage Object Viewer^)
call gcloud storage buckets add-iam-policy-binding %BUCKET% --project=%PROJECT_ID% --member=allUsers --role=roles/storage.objectViewer --format="value(etag)"
if errorlevel 1 goto :fail

echo [3/5] CORS: the twin's origins may read the files from the browser
call gcloud storage buckets update %BUCKET% --project=%PROJECT_ID% --cors-file=deploy\twin_bucket_cors.json
if errorlevel 1 goto :fail

echo [4/5] imagery tiles (started %time%)
call gcloud storage cp --recursive %TWIN%\imagery %BUCKET% --project=%PROJECT_ID% --cache-control="public, max-age=86400"
if errorlevel 1 goto :fail
echo       finished %time%

echo [5/5] GeoJSON layers
call gcloud storage cp %TWIN%\flood_risk.geojson %TWIN%\lidar_buildings.geojson %BUCKET% --project=%PROJECT_ID% --content-type=application/geo+json --cache-control="public, max-age=3600"
if errorlevel 1 goto :fail

call gcloud storage du --summarize %BUCKET% --project=%PROJECT_ID%
echo Done - test: https://storage.googleapis.com/geosense-agent-2-twin/flood_risk.geojson
exit /b 0

:fail
echo FAILED - read the message above
exit /b 1
