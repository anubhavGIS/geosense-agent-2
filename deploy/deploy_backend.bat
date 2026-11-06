@echo off
rem deploy_backend.bat - Phase 5, Exercise 3 (Tasks E.3.1 and E.3.2): build the backend image,
rem   push it to Artifact Registry and deploy it to Cloud Run as the service geosense-backend.
rem   Run from the project root, after "python deploy\cloud_setup.py prepare" has stored the
rem   secrets and written deploy\cloudrun.env.yaml:
rem       deploy\deploy_backend.bat           (tag v1)
rem       deploy\deploy_backend.bat v2        (a new tag = a new image = a new revision)
rem   No secret is written here: the environment comes from cloudrun.env.yaml, the password,
rem   the Mapillary token and the Earth Engine key from Secret Manager (--set-secrets).
rem   gcloud is a batch file itself, so every gcloud line starts with "call".
setlocal
set PROJECT_ID=geosense-agent-2
set REGION=us-central1
set REPO=geosense-repo
set SERVICE=geosense-backend
set RUNTIME_SA=geosense-backend@%PROJECT_ID%.iam.gserviceaccount.com
set TAG=%~1
if "%TAG%"=="" set TAG=v1
set IMAGE=%REGION%-docker.pkg.dev/%PROJECT_ID%/%REPO%/backend:%TAG%
set SECRETS=DB_PASSWORD=db-password:latest,MAPILLARY_TOKEN=mapillary-token:latest,/secrets/gee/key.json=gee-sa-key:latest

if not exist deploy\Dockerfile.backend (
  echo Run this from the project root ^(the folder that contains deploy\^).
  exit /b 1
)
if not exist deploy\cloudrun.env.yaml (
  echo deploy\cloudrun.env.yaml is missing - run: python deploy\cloud_setup.py prepare
  exit /b 1
)

echo [1/4] Docker credentials for %REGION%-docker.pkg.dev
call gcloud auth configure-docker %REGION%-docker.pkg.dev --quiet
if errorlevel 1 goto :fail

echo [2/4] docker build %IMAGE%
docker build -f deploy\Dockerfile.backend -t %IMAGE% .
if errorlevel 1 goto :fail

echo [3/4] docker push %IMAGE%   (started %time%)
rem a slow upload link can break a push half-way; the layers already uploaded stay in the
rem registry, so pushing again only sends the rest - up to three attempts
set TRY=1
:retry_push
docker push %IMAGE%
if not errorlevel 1 goto :pushed
if %TRY% geq 3 goto :fail
set /a TRY+=1
echo       the push was interrupted - attempt %TRY% of 3 in 15 s (uploaded layers are kept)
timeout /t 15 /nobreak >nul
goto :retry_push
:pushed
echo       push finished %time%

echo [4/4] gcloud run deploy %SERVICE%
call gcloud run deploy %SERVICE% --project=%PROJECT_ID% --region=%REGION% ^
  --image=%IMAGE% --platform=managed --allow-unauthenticated ^
  --service-account=%RUNTIME_SA% ^
  --memory=2Gi --cpu=2 --min-instances=0 --max-instances=3 --timeout=300 ^
  --env-vars-file=deploy\cloudrun.env.yaml ^
  "--set-secrets=%SECRETS%"
if errorlevel 1 goto :fail

call gcloud run services describe %SERVICE% --project=%PROJECT_ID% --region=%REGION% ^
  --format="value(status.url)"
echo Deployed %IMAGE% - open the URL above + /docs
exit /b 0

:fail
echo FAILED - read the message above. Cloud Run was not changed by this run: an existing
echo          service keeps serving its current revision.
exit /b 1
