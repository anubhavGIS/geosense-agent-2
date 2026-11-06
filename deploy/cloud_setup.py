"""
cloud_setup.py - Phase 5, Exercise 3: the Google Cloud side of the deployment, from .env.

  prepare   enable the APIs, create the backend's own service account, store the three
            secrets in Secret Manager (cloud database password, Mapillary token, Earth
            Engine key) and let that service account - and only it - read them, then
            write deploy/cloudrun.env.yaml
  envfile   write deploy/cloudrun.env.yaml only (--cors adds the public frontends;
            MCP_ALLOWED_HOSTS = the service's run.app host names, read from gcloud)
  twin-env  write frontend/digital_twin/.env.production (live backend URL + bucket URL)
  status    the service URL and revisions, the names of its variables and secrets, the
            secrets in Secret Manager and the twin bucket

Secret values reach gcloud on standard input and are never printed - only their names,
versions and lengths. deploy/cloudrun.env.yaml holds no secret (the password, the token and
the key come from Secret Manager at run time), so it can be read and shown.

Run from the project root in the geosense environment:
    python deploy\\cloud_setup.py prepare
    python deploy\\cloud_setup.py envfile --cors https://anubhavgis.github.io
    python deploy\\cloud_setup.py twin-env
    python deploy\\cloud_setup.py status
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'geosense-agent-2'
REGION = 'us-central1'
SERVICE = 'geosense-backend'
RUNTIME_SA = f'geosense-backend@{PROJECT}.iam.gserviceaccount.com'
BUCKET = 'geosense-agent-2-twin'
KEY_FILE = Path.home() / '.config' / 'geosense' / 'gee-sa-key.json'
KEY_MOUNT = '/secrets/gee/key.json'            # where Cloud Run mounts the key file
ENV_FILE = ROOT / 'deploy' / 'cloudrun.env.yaml'
TWIN_ENV = ROOT / 'frontend' / 'digital_twin' / '.env.production'
LOCAL_ORIGINS = ['http://localhost:5173', 'http://localhost:8501',   # twin dev, Streamlit
                 'http://localhost:4173']                           # the Pages build, previewed
APIS = ['secretmanager.googleapis.com', 'iam.googleapis.com', 'storage.googleapis.com']


def decode(raw):
    return raw.decode('utf-8', 'replace').strip()


def gcloud(args, stdin=None, check=True):
    """Run one gcloud command for the project; stdin carries a secret value, never argv."""
    cmd = f'gcloud {args} --project={PROJECT} --quiet'
    if stdin is None:            # nothing to send: gcloud must never wait for the console
        p = subprocess.run(cmd, shell=True, capture_output=True, stdin=subprocess.DEVNULL)
    else:
        p = subprocess.run(cmd, shell=True, capture_output=True, input=stdin.encode('utf-8'))
    if check and p.returncode != 0:
        sys.exit(f'FAILED: gcloud {args.split(" --")[0]}\n{decode(p.stderr)[-1200:]}')
    return p.returncode, decode(p.stdout), decode(p.stderr)


def ok(text):
    print(f'[OK  ] {text}')


def load_env():
    env = {**dotenv_values(ROOT / '.env'), **os.environ}
    for key in ('CLOUD_DATABASE_URL', 'GEE_PROJECT', 'MAPILLARY_TOKEN'):
        if not env.get(key):
            sys.exit(f'{key} missing from .env')
    return env


def cloud_db(url):
    """host, port, dbname, user, password and sslmode of the Neon connection string."""
    u = urlparse(url.strip())
    if u.scheme not in ('postgres', 'postgresql') or not u.hostname or not u.password:
        sys.exit('CLOUD_DATABASE_URL must look like postgresql://user:password@host/dbname')
    query = {k: v[-1] for k, v in parse_qs(u.query).items()}
    return {'host': u.hostname, 'port': str(u.port or 5432),
            'dbname': unquote(u.path.lstrip('/')) or 'postgres',
            'user': unquote(u.username), 'password': unquote(u.password),
            'sslmode': query.get('sslmode', 'require')}


def check_account():
    _, account, _ = gcloud('config get-value account', check=False)
    if '@' not in account:
        sys.exit('gcloud has no signed-in account - run: gcloud auth login')
    shown = account[:2] + '***' + account[account.find('@'):]
    ok(f'gcloud account {shown}; every command below uses --project={PROJECT}')


def enable_apis():
    gcloud('services enable ' + ' '.join(APIS))
    ok('APIs enabled: ' + ', '.join(a.split('.')[0] for a in APIS))


def ensure_runtime_sa():
    rc, _, _ = gcloud(f'iam service-accounts describe {RUNTIME_SA}', check=False)
    if rc == 0:
        ok(f'service account {RUNTIME_SA} exists')
        return
    gcloud(f'iam service-accounts create {RUNTIME_SA.split("@")[0]} '
           f'--display-name="GeoSense backend (Cloud Run runtime)"')
    ok(f'service account {RUNTIME_SA} created - the identity of the Cloud Run service')


def put_secret(name, value, what):
    """Create the secret or add a version when the value changed; grant the runtime SA."""
    rc, _, _ = gcloud(f'secrets describe {name} --format="value(name)"', check=False)
    if rc != 0:
        gcloud(f'secrets create {name} --replication-policy=automatic --data-file=- '
               f'--labels=app=geosense', stdin=value)
        state = 'created, version 1'
    else:
        rc, current, _ = gcloud(f'secrets versions access latest --secret={name}', check=False)
        if rc == 0 and current == value.strip():
            state = 'unchanged'
        else:
            gcloud(f'secrets versions add {name} --data-file=-', stdin=value)
            state = 'new version added'
    grant = (f'secrets add-iam-policy-binding {name} --member=serviceAccount:{RUNTIME_SA} '
             f'--role=roles/secretmanager.secretAccessor --condition=None')
    for attempt in range(1, 7):        # a service account created seconds ago may not be
        rc, _, err = gcloud(grant, check=False)          # visible to IAM yet: wait, retry
        if rc == 0:
            break
        if attempt == 6 or 'does not exist' not in err:
            sys.exit(f'FAILED: gcloud secrets add-iam-policy-binding {name}\n{err[-1200:]}')
        print('       the new service account is not visible to IAM yet - retry in 10 s')
        time.sleep(10)
    ok(f'secret {name:16s} {what}, {len(value.strip())} characters - {state}; '
       f'readable by {RUNTIME_SA.split("@")[0]}')


def read_key(path):
    """The Earth Engine service-account key: checked, never printed."""
    if not path.exists():
        sys.exit(f'Earth Engine key not found: {path} - create it first (hands-on step D)')
    text = path.read_text(encoding='utf-8')
    info = json.loads(text)
    if info.get('type') != 'service_account' or 'private_key' not in info:
        sys.exit(f'{path} is not a service-account key file')
    ok(f'Earth Engine key {path} - service account {info["client_email"]}')
    return text


def write_envfile(env, cors):
    """deploy/cloudrun.env.yaml: every variable of the service except the secrets."""
    db = cloud_db(env['CLOUD_DATABASE_URL'])
    origins = LOCAL_ORIGINS + [o.rstrip('/') for o in cors if o.rstrip('/') not in LOCAL_ORIGINS]
    values = {
        'DB_HOST': db['host'], 'DB_PORT': db['port'], 'DB_NAME': db['dbname'],
        'DB_USER': db['user'], 'DB_SSLMODE': db['sslmode'],
        'STUDY_AREA': env.get('STUDY_AREA') or 'Kolkata',
        'BBOX_MIN_LON': env.get('BBOX_MIN_LON') or '88.30',
        'BBOX_MIN_LAT': env.get('BBOX_MIN_LAT') or '22.45',
        'BBOX_MAX_LON': env.get('BBOX_MAX_LON') or '88.42',
        'BBOX_MAX_LAT': env.get('BBOX_MAX_LAT') or '22.67',
        'GEE_PROJECT': env['GEE_PROJECT'],
        'GEE_SERVICE_ACCOUNT_KEY': KEY_MOUNT,
        'CORS_ORIGINS': ','.join(origins),
        'MCP_ALLOWED_HOSTS': ','.join(service_hosts()),
    }
    text = ['# cloudrun.env.yaml - written by deploy/cloud_setup.py: the environment of the Cloud',
            '# Run service geosense-backend (gcloud run deploy --env-vars-file). No secret here:',
            '# DB_PASSWORD, MAPILLARY_TOKEN and the Earth Engine key file come from Secret Manager.']
    text += [f'{key}: "{value}"' for key, value in values.items()]
    ENV_FILE.write_text('\n'.join(text) + '\n', encoding='utf-8')
    ok(f'{ENV_FILE.relative_to(ROOT)} written - {len(values)} variables, CORS_ORIGINS '
       f'{values["CORS_ORIGINS"]}')
    ok(f'MCP_ALLOWED_HOSTS {values["MCP_ALLOWED_HOSTS"] or "(none)"}')


def pick_url(svc):
    """The service's address: Cloud Run lists two - https://SERVICE-PROJECTNUMBER.REGION.run.app
    (the one gcloud run deploy prints as Service URL) and the older ...-uc.a.run.app of
    status.url; both reach the same service, the first is used everywhere in this project."""
    urls = json.loads(svc.get('metadata', {}).get('annotations', {})
                      .get('run.googleapis.com/urls', '[]'))
    for url in urls:
        if url.endswith(f'.{REGION}.run.app'):
            return url
    url = svc.get('status', {}).get('url') or ''
    return url if url.startswith('https://') else None


def describe_service():
    rc, raw, _ = gcloud(f'run services describe {SERVICE} --region={REGION} --format=json',
                        check=False)
    return json.loads(raw) if rc == 0 and raw.startswith('{') else None


def service_url():
    svc = describe_service()
    return pick_url(svc) if svc else None


def service_hosts():
    """The host names /mcp must accept (MCP_ALLOWED_HOSTS): both addresses of the service
    when it exists, else the one it will get, SERVICE-PROJECTNUMBER.REGION.run.app."""
    svc = describe_service()
    if svc:
        urls = json.loads(svc.get('metadata', {}).get('annotations', {})
                          .get('run.googleapis.com/urls', '[]'))
        urls = urls or [svc.get('status', {}).get('url') or '']
        hosts = [urlparse(u).hostname for u in urls if u.startswith('https://')]
        if hosts:
            return hosts
    _, number, _ = gcloud(f'projects describe {PROJECT} --format="value(projectNumber)"',
                          check=False)
    return [f'{SERVICE}-{number}.{REGION}.run.app'] if number.isdigit() else []


def write_twin_env(bucket):
    url = service_url()
    if not url:
        sys.exit(f'service {SERVICE} not found in {REGION} - deploy the backend first')
    base = f'https://storage.googleapis.com/{bucket}/'
    TWIN_ENV.write_text(
        '# .env.production - written by deploy/cloud_setup.py twin-env: the addresses the\n'
        '# GitHub Pages build of the twin uses (npm run build:pages). Not secret; the Cesium\n'
        '# ion token stays in .env.\n'
        f'VITE_MCP_SERVER_URL={url}\n'
        f'VITE_ASSET_BASE_URL={base}\n', encoding='utf-8')
    ok(f'{TWIN_ENV.relative_to(ROOT)} written: backend {url}, assets {base}')


def status():
    svc = describe_service()
    url = pick_url(svc) if svc else None
    if not url:
        print(f'service {SERVICE} not deployed yet in {REGION}')
    else:
        container = svc['spec']['template']['spec']['containers'][0]
        names = [e['name'] for e in container.get('env', [])]
        plain = [n for e, n in zip(container.get('env', []), names) if 'valueFrom' not in e]
        from_secrets = [n for e, n in zip(container.get('env', []), names) if 'valueFrom' in e]
        mounts = [m['mountPath'] for m in container.get('volumeMounts', [])]
        limits = container.get('resources', {}).get('limits', {})
        print(f'service {SERVICE}: {url}')
        print(f'  image {container["image"]}')
        print(f'  limits cpu {limits.get("cpu")}, memory {limits.get("memory")}; latest ready '
              f'revision {svc["status"].get("latestReadyRevisionName")}')
        print(f'  variables: {", ".join(plain)}')
        print(f'  from Secret Manager: {", ".join(from_secrets)}; files: {", ".join(mounts)}')
        _, rev, _ = gcloud(f'run revisions list --service={SERVICE} --region={REGION} '
                           f'--format="table(metadata.name,status.conditions[0].status:label=READY,'
                           f'metadata.creationTimestamp:label=CREATED)"', check=False)
        print('\n'.join('  ' + line for line in rev.splitlines()))
    _, secrets, _ = gcloud('secrets list --format="table(name.basename(),createTime.date())"',
                           check=False)
    print('secrets:\n' + '\n'.join('  ' + line for line in secrets.splitlines()))
    rc, du, _ = gcloud(f'storage du -s gs://{BUCKET}', check=False)
    print(f'bucket gs://{BUCKET}: ' + (f'{int(du.split()[0]) / 1e6:.1f} MB' if rc == 0 and du
                                       else 'not created yet'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='step', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--key', type=Path, default=KEY_FILE, help='Earth Engine key file')
    p.add_argument('--cors', nargs='*', default=[], help='extra browser origins')
    e = sub.add_parser('envfile')
    e.add_argument('--cors', nargs='*', default=[], help='extra browser origins')
    t = sub.add_parser('twin-env')
    t.add_argument('--bucket', default=BUCKET)
    sub.add_parser('status')
    args = ap.parse_args()

    if args.step == 'status':
        status()
        return
    if args.step == 'twin-env':
        write_twin_env(args.bucket)
        return
    env = load_env()
    if args.step == 'envfile':
        write_envfile(env, args.cors)
        return
    check_account()
    enable_apis()
    ensure_runtime_sa()
    key_text = read_key(args.key)
    db = cloud_db(env['CLOUD_DATABASE_URL'])
    put_secret('db-password', db['password'], 'cloud database password')
    put_secret('mapillary-token', env['MAPILLARY_TOKEN'], 'Mapillary access token')
    put_secret('gee-sa-key', key_text, 'Earth Engine key (JSON)')
    write_envfile(env, args.cors)
    print('next: deploy\\deploy_backend.bat v1   (build, push, deploy)')


if __name__ == '__main__':
    main()
