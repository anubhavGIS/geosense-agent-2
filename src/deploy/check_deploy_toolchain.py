"""
check_deploy_toolchain.py - Phase 5, Exercise 1: verify the deployment toolchain.

Checks, in order, everything Exercise 1 installs or prepares and writes a report to
outputs/deploy_logs/toolchain_check_<stamp>.txt:
  1. Docker Desktop   - client and server versions, the WSL 2 backend, hello-world
  2. Google Cloud SDK - gcloud version, active account and project, billing state,
                        the three APIs of Phase 5, the Artifact Registry repository
  3. Other tools      - Python, Node.js, Streamlit, git
  4. Workspace        - the Phase 5 folders, the .gitignore block, the portfolio files
  5. Secrets          - the key NAMES the cloud deployment will need, read from .env;
                        values are never printed, only "set (n chars)"

Run from the project root in the geosense environment:
    python src\\deploy\\check_deploy_toolchain.py            (full check)
    python src\\deploy\\check_deploy_toolchain.py --no-hello (skip docker run hello-world)
Every line is PASS, WARN or FAIL; the exit code is 1 when any check fails.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]          # GeoSense_Agent
LOG_DIR = ROOT / 'outputs' / 'deploy_logs'
REGION = 'us-central1'
REPO = 'geosense-repo'
APIS = ['run.googleapis.com', 'artifactregistry.googleapis.com',
        'cloudbuild.googleapis.com']
FOLDERS = ['deploy', '.github/workflows', 'docs/portfolio/screenshots',
           'outputs/deploy_logs']
GITIGNORE_LINES = ['deploy/key.json', '*.tfstate', '.streamlit/secrets.toml',
                   'outputs/deploy_logs/']
PORTFOLIO_FILES = ['docs/portfolio/cloud_secrets_checklist.md',
                   'docs/portfolio/demo_questions.md']
# the runtime secrets of the deployed system (server side unless noted) - names only
SECRETS = ['GOOGLE_API_KEY', 'AGENT_MODEL', 'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER',
           'DB_PASSWORD', 'MAPILLARY_TOKEN', 'GEE_PROJECT', 'CESIUM_ION_TOKEN']

lines = []            # the report


def decode(raw):
    """Windows 'wsl' prints UTF-16; everything else UTF-8."""
    return raw.decode('utf-16') if b'\x00' in raw[:200] else raw.decode('utf-8', 'replace')


def run(cmd, timeout=180, stdout_only=False):
    """Run a shell command; return (ok, text). stdout_only=True drops stderr so that a
    JSON answer is not mixed with gcloud's progress lines ('Listing items under ...')."""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f'timed out after {timeout} s'
    raw = p.stdout if stdout_only else p.stdout + p.stderr
    return p.returncode == 0, decode(raw).strip()


def report(status, name, detail=''):
    line = f'[{status:4}] {name:<34} {detail}'.rstrip()
    print(line)
    lines.append(line)
    return status


def section(title):
    print(f'\n{title}')
    lines.extend(['', title, '-' * len(title)])


def check_docker(hello):
    section('1. Docker Desktop')
    if not shutil.which('docker'):
        report('FAIL', 'docker client', 'docker not on PATH - install Docker Desktop')
        return
    ok, out = run('docker --version')
    report('PASS' if ok else 'FAIL', 'docker client', out.splitlines()[0] if out else '')
    ok, out = run('docker info --format "{{.ServerVersion}} | {{.OSType}} | {{.Name}}"')
    if ok:
        report('PASS', 'docker daemon', f'server {out}')
    else:
        report('FAIL', 'docker daemon', 'not running - start Docker Desktop and wait')
    ok, out = run('wsl --status')
    version = re.search(r'(?:Default Version|version)\s*:?\s*(\d)', out)
    if ok and version:
        report('PASS' if version.group(1) == '2' else 'WARN', 'WSL backend',
               f'default version {version.group(1)}')
    else:
        report('WARN', 'WSL backend', 'wsl --status gave no version line')
    if hello:
        ok, out = run('docker run --rm hello-world', timeout=300)
        report('PASS' if ok and 'Hello from Docker' in out else 'FAIL', 'hello-world',
               'image pulled and container ran' if ok
               else ('daemon not reachable - Docker Desktop not running' if 'pipe' in out
                     else out[-160:]))


def gcloud_json(args):
    ok, out = run(f'gcloud {args} --format=json', stdout_only=True)
    if not ok:
        return None, run(f'gcloud {args} --format=json')[1]      # the error text
    try:
        return json.loads(out or 'null'), ''
    except json.JSONDecodeError:
        return None, out


def check_gcloud():
    section('2. Google Cloud SDK and project')
    if not shutil.which('gcloud'):
        report('FAIL', 'gcloud', 'gcloud not on PATH - install the Google Cloud SDK')
        return None
    ok, out = run('gcloud --version')
    first = out.splitlines()[0] if out else ''
    report('PASS' if ok else 'FAIL', 'gcloud version', first)
    cfg, err = gcloud_json('config list')
    account = (cfg or {}).get('core', {}).get('account')
    project = (cfg or {}).get('core', {}).get('project')
    report('PASS' if account else 'FAIL', 'active account',
           mask_email(account) if account else 'none - run gcloud init')
    report('PASS' if project else 'FAIL', 'active project', project or 'none')
    if not project:
        return None
    bill, err = gcloud_json(f'billing projects describe {project}')
    if bill is not None:
        enabled = bill.get('billingEnabled', False)
        report('PASS' if enabled else 'FAIL', 'billing',
               'enabled' if enabled else 'NOT enabled')
    else:
        report('WARN', 'billing', 'could not query (' + err.splitlines()[0][:70] + ')')
    svc, err = gcloud_json('services list --enabled')
    names = {s.get('config', {}).get('name') for s in (svc or [])}
    for api in APIS:
        report('PASS' if api in names else 'FAIL', f'API {api.split(".")[0]}',
               'enabled' if api in names else 'not enabled')
    repos, err = gcloud_json(f'artifacts repositories list --location={REGION}')
    found = [r for r in (repos or []) if r.get('name', '').endswith('/' + REPO)]
    if found:
        report('PASS', 'Artifact Registry repo',
               f'{REPO} ({found[0].get("format", "?")}, {REGION})')
    else:
        report('FAIL', 'Artifact Registry repo', f'{REPO} not found in {REGION}')
    return project


def mask_email(email):
    user, _, domain = email.partition('@')
    return f'{user[:2]}***@{domain}'


def check_tools():
    section('3. Other tools')
    report('PASS', 'Python', sys.version.split()[0] + f'  ({sys.executable})')
    for name, cmd in [('Node.js', 'node --version'), ('npm', 'npm --version'),
                      ('git', 'git --version'), ('Streamlit', 'streamlit version')]:
        ok, out = run(cmd)
        report('PASS' if ok else 'WARN', name, out.splitlines()[0] if out else 'not found')


def check_workspace():
    section('4. Deployment workspace')
    for folder in FOLDERS:
        exists = (ROOT / folder).is_dir()
        report('PASS' if exists else 'FAIL', f'folder {folder}',
               '' if exists else 'missing')
    gi = ROOT / '.gitignore'
    text = gi.read_text(encoding='utf-8') if gi.exists() else ''
    missing = [l for l in GITIGNORE_LINES if l not in text.splitlines()]
    report('PASS' if not missing else 'FAIL', '.gitignore Phase 5 block',
           'all four lines present' if not missing else 'missing: ' + ', '.join(missing))
    for f in PORTFOLIO_FILES:
        p = ROOT / f
        report('PASS' if p.exists() else 'FAIL', f, f'{p.stat().st_size} bytes'
               if p.exists() else 'missing')


def check_secrets():
    section('5. Secrets checklist (.env key names - values never printed)')
    env = dotenv_values(ROOT / '.env')
    for key in SECRETS:
        value = env.get(key)
        if value:
            report('PASS', key, f'set ({len(value)} chars)')
        else:
            report('FAIL' if key != 'AGENT_MODEL' else 'WARN', key, 'missing')
    extra = sorted(k for k in env if k not in SECRETS)
    report('PASS', 'other .env keys', f'{len(extra)} more (not needed in the cloud)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-hello', action='store_true', help='skip docker run hello-world')
    args = ap.parse_args()
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'GeoSense Agent 2.0 - Phase 5 toolchain check  ({stamp})\nproject root: {ROOT}')
    lines.append(f'Phase 5 toolchain check {stamp}  root={ROOT}')
    check_docker(hello=not args.no_hello)
    check_gcloud()
    check_tools()
    check_workspace()
    check_secrets()
    status = [l[1:5].strip() for l in lines if l.startswith('[')]
    summary = (f'{status.count("PASS")} passed, {status.count("WARN")} warnings, '
               f'{status.count("FAIL")} failed')
    print(f'\nSUMMARY: {summary}')
    lines.extend(['', 'SUMMARY: ' + summary])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f'toolchain_check_{stamp}.txt'
    log.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'report: {log}')
    sys.exit(1 if 'FAIL' in status else 0)


if __name__ == '__main__':
    main()
