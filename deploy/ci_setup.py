"""
ci_setup.py - Phase 5, Exercise 4 (Task E.4.2): the Google Cloud identity of the GitHub
Actions pipeline and its key - created, handed to GitHub without being shown, removed.

  account     create the service account github-actions-deployer (when it is missing) and
              give it the three roles a deploy needs, each on the narrowest resource:
                roles/run.developer           on the project: deploy new revisions of a
                                              Cloud Run service (but not change its IAM)
                roles/iam.serviceAccountUser  on geosense-backend only: let those revisions
                                              run as the backend's own identity
                roles/artifactregistry.writer on geosense-repo only: push (and pull) images
              then create one JSON key in %USERPROFILE%\\.config\\geosense - outside the
              project folder, so it can never be committed
  clip WHAT   copy a value to the Windows clipboard for a GitHub secret without printing it:
              key (the key file, for GCP_SA_KEY), project (for GCP_PROJECT_ID), cesium (the
              CESIUM_ION_TOKEN of .env, for CESIUM_ION_TOKEN) or clear (empty the clipboard)
  delete-key  delete the local key file once GitHub holds it; the key itself stays valid in
              Google Cloud (its ID is listed), and GitHub's secret is its only copy
  status      the deployer's roles and keys (key IDs and dates only)

Nothing secret is printed: the key and the token are read from their files and written to
the clipboard; the IAM commands' output (which lists the project's members) is not shown.

Run from the project root in the geosense environment:
    python deploy\\ci_setup.py account
    python deploy\\ci_setup.py clip key          (paste into the secret GCP_SA_KEY)
    python deploy\\ci_setup.py clip clear
    python deploy\\ci_setup.py delete-key
    python deploy\\ci_setup.py status
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'geosense-agent-2'
REGION = 'us-central1'
REPO = 'geosense-repo'
DEPLOYER = f'github-actions-deployer@{PROJECT}.iam.gserviceaccount.com'
RUNTIME_SA = f'geosense-backend@{PROJECT}.iam.gserviceaccount.com'
MEMBER = f'serviceAccount:{DEPLOYER}'
KEY_FILE = Path.home() / '.config' / 'geosense' / 'github-deployer-key.json'
ROLES = [  # (role, where, the gcloud command that grants it on that resource)
    ('roles/run.developer', f'project {PROJECT}',
     f'projects add-iam-policy-binding {PROJECT} --condition=None'),
    ('roles/iam.serviceAccountUser', f'service account {RUNTIME_SA.split("@")[0]}',
     f'iam service-accounts add-iam-policy-binding {RUNTIME_SA}'),
    ('roles/artifactregistry.writer', f'repository {REPO}',
     f'artifacts repositories add-iam-policy-binding {REPO} --location={REGION}'),
]


def decode(raw):
    return raw.decode('utf-8', 'replace').strip()


def gcloud(args, check=True):
    """Run one gcloud command for the project; its output is returned, not printed."""
    p = subprocess.run(f'gcloud {args} --project={PROJECT} --quiet', shell=True,
                       capture_output=True, stdin=subprocess.DEVNULL)
    if check and p.returncode != 0:
        sys.exit(f'FAILED: gcloud {args.split(" --")[0]}\n{decode(p.stderr)[-1200:]}')
    return p.returncode, decode(p.stdout), decode(p.stderr)


def ok(text):
    print(f'[OK  ] {text}')


def check_account():
    _, account, _ = gcloud('config get-value account', check=False)
    if '@' not in account:
        sys.exit('gcloud has no signed-in account - run: gcloud auth login')
    ok(f'gcloud account {account[:2]}***{account[account.find("@"):]}; '
       f'every command below uses --project={PROJECT}')


def grant(role, where, command):
    """Add one role binding; a service account created seconds ago may not be visible to
    IAM yet, so 'does not exist' is retried (6 x 10 s)."""
    args = f'{command} --member={MEMBER} --role={role} --format=none'
    for attempt in range(1, 7):
        rc, _, err = gcloud(args, check=False)
        if rc == 0:
            ok(f'{role:30s} on {where}')
            return
        if attempt == 6 or 'does not exist' not in err:
            sys.exit(f'FAILED: {role} on {where}\n{err[-1200:]}')
        print('       the new service account is not visible to IAM yet - retry in 10 s')
        time.sleep(10)


def user_keys():
    """(key id, created, expires) of the deployer's user-managed keys."""
    rc, out, _ = gcloud(f'iam service-accounts keys list --iam-account={DEPLOYER} '
                        f'--managed-by=user --format=json', check=False)
    keys = json.loads(out) if rc == 0 and out.startswith('[') else []
    return [(k['name'].split('/')[-1], k.get('validAfterTime', '')[:10],
             k.get('validBeforeTime', '')[:10]) for k in keys]


def account():
    check_account()
    rc, _, _ = gcloud(f'iam service-accounts describe {DEPLOYER}', check=False)
    if rc == 0:
        ok(f'service account {DEPLOYER} exists')
    else:
        gcloud(f'iam service-accounts create {DEPLOYER.split("@")[0]} '
               f'--display-name="GitHub Actions Deployer"')
        ok(f'service account {DEPLOYER} created - the identity of the pipeline')
    for role, where, command in ROLES:
        grant(role, where, command)
    if KEY_FILE.exists():
        ok(f'key file {KEY_FILE} already exists - no new key created')
    else:
        existing = user_keys()
        if existing:
            print(f'       note: the account already has {len(existing)} key(s): '
                  + ', '.join(k[0][:8] for k in existing) + ' (python deploy\\ci_setup.py status)')
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        gcloud(f'iam service-accounts keys create "{KEY_FILE}" --iam-account={DEPLOYER}')
        key_id = json.loads(KEY_FILE.read_text(encoding='utf-8'))['private_key_id']
        ok(f'key {key_id[:8]}... written to {KEY_FILE} ({KEY_FILE.stat().st_size:,} bytes) - '
           f'outside the project, never committed')
    print('next: GitHub -> Settings -> Secrets and variables -> Actions -> New repository secret:')
    print('      GCP_PROJECT_ID   python deploy\\ci_setup.py clip project, paste')
    print('      GCP_SA_KEY       python deploy\\ci_setup.py clip key, paste')
    print('      CESIUM_ION_TOKEN python deploy\\ci_setup.py clip cesium, paste')
    print('      then: python deploy\\ci_setup.py clip clear  and  python deploy\\ci_setup.py delete-key')


def to_clipboard(text):
    if sys.platform != 'win32':
        sys.exit('the clipboard copy uses Windows clip.exe - run this on the laptop')
    subprocess.run('clip', input=text.encode('utf-8'), shell=True, check=True)


def clip(what):
    if what == 'key':
        if not KEY_FILE.exists():
            sys.exit(f'{KEY_FILE} not found - run: python deploy\\ci_setup.py account')
        text = KEY_FILE.read_text(encoding='utf-8')
        info = json.loads(text)
        if info.get('type') != 'service_account' or info.get('client_email') != DEPLOYER:
            sys.exit(f'{KEY_FILE} is not a key of {DEPLOYER}')
        target = 'GCP_SA_KEY'
    elif what == 'project':
        text, target = PROJECT, 'GCP_PROJECT_ID'
    elif what == 'cesium':
        text = (dotenv_values(ROOT / '.env').get('CESIUM_ION_TOKEN') or '').strip()
        if not text:
            sys.exit('CESIUM_ION_TOKEN missing from .env')
        target = 'CESIUM_ION_TOKEN'
    else:                                                        # clear
        to_clipboard('')
        ok('clipboard emptied')
        return
    to_clipboard(text)
    shown = PROJECT if what == 'project' else f'{len(text):,} characters, not shown'
    ok(f'copied for the secret {target} ({shown}) - paste it into the Value box, Add secret')


def delete_key():
    if KEY_FILE.exists():
        KEY_FILE.unlink()
        ok(f'{KEY_FILE} deleted - GitHub\'s secret GCP_SA_KEY is now the only copy of the key')
    else:
        ok(f'{KEY_FILE} is not on this laptop')
    left = [p.name for p in KEY_FILE.parent.glob('github-deployer-key*.json')]
    ok('no deployer key file left in ' + str(KEY_FILE.parent) if not left
       else f'still there: {", ".join(left)}')
    print_keys()


def print_keys():
    keys = user_keys()
    print(f'keys of {DEPLOYER.split("@")[0]} in Google Cloud: {len(keys)}')
    for key_id, made, expires in keys:
        never = not expires or expires.startswith('9999')       # Google's "no expiry"
        print(f'  {key_id[:8]}...  created {made}  expires {"never" if never else expires}')


def roles_on(command_and_resource):
    rc, out, _ = gcloud(f'{command_and_resource} --flatten="bindings[].members" '
                        f'--filter="bindings.members:{MEMBER}" --format="value(bindings.role)"',
                        check=False)
    return out.split() if rc == 0 else ['(not readable)']


def status():
    rc, _, _ = gcloud(f'iam service-accounts describe {DEPLOYER}', check=False)
    if rc != 0:
        print(f'{DEPLOYER} does not exist - run: python deploy\\ci_setup.py account')
        return
    print(f'service account {DEPLOYER}')
    print(f'  project {PROJECT}: ' + ', '.join(roles_on(f'projects get-iam-policy {PROJECT}')))
    print(f'  service account {RUNTIME_SA.split("@")[0]}: ' + ', '.join(
        roles_on(f'iam service-accounts get-iam-policy {RUNTIME_SA}')))
    print(f'  repository {REPO}: ' + ', '.join(
        roles_on(f'artifacts repositories get-iam-policy {REPO} --location={REGION}')))
    print_keys()
    print(f'local key file {KEY_FILE}: ' + ('PRESENT - delete it once GitHub has it'
                                           if KEY_FILE.exists() else 'not present'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='step', required=True)
    sub.add_parser('account')
    c = sub.add_parser('clip')
    c.add_argument('what', choices=['key', 'project', 'cesium', 'clear'])
    sub.add_parser('delete-key')
    sub.add_parser('status')
    args = ap.parse_args()
    {'account': account, 'delete-key': delete_key, 'status': status}.get(
        args.step, lambda: clip(args.what))()


if __name__ == '__main__':
    main()
