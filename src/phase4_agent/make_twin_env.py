# make_twin_env.py
# Purpose: write frontend/digital_twin/.env with VITE_CESIUM_ION_TOKEN, copied from the
#          CESIUM_ION_TOKEN of the project's .env - the token is never printed. Vite gives
#          the browser only variables that start with VITE_.
#          --break writes a deliberately altered token (troubleshooting practice); run the
#          script again without it to restore the real token.
# Run from the project root:   python src/phase4_agent/make_twin_env.py [--break]
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
TWIN_ENV = ROOT / 'frontend' / 'digital_twin' / '.env'

token = dotenv_values(ROOT / '.env').get('CESIUM_ION_TOKEN') or ''
if not token:
    sys.exit('CESIUM_ION_TOKEN is missing from .env - add it first (Exercise 1).')
if '--break' in sys.argv:
    token = token[:-6] + 'XXXXXX'                  # last characters changed on purpose
TWIN_ENV.write_text(f'VITE_CESIUM_ION_TOKEN={token}\n', encoding='utf-8')
state = 'ALTERED (troubleshooting)' if '--break' in sys.argv else 'copied from .env'
print(f'Wrote {TWIN_ENV.relative_to(ROOT)}: VITE_CESIUM_ION_TOKEN = {len(token)} '
      f'characters, {state}. Restart "npm run dev" so that Vite reads it.')
