# test_llm_connection.py
# Purpose: confirm live connectivity to the agent's language model before any agent code is
# written. The GeoAI agent is model-agnostic: AGENT_MODEL in .env names the provider and
# the model in LangChain's "provider:model" form, e.g.
#     google_genai:gemini-3.8-flash      (Google AI Studio key  -> GOOGLE_API_KEY)
#     anthropic:claude-sonnet-4-6        (Anthropic Console key -> ANTHROPIC_API_KEY)
#     groq:llama-3.3-70b-versatile       (GroqCloud key         -> GROQ_API_KEY)
# The script
#   1. loads .env from the project root and picks the key that belongs to the provider
#   2. lists the models the key can use and checks that AGENT_MODEL is one of them
#   3. sends "Say hello in one sentence." through LangChain's init_chat_model
#   4. prints the reply with its token usage and saves a transcript in outputs/agent_logs/
#
# Run from the project root:   python src/phase4_agent/test_llm_connection.py
# Options:
#   --bad-key   alter the last characters of the loaded key to reproduce the provider's
#               authentication error (troubleshooting practice) without touching .env
#   --no-log    do not write the transcript file
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# the google-genai SDK prints a one-time advisory about automatic function calling on every
# plain generate_content call; it does not apply to LangChain's use of the SDK
logging.getLogger('google_genai.models').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]            # ...\GeoSense_Agent
ENV_FILE = ROOT / '.env'
LOG_DIR = ROOT / 'outputs' / 'agent_logs'
PROMPT = 'Say hello in one sentence.'
DEFAULT_MODEL = 'google_genai:gemini-3.8-flash'
KEY_VARS = {                                          # provider -> environment variable
    'google_genai': 'GOOGLE_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'groq': 'GROQ_API_KEY',
    'openai': 'OPENAI_API_KEY',
}


def mask(secret):
    """Show only the prefix and the length of a secret so it can appear in a lab record."""
    if not secret:
        return '<missing>'
    return f'{secret[:6]}... ({len(secret)} characters)'


def split_spec(spec):
    """'google_genai:gemini-3.8-flash' -> ('google_genai', 'gemini-3.8-flash')."""
    if ':' in spec:
        provider, model = spec.split(':', 1)
        return provider.strip(), model.strip()
    return 'anthropic', spec.strip()                  # bare Claude ids keep working


def list_models(provider, key):
    """Return the model ids this key can use (empty if the provider has no listing)."""
    if provider == 'google_genai':
        from google import genai
        client = genai.Client(api_key=key)
        return [m.name.replace('models/', '') for m in client.models.list()
                if 'generateContent' in (m.supported_actions or [])]
    if provider == 'anthropic':
        import anthropic
        return [m.id for m in anthropic.Anthropic(api_key=key).models.list(limit=50)]
    if provider == 'groq':
        from groq import Groq
        return [m.id for m in Groq(api_key=key).models.list().data]
    return []


def message_text(msg):
    """Return the plain text of a LangChain AIMessage (string or content-block list)."""
    text = getattr(msg, 'text', None)             # a property in langchain-core 1.x
    if not isinstance(text, str) and callable(text):
        text = text()                             # older releases: a method
    if not isinstance(text, str) or not text:
        content = msg.content
        if isinstance(content, list):
            text = ' '.join(b.get('text', '') for b in content if isinstance(b, dict))
        else:
            text = str(content)
    return text.strip()


def hint(error_text):
    """A one-line fix suggestion chosen from the wording of the provider's error."""
    e = error_text.lower()
    if 'api key' in e or 'x-api-key' in e or '401' in e or 'authentication' in e:
        return ('the key in .env is missing, mistyped or revoked - create a fresh key, '
                'update .env and start a NEW Python process (.env is read only once, '
                'at startup)')
    if '429' in e or 'quota' in e or 'resource_exhausted' in e or 'rate limit' in e:
        return 'free-tier rate limit reached - wait a minute, or switch AGENT_MODEL'
    if 'not found' in e or '404' in e or 'not_found' in e:
        return 'AGENT_MODEL is not a model this key can use - choose one listed above'
    if 'credit' in e or 'billing' in e:
        return 'billing - add credits in the provider console'
    if 'getaddrinfo' in e or 'connect' in e or 'timed out' in e:
        return 'network - check the internet connection and DNS, then run again'
    return 'read the last line of the error and search it together with the provider name'


def write_log(spec, used_model, text, usage, seconds):
    """Save a small transcript (never the key) under outputs/agent_logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = LOG_DIR / f'connectivity_test_{stamp}.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'GeoSense Agent 2.0 - LLM connectivity test  ({stamp})\n')
        f.write(f'AGENT_MODEL       : {spec}\n')
        f.write(f'model that replied: {used_model}\n')
        f.write(f'prompt            : {PROMPT}\n')
        f.write(f'reply             : {text}\n')
        thinking = (usage.get('output_token_details') or {}).get('reasoning')
        f.write(f'tokens            : input {usage.get("input_tokens")}, '
                f'output {usage.get("output_tokens")} (thinking {thinking}), '
                f'total {usage.get("total_tokens")}\n')
        f.write(f'round trip        : {seconds:.1f} s\n')
    return path


def main():
    load_dotenv(ENV_FILE)
    spec = os.getenv('AGENT_MODEL', DEFAULT_MODEL)
    provider, model = split_spec(spec)
    key_var = KEY_VARS.get(provider)
    key = os.getenv(key_var, '') if key_var else ''
    if '--bad-key' in sys.argv and key:
        key = key[:-4] + 'XXXX'                        # last characters changed on purpose
        os.environ[key_var] = key                     # LangChain sees the altered key
        print('[troubleshooting practice] the last 4 characters of the key were altered')
    print(f'.env file : {ENV_FILE}')
    print(f'Provider  : {provider}   ({key_var or "no key needed"})')
    print(f'API key   : {mask(key) if key_var else "-"}')
    print(f'Model     : {model}')
    if key_var and (not key or key.startswith('paste-')):
        print(f'{key_var} is missing - add it to .env and run again.')
        return 1

    from langchain.chat_models import init_chat_model

    try:
        # 1. which models can this key use?  (listing costs no tokens)
        ids = list_models(provider, key)
        if ids:
            status = 'available' if model in ids else 'NOT in the list - pick one below'
            print(f'\nModels visible to this key: {len(ids)}  ->  {model} is {status}')
            for mid in ids[:10]:
                print(f'    {mid}')

        # 2. the real chat call through LangChain (same call the agent will make)
        llm = init_chat_model(spec, temperature=0)
        t0 = time.time()
        reply = llm.invoke(PROMPT)
        seconds = time.time() - t0
        text = message_text(reply)
        usage = reply.usage_metadata or {}
        meta = reply.response_metadata or {}
        used_model = meta.get('model') or meta.get('model_name') or model

        print(f'\nPrompt : {PROMPT}')
        print(f'Reply  : {text}')
        thinking = (usage.get('output_token_details') or {}).get('reasoning')
        think_note = f' (of which {thinking} thinking)' if thinking else ''
        print(f'\nTokens : input {usage.get("input_tokens")}, '
              f'output {usage.get("output_tokens")}{think_note}, '
              f'total {usage.get("total_tokens")}   ({seconds:.1f} s, model {used_model})')
        if '--no-log' not in sys.argv:
            print(f'Log    : {write_log(spec, used_model, text, usage, seconds)}')
        print('\nLLM connectivity OK')
        return 0

    except Exception as e:                            # report any provider error
        print(f'\n{type(e).__name__}: {e}')
        print(f'Fix: {hint(str(e))}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
