# geoai_agent.py
# Purpose: LangGraph GeoAI agent - takes a plain-English question, asks the language model
#          which MCP tools are relevant and which place the question is about, calls the
#          tools on the MCP server and turns the results into an evidence-based answer
# Version of Task 3.3 (Exercise 3):
#   - tool descriptions in the selection prompt (three prompt styles for comparison)
#   - location from the question: stated coordinates, a place name geocoded with
#     OpenStreetMap Nominatim, the previous question of the thread, or the demo point
#   - conversation memory: one compiled graph per process and a history field that
#     accumulates the question-answer pairs of a thread
#   - error notes instead of crashes when a tool cannot be reached or rejects a request
#   - a transcript (text + JSON) of every answer in outputs/agent_logs/
# Run from the project root while mcp_server.py runs in another window:
#   python src/phase4_agent/geoai_agent.py "Is 22.5530, 88.3540 suitable for a clinic?"
import json
import logging
import operator
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')
# the google-genai SDK logs a notice about automatic function calling (AFC) with the first
# request; the agent passes no functions to the model, so that one notice is filtered out
logging.getLogger('google_genai.models').addFilter(
    lambda record: 'automatic function calling' not in record.getMessage())
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
MODEL = os.getenv('AGENT_MODEL', 'google_genai:gemini-3.8-flash')
THINKING = os.getenv('AGENT_THINKING')        # Gemini 3 thinking level; unset = default
PROMPT_STYLE = os.getenv('AGENT_PROMPT', 'described')   # vague | basic | described
LOG_DIR = ROOT / 'outputs' / 'agent_logs'
DEMO_POINT = {'latitude': 13.0827, 'longitude': 80.2707}      # demo coordinate (Chennai)
DEFAULT_QUESTION = 'Is this location in Chennai suitable for a new clinic?'
TIMEOUT_S = 60           # a street-view query at a new place first downloads its photos
BOX_DEG = 0.0015         # half-size of the box sent to lidar_profile around the point
GEOCODER = 'https://nominatim.openstreetmap.org/search'
USER_AGENT = 'GeoSense-Agent-2.0 (university lab exercise)'

# what each tool measures, from which data, and where it applies
TOOLS = {
    'score_location': 'site suitability for a new facility: opportunity and risk scores '
                      '(0-10) and a GOOD/POOR verdict from the Phase I model (distance to '
                      'the nearest road and hospital); Kolkata study area only',
    'classify_imagery': 'land cover (urban, vegetation, water, bare land, agriculture), '
                        'NDVI, NDWI and vegetation change 2018-2023 from Sentinel-2 '
                        'imagery; Kolkata study area only',
    'lidar_profile': 'building heights, tree canopy cover, impervious surface and slope '
                     'from airborne LiDAR; only around Dupont Circle, Washington DC',
    'streetview_scene': 'street-level scene from Mapillary photos: shares of road, '
                        'sidewalk, buildings, vegetation, cars and people, and a '
                        'walkability score; wherever photos exist',
    'get_climate_risk': 'climate of the last 12 months against 1991-2020 (ERA5-Land): '
                        'temperature anomaly, rainfall anomaly and a drought index; '
                        'worldwide',
    'get_demographics': 'people living within 1 km and population density (WorldPop), '
                        'the nearest hospital and the hospitals within 2 km (hospitals in '
                        'the Kolkata study area only)',
    'find_similar_sites': 'other candidate sites within a radius that have a similar '
                          'suitability profile, ranked (PostGIS); Kolkata study area only',
}
TOOL_CATALOG = list(TOOLS)
DATA_NOTES = """\
- scales: opportunity_score and risk_score 0-10, walkability_score 0-10, drought_index
  0-1, scene and land-cover shares in % of pixels; always give the scale with a score
- score_location: flood_risk is a placeholder (0 for every site) and says nothing about
  flooding; the project has no flood layer yet
- a result with available = false means the tool has no data for this place: say so and
  give the reason
- an entry with an error means the tool could not be used: mention it briefly"""
PROMPTS = {
    'vague': 'Which of these tools would you use? {names}\nQuestion: {question}',
    'basic': ("Given this user question: '{question}'\n"
              "and this list of available tools: {names}\n"
              "Return a comma-separated list of the tool names that should be called."),
    'described': """\
You select the analysis tools that a geospatial assistant calls to answer a question.

Available tools:
{described}
{context}
Question: {question}

Choose every tool whose output is needed to answer the question, and no others.
Reply with JSON only, in this form:
{{"tools": ["tool_name"], "latitude": null, "longitude": null, "place": null}}
- latitude and longitude: copy them only if the question states coordinates
- place: the place name in the question (for example "Park Street, Kolkata"), else null""",
}
ANSWER_PROMPT = """\
You are GeoSense Agent, a geospatial analyst. Answer the question from the tool results.
{context}
Question: {question}
Location used: {lat}, {lon} ({source})

Tool results (JSON):
{results}

Notes on the data:
{notes}

Write at most 180 words: the direct answer first, then the evidence with numbers, units
and the tool each comes from. Say plainly what is not known or not covered. Never invent
values."""
DROP_KEYS = {'image_ids', 'enclosed_ids', 'patch_window_rc', 'bbox_projected',
             'archive_snapshot', 'inference_time_s', 'from_cache'}


class AgentState(TypedDict, total=False):
    question: str
    tool_calls: list
    tool_results: list
    final_answer: str
    location: dict                                  # latitude, longitude, source
    model_error: str                                # set when the model could not answer
    history: Annotated[list, operator.add]          # earlier turns of the thread


def make_llm():
    """
    Create the chat model named by AGENT_MODEL in .env.

    Args:
        none - AGENT_MODEL has the form 'provider:model'; LangChain reads the provider's
        key from the environment (GOOGLE_API_KEY for google_genai, ANTHROPIC_API_KEY for
        anthropic). Gemini 3 keeps its default temperature; AGENT_THINKING (minimal,
        low, medium, high) sets its thinking level when given. The client retries
        twice, ask_model() decides what happens after that.

    Returns:
        BaseChatModel: the chat model used by both language-model nodes

    Example:
        llm = make_llm()
    """
    if MODEL.startswith('google_genai:gemini-3') and THINKING:
        return init_chat_model(MODEL, thinking_level=THINKING, max_retries=2)
    if MODEL.startswith('google_genai:'):
        return init_chat_model(MODEL, max_retries=2)
    return init_chat_model(MODEL, temperature=0)


llm = make_llm()


def message_text(msg):
    """The plain text of a reply (a string, or a list of content blocks for Gemini 3)."""
    text = getattr(msg, 'text', None)
    if isinstance(text, str) and text:
        return text.strip()
    content = msg.content
    if isinstance(content, list):
        return ' '.join(b.get('text', '') for b in content if isinstance(b, dict)).strip()
    return str(content).strip()


class ModelUnavailable(Exception):
    """The language model could not answer (overload, quota, network)."""


def ask_model(prompt, attempts=2, wait_s=15):
    """
    Send one prompt to the model; retry a temporary overload once, report the rest.

    Args:
        prompt (str): the prompt
        attempts (int): how many times a 503 / per-minute 429 is tried
        wait_s (int): seconds before the next attempt

    Returns:
        str: the text of the reply

    Example:
        ask_model('Say hello in one sentence.')
    """
    for attempt in range(1, attempts + 1):
        try:
            return message_text(llm.invoke(prompt))
        except Exception as err:                         # provider errors differ in type
            text = ' '.join(str(err).split())
            daily = 'PerDay' in text                    # a daily quota will not recover
            busy = any(k in text for k in ('503', 'UNAVAILABLE', '429', 'overloaded'))
            if daily or not busy or attempt == attempts:
                code = re.search(r'\b(4\d\d|5\d\d) [A-Z][A-Za-z_]*', text)
                reason = code.group(0) if code else type(err).__name__
                if daily:
                    reason += ' - the daily free-tier quota of this model is used up'
                raise ModelUnavailable(reason) from err
            print(f'[model] {text[:80]}... - trying again in {wait_s} s')
            time.sleep(wait_s)


def wrap(text, width=88, indent=''):
    """Wrap every line of a reply separately, so paragraphs and lists keep their breaks."""
    out = []
    for line in text.splitlines() or ['']:
        out += textwrap.wrap(line, width, initial_indent=indent,
                             subsequent_indent=indent) or ['']
    return '\n'.join(out)


def parse_selection(text):
    """Tool names (and location fields) from a JSON or a comma-separated reply."""
    data = {}
    match = re.search(r'\{.*\}', text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    names = data.get('tools') if isinstance(data.get('tools'), list) else \
        re.split(r'[,\n]', text)
    chosen = []
    for name in names:
        name = str(name).strip().strip('`"\'*-. ')
        if name in TOOL_CATALOG and name not in chosen:
            chosen.append(name)
    return chosen, data


def select_tools(question, style=None, history=None):
    """
    Ask the model which tools answer the question (and where the question is about).

    Args:
        question (str): the user's question
        style (str): prompt style - 'vague', 'basic' or 'described' (default AGENT_PROMPT)
        history (list): earlier turns of the conversation, used to resolve 'its', 'there'

    Returns:
        dict: tools (list of names), latitude, longitude, place and the raw reply

    Example:
        select_tools('How tall are the buildings around 38.9097, -77.0434?')['tools']
    """
    style = style or PROMPT_STYLE
    context = ''
    if history:
        last = history[-1]
        loc = last.get('location') or {}
        context = (f"\nPrevious question in this conversation: {last['question']}"
                   f" (location {loc.get('latitude')}, {loc.get('longitude')})\n")
    described = '\n'.join(f'- {name}: {text}' for name, text in TOOLS.items())
    prompt = PROMPTS[style].format(question=question, names=TOOL_CATALOG,
                                   described=described, context=context)
    reply = ask_model(prompt)
    tools, data = parse_selection(reply)
    return {'tools': tools, 'latitude': data.get('latitude'),
            'longitude': data.get('longitude'), 'place': data.get('place'), 'raw': reply}


def stated(value, question):
    """True when the number really appears in the question (no invented coordinates)."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return False
    return any(abs(abs(float(n)) - v) < 1e-4
               for n in re.findall(r'-?\d+(?:\.\d+)?', question))


def coordinates_in(question):
    """The first 'latitude, longitude' pair written in the question (or None)."""
    for lat, lon in re.findall(r'(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)', question):
        if abs(float(lat)) <= 90 and abs(float(lon)) <= 180:
            return float(lat), float(lon)
    return None


def geocode(place):
    """Latitude and longitude of a place name from OpenStreetMap Nominatim (or None)."""
    try:
        resp = httpx.get(GEOCODER, params={'q': place, 'format': 'jsonv2', 'limit': 1},
                         headers={'User-Agent': USER_AGENT}, timeout=10)
        hits = resp.json() if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        return None
    if not hits:
        return None
    hit = hits[0]
    return {'latitude': round(float(hit['lat']), 5),
            'longitude': round(float(hit['lon']), 5),
            'source': f'"{place}" geocoded with OpenStreetMap Nominatim'}


def resolve_location(selection, question, history):
    """Coordinates of the question, else its place name, else the last turn, else demo."""
    lat, lon = selection.get('latitude'), selection.get('longitude')
    if stated(lat, question) and stated(lon, question) and abs(float(lat)) <= 90 \
            and abs(float(lon)) <= 180:
        return {'latitude': float(lat), 'longitude': float(lon),
                'source': 'coordinates stated in the question'}
    pair = coordinates_in(question)                  # if the model missed or failed
    if pair:
        return {'latitude': pair[0], 'longitude': pair[1],
                'source': 'coordinates read from the question'}
    if selection.get('place'):
        found = geocode(selection['place'])
        if found:
            return found
    for turn in reversed(history or []):
        if turn.get('location'):
            return {'latitude': turn['location']['latitude'],
                    'longitude': turn['location']['longitude'],
                    'source': 'carried over from the previous question'}
    return {**DEMO_POINT, 'source': 'demo point - no location in the question'}


def understand_query(state: AgentState) -> dict:
    """Node 1: choose the tools and the location for this question."""
    history = state.get('history') or []
    try:
        selection = select_tools(state['question'], history=history)
    except ModelUnavailable as err:
        print(f'[understand_query] model unavailable: {err}')
        selection = {'tools': [], 'model_error': str(err)}
    location = resolve_location(selection, state['question'], history)
    print(f'[understand_query] tools chosen: {", ".join(selection["tools"]) or "none"}')
    print(f'[understand_query] location: {location["latitude"]}, {location["longitude"]}'
          f' ({location["source"]})')
    return {'tool_calls': selection['tools'], 'location': location,
            'model_error': selection.get('model_error', '')}


def tool_payload(tool, location):
    """The request body of a tool: a point, or a small box around it for lidar_profile."""
    lat, lon = location['latitude'], location['longitude']
    if tool == 'lidar_profile':
        return {'min_lat': round(lat - BOX_DEG, 6), 'min_lon': round(lon - BOX_DEG, 6),
                'max_lat': round(lat + BOX_DEG, 6), 'max_lon': round(lon + BOX_DEG, 6)}
    return {'latitude': lat, 'longitude': lon}


def call_tool(client, tool, payload):
    """POST one tool; a readable error entry instead of an exception."""
    t0 = time.perf_counter()
    try:
        resp = client.post(f'{MCP_SERVER_URL}/tools/{tool}', json=payload,
                           timeout=TIMEOUT_S)
    except httpx.ConnectError:
        return {'tool': tool, 'status': 'error',
                'error': f'MCP server not reachable at {MCP_SERVER_URL} '
                         '(is mcp_server.py running?)'}
    except httpx.TimeoutException:
        return {'tool': tool, 'status': 'error', 'error': f'no answer in {TIMEOUT_S} s'}
    except httpx.HTTPError as err:
        return {'tool': tool, 'status': 'error', 'error': f'{type(err).__name__}: {err}'}
    seconds = round(time.perf_counter() - t0, 1)
    try:
        body = resp.json()
    except ValueError:
        body = {'detail': resp.text[:200]}
    if resp.status_code != 200:
        detail = body.get('detail', body) if isinstance(body, dict) else body
        detail = ' '.join(str(detail).split())[:200]          # one line, at most 200 chars
        return {'tool': tool, 'status': resp.status_code, 'seconds': seconds,
                'error': f'HTTP {resp.status_code}: {detail}'}
    return {'tool': tool, 'status': 200, 'seconds': seconds, 'result': body}


def execute_tools(state: AgentState) -> dict:
    """Node 2: call each chosen MCP tool for the location of the question."""
    results = []
    with httpx.Client() as client:
        for tool in state['tool_calls']:
            entry = call_tool(client, tool, tool_payload(tool, state['location']))
            note = entry.get('error') or ('available' if entry['result'].get(
                'available', True) else 'not available here')
            print(f'[execute_tools] {tool}: {entry["status"]} '
                  f'({entry.get("seconds", 0):.1f} s) {note}')
            results.append(entry)
    return {'tool_results': results}


def compact(value):
    """Drop long id lists and technical fields before the results go to the model."""
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in DROP_KEYS}
    if isinstance(value, list):
        return [compact(v) for v in value[:8]]
    return value


SKIP_FACTS = ('latitude', 'longitude', 'available')


def fallback_answer(state, reason):
    """An answer without the model: the reason and the facts the tools returned."""
    lines = [f'The language model is not available ({reason}), so no written answer '
             'could be produced. What the tools returned:']
    for entry in state.get('tool_results', []):
        if 'error' in entry:
            lines.append(f'- {entry["tool"]}: {entry["error"]}')
            continue
        facts = {k: v for k, v in entry['result'].items() if k not in SKIP_FACTS
                 and isinstance(v, (int, float, str, bool))}
        lines.append(f'- {entry["tool"]}: ' + ', '.join(f'{k} {v}' for k, v in
                                                          list(facts.items())[:6]))
    if not state.get('tool_results'):
        lines.append('- no tool was called')
    return '\n'.join(lines)


def synthesize_response(state: AgentState) -> dict:
    """Node 3: write the answer from the tool results and the earlier turns."""
    history = state.get('history') or []
    context = ''.join(f'\nEarlier question: {t["question"]}\nEarlier answer: '
                      f'{t["answer"][:600]}\n' for t in history[-3:])
    loc = state['location']
    prompt = ANSWER_PROMPT.format(
        context=context, question=state['question'], lat=loc['latitude'],
        lon=loc['longitude'], source=loc['source'], notes=DATA_NOTES,
        results=json.dumps(compact(state['tool_results']), indent=1))
    try:
        answer = state.get('model_error') and fallback_answer(state, state['model_error'])
        answer = answer or ask_model(prompt)
    except ModelUnavailable as err:
        print(f'[synthesize_response] model unavailable: {err}')
        answer = fallback_answer(state, str(err))
    turn = {'question': state['question'], 'answer': answer, 'location': loc,
            'tools': state['tool_calls']}
    return {'final_answer': answer, 'history': [turn]}


def build_agent_graph():
    """Three nodes in a line: understand_query -> execute_tools -> synthesize_response."""
    graph = StateGraph(AgentState)
    graph.add_node('understand_query', understand_query)
    graph.add_node('execute_tools', execute_tools)
    graph.add_node('synthesize_response', synthesize_response)
    graph.set_entry_point('understand_query')
    graph.add_edge('understand_query', 'execute_tools')
    graph.add_edge('execute_tools', 'synthesize_response')
    graph.add_edge('synthesize_response', END)
    return graph.compile(checkpointer=MemorySaver())


_GRAPH = None


def get_graph():
    """The compiled graph of this process - one MemorySaver keeps every thread's turns."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


def token_totals(usage):
    """Input, output, thinking and total tokens summed over the model calls of a run."""
    total = {k: sum(u.get(k, 0) for u in usage.values())
             for k in ('input_tokens', 'output_tokens', 'total_tokens')}
    total['thinking_tokens'] = sum((u.get('output_token_details') or {}).get(
        'reasoning', 0) for u in usage.values())
    return total


def save_transcript(run):
    """Write one answer as text and as JSON (with the full tool results) to agent_logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]       # milliseconds
    loc, tok = run['location'], run['tokens']
    tools = ', '.join(f'{r["tool"]} ({r["status"]})' for r in run['tool_results'])
    lines = [f'GeoSense Agent 2.0 - conversation log ({stamp})',
             f'Model    : {MODEL} (thinking {THINKING or "default"}, prompt '
             f'{PROMPT_STYLE})',
             f'Thread   : {run["thread_id"]}, turn {run["turn"]}',
             f'Question : {run["question"]}',
             f'Location : {loc["latitude"]}, {loc["longitude"]} - {loc["source"]}',
             f'Tools    : {tools or "none"}',
             f'Tokens   : input {tok["input_tokens"]}, output {tok["output_tokens"]} '
             f'(thinking {tok["thinking_tokens"]}), total {tok["total_tokens"]}',
             f'Time     : {run["seconds"]:.1f} s', 'Answer   :',
             wrap(run['final_answer'], 88, '  ')]
    path = LOG_DIR / f'agent_transcript_{stamp}.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with open(path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=1, default=str)
    return path


def run_agent(question, thread_id='default-session', remember=True, log=True):
    """
    Run the graph for one question and return the answer with everything behind it.

    Args:
        question (str): the user's plain English question
        thread_id (str): conversation session id - turns of one thread share a history
        remember (bool): False compiles a fresh graph (and memory) for this call only
        log (bool): write the transcript to outputs/agent_logs/

    Returns:
        dict: question, final_answer, tool_calls, tool_results, location, turn,
              tokens, seconds, thread_id and the transcript path

    Example:
        run = run_agent('What about its flood risk?', thread_id='clinic-1')
        print(run['tool_calls'], run['final_answer'])
    """
    app = get_graph() if remember else build_agent_graph()
    config = {'configurable': {'thread_id': thread_id}}
    t0 = time.perf_counter()
    with get_usage_metadata_callback() as cb:
        state = app.invoke({'question': question, 'tool_calls': [], 'tool_results': [],
                            'final_answer': ''}, config)
    run = {'question': question, 'final_answer': state['final_answer'],
           'tool_calls': state['tool_calls'], 'tool_results': state['tool_results'],
           'location': state['location'], 'turn': len(state.get('history') or []),
           'thread_id': thread_id, 'tokens': token_totals(cb.usage_metadata),
           'models': sorted(cb.usage_metadata), 'model_error': state.get('model_error'),
           'seconds': round(time.perf_counter() - t0, 1)}
    if log:
        run['transcript'] = str(save_transcript(run).relative_to(ROOT))
    return run


def ask_agent(question, thread_id='default-session'):
    """
    Ask the GeoAI Agent a plain English question and get a synthesised answer.

    Args:
        question (str): the user's plain English question
        thread_id (str): conversation session id, used for agent memory

    Returns:
        str: a readable, evidence-based answer built from MCP tool results

    Example:
        answer = ask_agent('Is this location safe from flooding?')
        print(answer)
    """
    return run_agent(question, thread_id)['final_answer']


if __name__ == '__main__':
    question = ' '.join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f'Question: {question}\n')
    run = run_agent(question)
    tok = run['tokens']
    print(f'\n{wrap(run["final_answer"])}\n')
    print(f'Tokens: input {tok["input_tokens"]}, output {tok["output_tokens"]} (thinking '
          f'{tok["thinking_tokens"]}); {run["seconds"]} s')
    print(f'Transcript: {run["transcript"]}')
