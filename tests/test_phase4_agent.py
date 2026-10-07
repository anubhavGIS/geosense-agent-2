# test_phase4_agent.py
# Purpose: Automated tests to verify Phase 4 modules work correctly
# Run with: python -m pytest tests/test_phase4_agent.py -v
# NOTE: requires the MCP server running locally on port 8000. The two agent tests of
#       Section G.1 call the language model of AGENT_MODEL (two requests each); the
#       mocked test at the end needs neither the model nor the server.
import sys

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

sys.path.append('src/phase4_agent')
import geoai_agent                                                    # noqa: E402
from geoai_agent import ask_agent                                     # noqa: E402

MCP_URL = 'http://localhost:8000'
KOLKATA = {'latitude': 22.5530, 'longitude': 88.3540}
DUPONT = {'min_lat': 38.9085, 'min_lon': -77.0450, 'max_lat': 38.9110, 'max_lon': -77.0420}


# ---- the four tests of Section G.1 ------------------------------------------------
def test_mcp_server_docs_reachable():
    resp = httpx.get(f'{MCP_URL}/docs')
    assert resp.status_code == 200, 'MCP server docs page should be reachable'


def test_score_location_tool():
    resp = httpx.post(f'{MCP_URL}/tools/score_location',
                      json={'latitude': 13.0827, 'longitude': 80.2707}, timeout=60)
    assert resp.status_code == 200
    assert 'opportunity_score' in resp.json()


def test_agent_returns_nonempty_answer():
    answer = ask_agent('Is this location suitable for a clinic?')
    assert isinstance(answer, str) and len(answer) > 0, \
        'Agent should return a non-empty answer'
    assert not answer.startswith('The language model is not available'), answer[:120]


def test_agent_selects_at_least_one_tool():
    from geoai_agent import build_agent_graph
    app = build_agent_graph()
    config = {'configurable': {'thread_id': 'test-thread'}}
    result = app.invoke({'question': 'Score this location for flood risk',
                         'tool_calls': [], 'tool_results': [], 'final_answer': ''}, config)
    assert len(result['tool_calls']) > 0, 'Agent should select at least one tool'


# ---- additional tests ----------------------------------------------------------------
@pytest.mark.parametrize('tool', ['score_location', 'classify_imagery', 'lidar_profile',
                                  'streetview_scene', 'get_climate_risk',
                                  'get_demographics', 'find_similar_sites'])
def test_every_tool_returns_200(tool):
    body = DUPONT if tool == 'lidar_profile' else KOLKATA
    resp = httpx.post(f'{MCP_URL}/tools/{tool}', json=body, timeout=120)
    assert resp.status_code == 200, f'{tool} should answer 200'
    assert resp.json()['available'] is True, f'{tool} should have data for its test place'


@pytest.mark.parametrize('body, field', [
    ({'latitude': 22.5530}, 'longitude'),                         # missing field
    ({'latitude': 'north', 'longitude': 88.354}, 'latitude'),     # text for a number
    ({'latitude': 95, 'longitude': 88.354}, 'latitude'),          # out of range
])
def test_invalid_payload_returns_422(body, field):
    resp = httpx.post(f'{MCP_URL}/tools/score_location', json=body)
    assert resp.status_code == 422
    assert resp.json()['detail'][0]['loc'][-1] == field


def test_latest_result_round_trip():
    body = {'latitude': 22.5530, 'longitude': 88.3540, 'question': 'pytest'}
    stored = httpx.post(f'{MCP_URL}/agent/latest_result', json=body).json()
    latest = httpx.get(f'{MCP_URL}/agent/latest_result').json()
    assert latest['id'] == stored['id'] and latest['question'] == 'pytest'


def test_agent_with_mocked_model(monkeypatch):
    """The whole graph without the language model or the server: both are replaced."""
    replies = iter([
        AIMessage(content='{"tools": ["get_climate_risk"], "latitude": 22.553, '
                          '"longitude": 88.354, "place": null}'),
        AIMessage(content='Mocked answer: the last 12 months were near normal.'),
    ])
    monkeypatch.setattr(geoai_agent, 'llm', GenericFakeChatModel(messages=replies))
    monkeypatch.setattr(geoai_agent, 'call_tool', lambda client, tool, payload: {
        'tool': tool, 'status': 200, 'seconds': 0.0,
        'result': {'temperature_anomaly_c': -0.1, 'drought_index': 0.0}})
    run = geoai_agent.run_agent('Has the last year been dry at 22.5530, 88.3540?',
                                thread_id='mock-thread', log=False, publish=False)
    assert run['tool_calls'] == ['get_climate_risk']
    assert run['location']['latitude'] == 22.553
    assert run['final_answer'].startswith('Mocked answer')
