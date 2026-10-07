# app.py
# Purpose: Streamlit demo app - a question box, the agent's answer with the tools it used,
#          and the 3D digital twin embedded below it (the twin flies to the place the
#          agent discussed, because the agent posts each answer's location to the server)
# Start the three services in this order, each in its own window, from the project root:
#   python src/phase4_agent/mcp_server.py
#   cd frontend/digital_twin  then  npm run dev
#   streamlit run src/phase4_agent/app.py
import uuid

import streamlit as st

from geoai_agent import MODEL, run_agent

TWIN_URL = 'http://localhost:5173'
DEFAULT_QUESTION = 'Is Park Street in Kolkata (22.5530, 88.3540) a good place for a new ' \
                   'clinic, and is it safe from flooding?'


def new_thread():
    """Start a new conversation: a fresh thread_id gives the agent an empty memory."""
    st.session_state.thread_id = f'streamlit-{uuid.uuid4().hex[:8]}'
    st.session_state.turns = []


def tool_rows(run):
    """One row per tool call: status, time and what came back."""
    rows = []
    for entry in run['tool_results']:
        result = entry.get('result') or {}
        note = entry.get('error') or ('result' if result.get('available', True)
                                      else result.get('reason', 'not available'))
        rows.append({'tool': entry['tool'], 'status': str(entry['status']),
                     'seconds': entry.get('seconds'), 'outcome': note})
    return rows


st.set_page_config(page_title='GeoSense Agent 2.0', layout='wide')
st.title('GeoSense Agent 2.0 — Conversational Geospatial Intelligence')
if 'thread_id' not in st.session_state:
    new_thread()

question = st.text_input('Ask a question about any location:', DEFAULT_QUESTION)
ask_col, new_col, info_col = st.columns([1.2, 1.2, 5])
ask = ask_col.button('Ask GeoSense Agent', type='primary')
new_col.button('New conversation', on_click=new_thread)
info_col.caption(f'Model {MODEL} · thread {st.session_state.thread_id} · '
                 f'{len(st.session_state.turns)} earlier question(s) in this thread')

if ask and question.strip():
    with st.spinner('Agent is reasoning and calling tools...'):
        st.session_state.turns.append(run_agent(question,
                                                thread_id=st.session_state.thread_id))

if st.session_state.turns:
    run = st.session_state.turns[-1]
    st.success(run['final_answer'])
    loc, tok = run['location'], run['tokens']
    sent = 'yes' if run['published'] else 'no'
    st.caption(f'Location {loc["latitude"]}, {loc["longitude"]} ({loc["source"]}) · '
               f'turn {run["turn"]} · {tok["total_tokens"]:,} tokens · '
               f'{run["seconds"]} s · sent to the twin: {sent}')
    st.markdown('**Tools the agent used**')
    if run['tool_results']:
        st.dataframe(tool_rows(run), hide_index=True, width='stretch')
    else:
        st.write('No tool was called for this question.')
    if len(st.session_state.turns) > 1:
        with st.expander('Earlier questions in this conversation'):
            for earlier in st.session_state.turns[-2::-1]:
                st.markdown(f'**{earlier["question"]}**')
                st.write(earlier['final_answer'])

st.subheader('3D Digital Twin')
st.iframe(TWIN_URL, height=600)      # Streamlit 1.65: st.iframe replaces components.iframe
