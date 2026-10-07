# memory_test.py
# Purpose: test the agent's short-term memory - a question about a location, then a
#          follow-up that only makes sense with the first turn ('its'). The pair is asked
#          twice in one thread_id:
#            without memory  a fresh graph (and MemorySaver) for every call, as in the
#                            first version of ask_agent()
#            with memory     one graph for the process and the history field of the state
# Run from the project root while mcp_server.py runs in another window:
#   python src/phase4_agent/memory_test.py
import time
from datetime import datetime

from geoai_agent import run_agent, wrap

QUESTIONS = ['Is this location (22.5530, 88.3540) suitable for a clinic?',
             'What about its flood risk?']


def main(pause=10):
    """
    Ask the two questions without and with memory and print what each turn used.

    Args:
        pause (float): seconds between the two runs of a turn pair (free-tier limits)

    Returns:
        None: prints the location, tools and answer of every turn; the transcripts are
              saved in outputs/agent_logs/

    Example:
        main()
    """
    stamp = datetime.now().strftime('%H%M%S')
    for remember in (False, True):
        mode = 'WITH memory' if remember else 'WITHOUT memory'
        thread = f'memory-test-{"on" if remember else "off"}-{stamp}'
        print(f'{"=" * 30} {mode} (thread {thread}) {"=" * 30}')
        for question in QUESTIONS:
            run = run_agent(question, thread_id=thread, remember=remember)
            loc = run['location']
            print(f'\nTurn {run["turn"]}: {question}')
            print(f'  location : {loc["latitude"]}, {loc["longitude"]} ({loc["source"]})')
            print(f'  tools    : {", ".join(run["tool_calls"]) or "none"}')
            print(wrap(run['final_answer'], 96, '  | '))
            print(f'  transcript {run["transcript"]}')
            time.sleep(pause)
        print()


if __name__ == '__main__':
    main()
