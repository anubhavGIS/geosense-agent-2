# evaluate_tool_selection.py
# Purpose: measure how well the agent chooses its tools - six clear test questions, the
#          tools each one needs, and the tools the model selects with three prompt styles:
#            vague      a bare question without format or tool descriptions
#            basic      the blueprint prompt: tool names only, comma-separated reply
#            described  every tool with what it measures and where it applies, JSON reply
#          Only the selection step runs (one model call per question and style); no MCP
#          tool is called, so the MCP server does not need to be running.
# Run from the project root:
#   python src/phase4_agent/evaluate_tool_selection.py
#   python src/phase4_agent/evaluate_tool_selection.py --styles basic described
import argparse
import csv
import json
import time
from datetime import datetime

from langchain_core.callbacks import get_usage_metadata_callback

from geoai_agent import LOG_DIR, MODEL, ROOT, select_tools, token_totals

QUESTIONS = [  # question, tools it needs
    ('What is the land cover around 22.5726, 88.3639, and has the vegetation changed '
     'since 2018?', {'classify_imagery'}),
    ('How tall are the buildings and how much tree canopy is there around '
     '38.9097, -77.0434?', {'lidar_profile'}),
    ('Is Park Street in Kolkata pleasant to walk along?', {'streetview_scene'}),
    ('How many people live near 22.5530, 88.3540, and how far is the nearest hospital?',
     {'get_demographics'}),
    ('Has the past year been unusually hot or dry in Chennai?', {'get_climate_risk'}),
    ('Is 22.5450, 88.3600 a good site for a new clinic, and which nearby sites are '
     'similar?', {'score_location', 'find_similar_sites'}),
]


def verdict(expected, selected):
    """MATCH (same set), EXTRA (all needed + more), MISSED (a needed tool absent)."""
    if set(selected) == expected:
        return 'MATCH'
    return 'EXTRA' if expected <= set(selected) else 'MISSED'


def main(styles, pause):
    """
    Run every question with every prompt style and print the comparison table.

    Args:
        styles (list): prompt styles to compare, e.g. ['vague', 'basic', 'described']
        pause (float): seconds between model calls (free-tier requests per minute)

    Returns:
        dict: number of exact matches per style

    Example:
        main(['basic', 'described'], 13)
    """
    rows, matches, tokens = [], {s: 0 for s in styles}, {}
    print(f'Model: {MODEL}\n')
    for n, (question, expected) in enumerate(QUESTIONS, 1):
        print(f'Q{n}  {question}')
        print(f'    {"expected":<10}: {", ".join(sorted(expected))}')
        for style in styles:
            with get_usage_metadata_callback() as cb:
                selected = select_tools(question, style=style)['tools']
            tok = token_totals(cb.usage_metadata)
            tokens[style] = tokens.get(style, 0) + tok['total_tokens']
            result = verdict(expected, selected)
            matches[style] += result == 'MATCH'
            print(f'    {style:<10}: {", ".join(selected) or "-":<52} {result}')
            rows.append({'question_no': n, 'question': question, 'style': style,
                         'expected': ' + '.join(sorted(expected)),
                         'selected': ' + '.join(selected), 'result': result,
                         'tokens': tok['total_tokens']})
            time.sleep(pause)
        print()
    for style in styles:
        print(f'{style:<10}: {matches[style]} of {len(QUESTIONS)} questions matched '
              f'exactly ({tokens[style]:,} tokens)')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = LOG_DIR / f'tool_selection_{stamp}'
    with open(out.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps(
        {'model': MODEL, 'matches': matches, 'tokens': tokens, 'rows': rows}, indent=1))
    print(f'\nSaved {out.relative_to(ROOT)}.csv and .json')
    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare tool selection by prompt style.')
    parser.add_argument('--styles', nargs='+', default=['vague', 'basic', 'described'],
                        choices=['vague', 'basic', 'described'])
    parser.add_argument('--pause', type=float, default=13.0,
                        help='seconds between model calls (default 13)')
    args = parser.parse_args()
    main(args.styles, args.pause)
