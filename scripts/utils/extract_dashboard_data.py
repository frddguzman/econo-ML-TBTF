"""Extract per-period time-series from the embedded const DATA object in dashboard.html.

dashboard.html line ~607 has:  const DATA = {"<cand>_<regime>_<eta>": {...}, ...};

We just need the key for our cell (e.g., "w58_st_e01") and to load it as JSON.
"""
import os, sys, json, re

sys.stdout.reconfigure(encoding='utf-8')

DASH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')


def extract_data():
    """Return the parsed DATA dict from dashboard.html."""
    with open(DASH_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    # Find 'const DATA = ' followed by an object literal terminated by ';\n' at the same column 0
    # The simplest robust approach: find the marker, then count braces.
    marker = 'const DATA = '
    start = text.find(marker)
    if start < 0:
        raise RuntimeError('const DATA = not found')
    obj_start = start + len(marker)
    # Walk the brace stack
    depth = 0
    i = obj_start
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif c == '\\':
            escape = True
        elif c == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    obj_end = i + 1
                    break
        i += 1
    raw = text[obj_start:obj_end]
    return json.loads(raw)


def main():
    data = extract_data()
    # Print top-level keys to sanity-check we got it
    keys = sorted(data.keys())
    print(f'DATA has {len(keys)} keys', flush=True)
    # Show w58 keys
    w58_keys = [k for k in keys if k.startswith('w58')]
    print(f'  w58 keys ({len(w58_keys)}):', w58_keys, flush=True)
    # Show one cell's column names
    sample_key = 'w58_st_e01' if 'w58_st_e01' in data else w58_keys[0] if w58_keys else keys[0]
    cols = sorted(data[sample_key].keys())
    print(f'\n  {sample_key} columns ({len(cols)}):', flush=True)
    for c in cols:
        v = data[sample_key][c]
        if isinstance(v, list):
            print(f'    {c:>30} : list len={len(v)}, first 3 = {v[:3]}', flush=True)
        else:
            print(f'    {c:>30} : {type(v).__name__} = {v}', flush=True)


if __name__ == '__main__':
    main()
