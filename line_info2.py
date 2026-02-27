from pathlib import Path
path = Path('backtester3/__init__.py')
for idx, line in enumerate(path.read_text().splitlines(), start=1):
    if 'min_support_resistance_gap' in line:
        print(idx, line.strip())
    if line.strip().startswith('def _attempt_entry'):
        print('_attempt_entry', idx)
    if line.strip().startswith('def _has_close_opposite'):
        print('_has_close_opposite', idx)
