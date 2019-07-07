from pathlib import Path

p = Path('.')

files = list(p.glob('*.md'))

for f in files:
    print(f)