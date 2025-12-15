import sys

with open(sys.argv[1], encoding='utf-8-sig') as f:
    content = f.read()
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.write(content)