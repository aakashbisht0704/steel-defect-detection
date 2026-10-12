import os
import json

dirs = [
    "app/types",
    "app/lib",
    "app/routes",
    "app/components/layout",
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

with open('files.json', 'r', encoding='utf-8') as f:
    files = json.load(f)

for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
