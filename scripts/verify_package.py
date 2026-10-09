from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parents[1]
records=json.loads((root/'SHA256SUMS.json').read_text())
for name,h in records.items():
    p=root/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,name
files=list(root.rglob('*.py'))
for p in files:ast.parse(p.read_text(),filename=str(p))
print(f'OK: {len(records)} file hashes, {len(files)} Python sources parsed')
