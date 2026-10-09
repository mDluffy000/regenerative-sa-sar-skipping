"""Generate relocated copies; do not alter archived netlists or run Spectre."""
from pathlib import Path
import argparse,re,shutil
p=argparse.ArgumentParser();p.add_argument('--pdk-model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
root=Path(__file__).resolve().parent
if not a.pdk_model.is_file():p.error('PDK model file does not exist')
out=a.output.resolve()
if out.exists():p.error('Choose a new output directory')
out.mkdir(parents=True)
for f in root.iterdir():
    if f.suffix not in ['.scs','.va']:continue
    s=f.read_text()
    def sub(m):
        source=Path(m[2]);name=source.name
        replacement=a.pdk_model.resolve() if name.endswith('.lib') else out/name
        if not name.endswith('.lib') and not (root/name).exists():raise FileNotFoundError(source)
        return m[1]+'"'+str(replacement)+'"'
    s=re.sub(r'(\b(?:include|ahdl_include)\s+)"([^"]+)"',sub,s)
    (out/f.name).write_text(s)
print(out)
