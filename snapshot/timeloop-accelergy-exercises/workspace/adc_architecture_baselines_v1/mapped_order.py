"""Derive N/P temporal traversal directly from the executed Timeloop mapping."""
from pathlib import Path
import re,itertools
ROOT=Path(__file__).resolve().parent

def mapped_windows(architecture,prefix,N,P):
 folder=ROOT/('isaac_w4a4/cost_compat1_'+prefix if architecture=='isaac' else 'raella_w4a4/cost_native_'+prefix+'_spec1')
 text=(folder/'timeloop-mapper.map.txt').read_text();before=text.split('for X in')[0]
 loops=[(d,int(n)) for d,n in re.findall(r'for ([NP]) in \[0:(\d+)\)',before)]
 assert loops,folder
 extents={'N':1,'P':1}
 for d,n in loops:extents[d]*=n
 assert extents=={'N':N,'P':P},(folder,loops,extents,N,P)
 for indices in itertools.product(*(range(n) for d,n in loops)):
  value={'N':0,'P':0}
  for (d,radix),v in zip(loops,indices):value[d]=value[d]*radix+v
  yield value['N'],value['P']
