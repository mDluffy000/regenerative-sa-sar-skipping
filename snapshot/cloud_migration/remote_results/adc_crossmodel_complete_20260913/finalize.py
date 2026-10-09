"""Finalize this one running experiment; never start additional training."""
from pathlib import Path
import json,time,csv,tarfile,hashlib
D=Path(__file__).resolve().parent
while not (D/'FINISHED.json').exists():time.sleep(15)
finished=json.loads((D/'FINISHED.json').read_text())
if any(finished['exit_codes'].values()):
 (D/'CLOSEOUT_BLOCKED.json').write_text(json.dumps(finished,indent=2)+'\n');raise SystemExit('Some jobs failed; preserve all artifacts, no retry or extra training')
rows=[];lines=['# 新 SA 跨模型复跑收尾报告','', '本报告的主表均为各模型固定的1000张历史测试图；不混用500张验证精度。三种模型均只覆盖原先指定的三个CIM层，不声称全网络模拟或全系统净节能。','']
for kind in ['resnet','deit','bn_refresh_finetune']:
 r=json.loads((D/kind/'results.json').read_text());assert r['status']=='complete'
 lines+=['## '+kind,'','| 测试路线 | 准确率 | SAR比较减少 | ADC转换数 |','|---|---:|---:|---:|']
 for name,v in r['test'].items():
  h=v.get('hardware',{});saving=h.get('comparison_saving');count=h.get('conversions')
  if h:assert h['gated']==0 and h['conversions']==h['requests'] and h['whole_ADC_conversion_skip_fraction']==0
  rows.append(dict(model=kind,route=name,n=v['n'],accuracy=v['accuracy'],SAR_comparison_saving=saving,ADC_conversions=count,SAR_comparisons=h.get('comparisons'),SA_near=h.get('near'),whole_conversion_skip=0 if h else None))
  text=f'{saving*100:.3f}%' if saving is not None else '—'
  lines.append(f"| {name} | {100*v['accuracy']:.2f}% | {text} | {count if count is not None else '—'} |")
 lines+=['','所有验证选择先冻结，再读取测试集。未恢复的操作点也完整报告。','']
lines+=['## BN校准补救的解释边界','','ResNet原矩阵保持原来的冻结BN流程，结果不被补救路线替换。另列的BN补救：同一W4A4起点，每路线仅用2000张训练图校准BN一次；再完成既定5轮微调，使用500张验证集选模，所有三个路线一起冻结后测试。SA残差表、阈值、ADC映射和零门控策略未更改。','', '旧ResNet FP32完整10000张测试准确率68.83%、W4A4为67.15%；这些数字不可直接与本轮500张验证精度或1000张测试精度相减。原始外部ResNet预训练可能已用过本轮验证图片，不能宣称完全独立的验证集。','', 'SA为B提供的55nm原理图数据的确定性双线性插值，包含网格外边界夹取、985ps单端读出和滚动参考的建模假设；不是芯片实测系统结果。SAR比较减少不是整次ADC跳过，也不是全系统净能耗下降。','', 'CNN与Transformer的层覆盖、激活编码和网络基线不同。本实验支持逐模型的误差/恢复结果，不能单独证明某一架构普遍更耐受。','', '本次仅完成暂停前已声明的预算，没有按测试结果扩展训练或挑选更有利的操作点。']
(D/'CROSSMODEL_CLOSEOUT.md').write_text('\n'.join(lines)+'\n')
with (D/'RESULTS_COMPARISON.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
files=[p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.log' and p.name!='DELIVERY_SHA256.json']
hashes={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(D/'DELIVERY_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
archive=D.parent/'crossmodel_complete_20260913.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
 for p in files+[D/'DELIVERY_SHA256.json']:tf.add(p,arcname=str(p.relative_to(D)))
(D/'DELIVERY_READY.json').write_text(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),time=time.time()),indent=2)+'\n')
