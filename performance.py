import numpy as np
from collections import defaultdict

# Giả sử bạn có list of dicts: records = [{'yaw':..., 'box':(l,t,w,h), 'gt':(...), 'conf':...}, ...]

# 1) Define bins
bins = {
  'frontal':   lambda r: abs(r['yaw'])<15 and abs(r['pitch'])<10,
  'moderate':  lambda r: 15<=abs(r['yaw'])<30 or 10<=abs(r['pitch'])<20,
  'extreme':   lambda r: abs(r['yaw'])>=30 or abs(r['pitch'])>=20
}

# 2) Group records
groups = defaultdict(list)
for r in records:
    for name, cond in bins.items():
        if cond(r):
            groups[name].append(r)
            break

# 3) Compute metrics per group
def iou(boxA, boxB):
    # compute intersection area and union area...
    return interArea / float(areaA + areaB - interArea)

results = {}
for name, recs in groups.items():
    det_rate = sum(1 for r in recs if r['box'] is not None)/len(recs)
    if recs and 'gt' in recs[0]:
        ious = [iou(r['box'], r['gt']) for r in recs if r['box'] and r['gt']]
        mean_iou = np.mean(ious)
    else:
        mean_iou = None
    confs = [r['conf'] for r in recs if r['box'] and 'conf' in r]
    mean_conf = np.mean(confs) if confs else None

    results[name] = {
        'det_rate': det_rate,
        'mean_iou': mean_iou,
        'mean_conf': mean_conf
    }

print(results)
