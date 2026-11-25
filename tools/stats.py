"""Park statistics inside data/park_boundary.json -> data/park_stats.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, grids as G
H, N, Z = G.load()
m = ~np.isnan(H)
tall = H[m] > 5
st = {'area_ha': json.load(open('data/park_boundary.json'))['area_ha'],
      'n_quarters': 68,
      'tree_cover_gt5_share': round(float(tall.mean()), 3),
      'median_tree_h': round(float(np.median(H[m][tall])), 1),
      'p90_tree_h': round(float(np.percentile(H[m][tall], 90)), 1),
      'max_h': round(float(np.nanmax(H)), 1),
      'ndvi': {m_: round(float(np.nanmean(a)), 3) for m_, a in N.items() if np.isfinite(a).any()}}
json.dump(st, open('data/park_stats.json', 'w'), ensure_ascii=False)
print(st)
