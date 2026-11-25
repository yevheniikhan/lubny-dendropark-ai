"""Park boundary: outline of the brochure map scheme, simplified to straight segments, mapped onto the image by an
affine transform fitted to ground control points (data/lltk/gcp.json), then widened by 8 m to take in the edge trees."""
import numpy as np, json
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon
BUF = 8.0
def affine():
    g = json.load(open('data/lltk/gcp.json'))
    src = np.array([a[0] for a in g], float); dst = np.array([a[1] for a in g], float)
    w = np.array([a[2] if len(a) > 2 else 1 for a in g], float)
    A = np.c_[src, np.ones(len(src))]
    P, *_ = np.linalg.lstsq(A * w[:, None], dst * w[:, None], rcond=None)
    return P, A @ P - dst
def fwd(xy, P=None):
    P = affine()[0] if P is None else P
    xy = np.atleast_2d(xy); return np.c_[xy, np.ones(len(xy))] @ P
def outline_px():
    # outline of the park traced on the brochure scheme (px of the 490 x 560 crop), straight segments
    return np.array(json.load(open('data/lltk/scheme_outline_px.json')), float)
if __name__ == '__main__':
    P, res = affine()
    print('residuals, m:', np.round(np.hypot(*res.T), 1), 'RMSE', round(float(np.sqrt((res ** 2).sum(1).mean())), 1))
    poly = Polygon(fwd(outline_px(), P)).buffer(0).buffer(BUF, join_style=2, mitre_limit=3)
    poly = poly.simplify(0.5)
    json.dump({'utm': list(map(list, poly.exterior.coords)), 'area_ha': round(poly.area / 1e4, 2),
               'method': 'brochure map scheme outline, straight segments, affine transform from data/lltk/gcp.json, +8 m'},
              open('data/park_boundary.json', 'w'))
    print(len(poly.exterior.coords), 'vertices', round(poly.area / 1e4, 2), 'ha')
