"""Park boundary = outline of the brochure map scheme warped with a thin-plate spline through data/lltk/gcp.json."""
import numpy as np, json
from PIL import Image
from skimage import measure
from scipy.interpolate import RBFInterpolator
from shapely.geometry import Polygon
def tps():
    g = json.load(open('data/lltk/gcp.json'))
    src = np.array([a[0] for a in g], float); dst = np.array([a[1] for a in g])
    return RBFInterpolator(src, dst, kernel='thin_plate_spline'), RBFInterpolator(dst, src, kernel='thin_plate_spline')
if __name__ == '__main__':
    fwd, inv = tps()
    m = np.array(Image.open('data/lltk/scheme_mask.png')) > 0
    c = max(measure.find_contours(m.astype(float), 0.5), key=len)
    P = Polygon(fwd(np.c_[c[:, 1], c[:, 0]])).buffer(0).buffer(-14, join_style=1).buffer(14, join_style=1).simplify(3)
    xy = np.array(P.exterior.coords)[:-1]
    for _ in range(3):
        q = np.roll(xy, -1, 0); xy = np.vstack([np.c_[.75 * xy + .25 * q], np.c_[.25 * xy + .75 * q]]).reshape(2, -1, 2).transpose(1, 0, 2).reshape(-1, 2)
    P = Polygon(xy)
    json.dump({'utm': list(map(list, P.exterior.coords)), 'area_ha': round(P.area / 1e4, 2),
               'method': 'brochure map scheme warped by thin-plate spline through data/lltk/gcp.json'}, open('data/park_boundary.json', 'w'))
    print(P.area / 1e4, P.bounds)
