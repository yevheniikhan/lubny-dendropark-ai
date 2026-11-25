"""Digitise the numbered quarter scheme of the dendropark (68 quarters) into pixel polygons.
Seeds = positions of the quarter numbers read from the scheme; watershed over the green fill with the
light boundary lines as barriers."""
import numpy as np, json
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, segmentation
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from shapely import coverage_simplify
from shapely.geometry import shape
from rasterio.features import shapes
from matplotlib.path import Path as MPath
TOL = 3.0                                                    # scheme px, about 5 m on the ground

SEEDS = {1: (135, 54), 2: (173, 54), 3: (214, 54), 4: (252, 54), 5: (293, 54), 6: (330, 54),
         7: (124, 103), 8: (175, 103), 9: (237, 102), 10: (278, 103), 11: (320, 103),
         12: (105, 147), 13: (169, 138), 14: (215, 132), 15: (260, 150), 16: (296, 150), 17: (317, 161),
         18: (329, 134), 19: (351, 114), 20: (341, 166), 21: (347, 149), 22: (362, 136), 23: (368, 161),
         24: (73, 194), 25: (111, 194), 26: (153, 194), 27: (186, 171), 28: (214, 211), 29: (258, 194),
         30: (293, 194), 31: (316, 189), 32: (340, 184), 33: (333, 216), 34: (348, 200), 35: (373, 190),
         36: (359, 233), 37: (364, 212), 38: (383, 212), 39: (386, 235),
         40: (76, 247), 41: (110, 247), 42: (145, 247), 43: (182, 247), 44: (222, 256), 45: (258, 250),
         46: (295, 250), 47: (335, 260), 48: (381, 265),
         49: (22, 312), 50: (46, 321), 51: (110, 324), 52: (196, 293), 53: (165, 329), 54: (208, 329),
         55: (278, 298), 56: (386, 317),
         57: (281, 366), 58: (338, 367), 59: (281, 383), 60: (338, 384), 61: (281, 397), 62: (338, 399),
         63: (281, 411), 64: (338, 413), 65: (281, 425), 66: (338, 427), 67: (281, 437), 68: (347, 451)}

# quarter 55 is cut into sectors by alleys radiating from the southern circle; each sector gets a seed
EXTRA = [(55, (247, 300)), (55, (250, 335)), (55, (268, 345)), (55, (290, 345)), (55, (312, 343)),
         (55, (328, 330)), (55, (320, 295)), (49, (45, 258)), (49, (12, 345))]


def digitise():
    a = np.array(Image.open('data/lltk/q/quarters_src.png').convert('RGB')).astype(float)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    green = (G > 55) & (G < 140) & (R < 50) & (B < 60)
    lab, n = ndi.label(ndi.binary_dilation(green, iterations=2))
    big = lab == (np.argmax(ndi.sum(green, lab, range(1, n + 1))) + 1)
    region = ndi.binary_erosion(ndi.binary_fill_holes(big), iterations=1)
    mk = np.zeros(R.shape, int)
    for q, (x, y) in list(SEEDS.items()) + EXTRA:
        mk[y - 1:y + 2, x - 1:x + 2] = q
    elev = np.where(B > 140, 0, R)                         # white digits are not boundaries
    elev = ndi.gaussian_filter(elev, 0.6)                  # boundary lines are bright in red
    L = segmentation.watershed(elev, mk, mask=region)
    # hairline boundaries the watershed misses: reassign pixels by the side of the line read off the scheme
    yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]
    pts = np.c_[xx.ravel() + .5, yy.ravel() + .5]
    def west_of(poly):
        return MPath(poly).contains_points(pts).reshape(L.shape)
    W = west_of([(61, 190), (60, 215), (59, 242), (42, 280), (5, 357), (-5, 390), (-60, 390), (-60, 190)])
    sel = np.isin(L, (40, 49, 50))
    L[sel & W] = 49
    L[np.isin(L, (49, 50)) & ~W] = 50
    for a_, b_, cut in [(65, 67, [(230, 430), (252, 431), (311, 434.4), (330, 435.5)]),
                        (66, 68, [(300, 435.3), (313, 435), (362, 433.4), (380, 433)])]:
        S = west_of(cut + [(cut[-1][0], 600), (cut[0][0], 600)])          # below the line
        pair = np.isin(L, (a_, b_))
        L[pair & S] = b_
        L[pair & ~S] = a_
    # boundary lines and circles go to the nearest quarter, so neighbouring quarters share one edge
    L = segmentation.expand_labels(L, distance=6)
    L[~region] = 0
    polys = {}
    for geom, v in shapes(L.astype(np.int32), mask=L > 0, connectivity=4):
        q = int(v)
        g = shape(geom)
        polys[q] = g if q not in polys else unary_union([polys[q], g])
    one = lambda g: g if g.geom_type == 'Polygon' else max(g.geoms, key=lambda h: h.area)
    keys = sorted(polys)
    cov = [Polygon(one(polys[q]).exterior) for q in keys]
    simp = coverage_simplify(cov, tolerance=TOL)            # shared edges stay shared
    P = {q: g for q, g in zip(keys, simp)}
    outer = unary_union(list(P.values())).buffer(0.01).buffer(-0.01)
    outer = Polygon(one(outer).exterior)
    return L, P, outer


if __name__ == '__main__':
    L, P, outer = digitise()
    np.save('/tmp/qL.npy', L)
    json.dump({str(q): [list(map(lambda v: round(v, 2), xy)) for xy in (p.exterior.coords if p.geom_type == 'Polygon'
               else max(p.geoms, key=lambda g: g.area).exterior.coords)] for q, p in P.items()},
              open('data/lltk/q/quarters_px.json', 'w'))
    print({q: round(p.area) for q, p in P.items()})
    print('outer', round(outer.area))
