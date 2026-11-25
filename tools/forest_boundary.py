"""Park boundary = forest-covered land of the park: canopy height > 3 m (Meta & WRI CHM, 5 m grid) inside the georeferenced
brochure scheme (+30 m), cleaned morphologically and drawn with straight segments."""
import numpy as np, json
from scipy import ndimage as ndi
from skimage import measure
from shapely.geometry import Polygon
X0, Y1, CELL, TH = 498880, 5537360, 5.0, 3.0
def forest_polygon():
    H = np.load('data/park3.npz')['H5'] / 2
    m = ndi.gaussian_filter(np.nan_to_num(H, nan=0), 1.0) > TH
    m = ndi.binary_opening(m, iterations=2); m = ndi.binary_closing(m, iterations=3); m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m); m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
    c = max(measure.find_contours(np.pad(m, 1).astype(float), 0.5), key=len) - 1
    xy = np.c_[X0 + (c[:, 1] + .5) * CELL, Y1 - (c[:, 0] + .5) * CELL]
    P = Polygon(xy).buffer(0)
    P = P.buffer(10, join_style=1).buffer(-10, join_style=1)     # bridge notches, smooth 5 m steps
    P = P.buffer(-6, join_style=1).buffer(6, join_style=1)       # drop thin spurs
    P = P.simplify(7.0, preserve_topology=True)                   # straight segments
    P = P.buffer(4, join_style=2, mitre_limit=2.5)                # take in the edge crowns
    return P
if __name__ == '__main__':
    P = forest_polygon()
    json.dump({'utm': list(map(list, P.exterior.coords)), 'area_ha': round(P.area / 1e4, 2),
               'method': 'forest-covered land: canopy height > 3 m inside the georeferenced brochure scheme, straight segments'},
              open('data/park_boundary.json', 'w'))
    print(len(P.exterior.coords), round(P.area / 1e4, 2))
