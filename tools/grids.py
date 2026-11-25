"""Load the park grids (EPSG:32636, origin X0=498900, Y1=5537360) and mask them with the park boundary."""
import numpy as np, json
from shapely.geometry import Polygon
from shapely import contains_xy
X0, Y1 = 498880, 5537360
d = np.load('data/park3.npz')
POLY = Polygon(json.load(open('data/park_boundary.json'))['utm'])
def mask(shape, cell):
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    return contains_xy(POLY, X0 + (c + .5) * cell, Y1 - (r + .5) * cell)
def load():
    H = d['H5'] / 2; H[~mask(H.shape, 5)] = np.nan
    N = {}
    for m in range(1, 13):
        a = d['N%02d' % m] / 40 - 0.1; a[~mask(a.shape, 20)] = np.nan; N[m] = a
    Z = d['ZS'].astype(float); Z[~mask(Z.shape, 10)] = np.nan
    return H, N, Z
