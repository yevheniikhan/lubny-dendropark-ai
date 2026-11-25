"""Georeference the digitised quarter scheme (data/lltk/q/quarters_px.json) onto the image.
Main body (quarters 1-56): affine transform scheme -> brochure outline -> UTM (data/lltk/gcp.json, 12 GCPs).
Southern strip (quarters 57-68) is drawn out of scale on the scheme, so it gets its own projective transform
from four control points (data/lltk/q/gcp_tail.json): the two top corners sit just inside quarter 55 and the
strip is then clipped by the main body, so it meets 55 along one edge; the bottom corners and the eastern edge
follow the forest edge against the houses in the image. Both transforms keep straight lines straight."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import boundary as B
from skimage.transform import ProjectiveTransform

OX0, OY1, RES = 498880, 5537400, 0.5                      # ortho2_utm.png grid
TAIL = range(57, 69)


def fit(src, dst):
    A = np.c_[src, np.ones(len(src))]
    P, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return P, A @ P - dst


def transforms():
    n2o = np.load('data/lltk/q/new2old.npy')               # scheme px -> brochure px (outline registration)
    a, b, d, e, xo, yo = n2o                                # shapely order: x' = a x + b y + xo, y' = d x + e y + yo
    M3 = np.array([[a, d, 0], [b, e, 0], [xo, yo, 1]])
    Pb = M3 @ B.affine()[0]
    g = json.load(open('data/lltk/q/gcp_tail.json'))
    src = np.array([a[0] for a in g], float)
    dst = np.array([[OX0 + a[1][0] * RES, OY1 - a[1][1] * RES] for a in g], float)
    H = ProjectiveTransform.from_estimate(src, dst)
    return Pb, H, H(src) - dst


def quarters_utm():
    P = {int(q): np.array(c) for q, c in json.load(open('data/lltk/q/quarters_px.json')).items()}
    Pb, Pt, _ = transforms()
    # extend the top row of the strip (57, 58) up into quarter 55 on the scheme; clipped below
    from shapely.geometry import box
    p57, p58 = Polygon(P[57]), Polygon(P[58])
    xs = (p57.bounds[2] + p58.bounds[0]) / 2
    P[57] = np.array(unary_union([p57, box(p57.bounds[0], 340, xs, p57.bounds[1] + 4)]).exterior.coords)
    P[58] = np.array(unary_union([p58.difference(box(0, 0, xs, 999)), box(xs, 340, p58.bounds[2], p58.bounds[1] + 4)]).exterior.coords)
    out = {}
    for q, c in P.items():
        out[q] = Polygon(Pt(c) if q in TAIL else np.c_[c, np.ones(len(c))] @ Pb).buffer(0)
    # the strip's control points sit slightly inside quarter 55; clipping by the main body makes the seam exact
    body = unary_union([p for q, p in out.items() if q not in TAIL])
    for q in TAIL:
        g = out[q].difference(body)
        out[q] = g if g.geom_type == 'Polygon' else max(g.geoms, key=lambda h: h.area)
    # small notches on the outline (the eastern circle at quarter 35, seam corners) go to the adjacent quarter
    U = unary_union(list(out.values()))
    outer = U.buffer(8, join_style=2).buffer(-8, join_style=2)
    outer = Polygon(outer.exterior) if outer.geom_type == 'Polygon' else Polygon(max(outer.geoms, key=lambda g: g.area).exterior)
    gaps = outer.difference(U)
    for gp in getattr(gaps, 'geoms', [gaps]):
        if gp.is_empty or gp.area < 1:
            continue
        q = max(out, key=lambda k: out[k].buffer(0.5).intersection(gp.boundary).length)
        g = unary_union([out[q], gp]).buffer(0.05, join_style=2).buffer(-0.05, join_style=2)
        out[q] = g if g.geom_type == 'Polygon' else max(g.geoms, key=lambda h: h.area)
    return out


if __name__ == '__main__':
    Q = quarters_utm()
    Pb, Pt, res = transforms()
    print('tail residuals, m', np.round(np.hypot(*res.T), 1))
    outer = unary_union([p.buffer(0.1, join_style=2) for p in Q.values()]).buffer(-0.1, join_style=2)
    outer = Polygon(outer.exterior) if outer.geom_type == 'Polygon' else Polygon(max(outer.geoms, key=lambda g: g.area).exterior)
    outer = outer.simplify(0.8)
    json.dump({'utm': [list(map(lambda v: round(v, 2), xy)) for xy in outer.exterior.coords],
               'area_ha': round(outer.area / 1e4, 2),
               'method': 'outer boundary of the 68 digitised quarters of the park scheme, affine georeferencing'},
              open('data/park_boundary.json', 'w'), ensure_ascii=False)
    feats = [{'type': 'Feature', 'properties': {'kvartal': q, 'area_ha': round(p.area / 1e4, 2)},
              'geometry': mapping(p.simplify(0.3))} for q, p in sorted(Q.items())]
    json.dump({'type': 'FeatureCollection', 'name': 'dendropark_kvartaly',
               'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::32636'}}, 'features': feats},
              open('data/park_quarters_utm36n.geojson', 'w'), ensure_ascii=False)
    print(len(outer.exterior.coords), 'vertices', round(outer.area / 1e4, 2), 'ha; quarters sum',
          round(sum(p.area for p in Q.values()) / 1e4, 2))
