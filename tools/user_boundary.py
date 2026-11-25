"""Park boundary drawn by the author (red line over the image, data/lltk/q/user_outline_src.png), converted to UTM.
The screenshot is registered to the previous boundary drawn in it (mean misfit < 1 m), the red line is filled,
traced and simplified to straight segments."""
import numpy as np, json
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
from shapely.geometry import Polygon

def user_polygon(tol=2.5):
    a = np.array(Image.open('data/lltk/q/user_outline_src.png').convert('RGB')).astype(int)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    red = (R > 170) & (G < 90) & (B < 90)
    red = ndi.binary_closing(red, iterations=1)
    from skimage.draw import line as dline
    for (x0, y0), (x1, y1) in [((474, 125), (479, 133)), ((218, 447), (217, 453)), ((245, 401), (240, 409))]:
        rr, cc = dline(y0, x0, y1, x1); red[rr, cc] = True          # gaps between the pen strokes
    filled = ndi.binary_fill_holes(red)
    lab, n = ndi.label(filled)
    m = lab == (np.argmax(ndi.sum(filled, lab, range(1, n + 1))) + 1)
    c = max(measure.find_contours(np.pad(m, 1).astype(float), 0.5), key=len) - 1
    s, tx, ty = np.load('data/lltk/q/shot2utm.npy')
    # the red stroke is ~5 px wide: its centre line is ~2.5 px inside the outer contour
    P = Polygon(np.c_[tx + (c[:, 1] + .5) * s, ty - (c[:, 0] + .5) * s]).buffer(0).buffer(-2.5 * s, join_style=2)
    return P.simplify(tol * s, preserve_topology=True)

if __name__ == '__main__':
    P = user_polygon()
    print(len(P.exterior.coords), round(P.area / 1e4, 2))
