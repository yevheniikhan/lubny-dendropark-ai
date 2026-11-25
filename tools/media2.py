"""Animations for the Lubny dendropark deck, iteration 2.
Geometry: everything is drawn in EPSG:32636. The park boundary is the brochure map scheme warped onto the image
(tools/boundary.py); grids come from data/park2.npz (CHM 5 m, NDVI 20 m, AI zones 10 m, origin 498900 / 5537360)."""
import subprocess, sys, os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter
from shapely.geometry import Polygon
from shapely import contains_xy
from pyproj import Transformer
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT + "/tools")
import grids as G, boundary as B
OUT = ROOT + "/media/"; os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1280, 720, 25
BG = (251, 250, 247); DARK = (12, 58, 52); TEAL = (20, 82, 74); GREY = (90, 101, 96); TXT = (22, 33, 29); WHITE = (255, 255, 255)
LINEC = (255, 236, 160)
F = "/usr/share/fonts/truetype/crosextra/"
def font(sz, b=True): return ImageFont.truetype(F + ("Carlito-Bold.ttf" if b else "Carlito-Regular.ttf"), sz)

# ------------------------------------------------------------------ geometry and data
POLY = G.POLY
CHM, NDVI, ZONE = G.load()
OX0, OY1, ORES = 498880, 5537400, 0.5
ORTHO = np.asarray(Image.open("data/esri/ortho2_utm.png").convert("RGB")).astype(float)
VX0, VX1, VY0, VY1 = 498880, 499650, 5536460, 5537340
S = H / (VY1 - VY0)                                   # px per metre, map 594 x 720
MW, MH = int(round((VX1 - VX0) * S)), H
GX0, GY1 = 498880, 5537360

def view_xy(w=MW, h=MH, x0=VX0, y1=VY1, s=S):
    c, r = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
    return x0 + c / s, y1 - r / s
VX, VY = view_xy()
INSIDE = contains_xy(POLY, VX, VY)

def ortho_view(x, y):
    c = (x - OX0) / ORES - .5; r = (OY1 - y) / ORES - .5
    return np.stack([map_coordinates(ORTHO[..., k], [r, c], order=1, mode="nearest") for k in range(3)], -1)
BASE = ortho_view(VX, VY)

def grid_view(arr, cell, order=0):
    c = (VX - GX0) / cell - .5; r = (GY1 - VY) / cell - .5
    a = np.nan_to_num(arr, nan=0.0)
    v = map_coordinates(a, [r, c], order=order, mode="nearest")
    ok = map_coordinates(np.isfinite(arr).astype(float), [r, c], order=0, mode="constant", cval=0) > .5
    v[~(ok & INSIDE)] = np.nan
    return v

def cmap(stops):
    return LinearSegmentedColormap.from_list("c", [(p, np.array(h) / 255) for p, h in stops])
def hx(s): return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
CM_H = cmap([(0, hx("F4F1E8")), (.2, hx("D9E3C3")), (.48, hx("8DB87F")), (.72, hx("3B7B58")), (1, hx("0C3A34"))])
CM_N = cmap([(0, hx("8C6D3F")), (.32, hx("D8C9A0")), (.58, hx("B5D19A")), (.8, hx("4E9A5B")), (1, hx("0C3A34"))])
ZNAME = ["", "інші землі", "узлісся, галявини й молоді насадження", "зімкнені лісові насадження"]
ZCOL = [hx("C9C4B8"), hx("E3D7B4"), hx("8DB87F"), hx("2F6B4F")]
ZMAP = {0: 3, 1: 2, 4: 2, 2: 1, 3: 1}                 # k-means id -> display class (by canopy height)

def colorize(v, cm, vmin, vmax, base=None, alpha=1.0, dim=0.35):
    b = BASE * (1 - dim) if base is None else base
    rgb = cm(np.clip((np.nan_to_num(v) - vmin) / (vmax - vmin), 0, 1))[..., :3] * 255
    m = np.isfinite(v)[..., None]
    return np.where(m, rgb * alpha + b * (1 - alpha), b)

def outside_dim(img, a=0.45):
    return np.where(INSIDE[..., None], img, img * (1 - a) + np.array(BG) * 0 * a)

def zones_smooth():
    cls = np.full(ZONE.shape, -1)
    for k, v in ZMAP.items():
        cls[ZONE == k] = v
    prob = []
    for k in range(4):
        p = gaussian_filter((cls == k).astype(float), 0.9)
        c = (VX - GX0) / 10 - .5; r = (GY1 - VY) / 10 - .5
        prob.append(map_coordinates(p, [r, c], order=1, mode="nearest"))
    z = np.argmax(np.stack(prob), 0).astype(float); z[~INSIDE] = np.nan
    return z
def zones_raw():
    cls = np.full(ZONE.shape, np.nan)
    for k, v in ZMAP.items():
        cls[ZONE == k] = v
    return grid_view(cls, 10, 0)
def zone_rgb(z, base=None, dim=0.3):
    b = BASE * (1 - dim) if base is None else base
    out = b.copy()
    for k, c in enumerate(ZCOL):
        out[z == k] = c
    return out

def boundary_px(s=S, x0=VX0, y1=VY1, offx=0):
    return [((x - x0) * s + offx, (y1 - y) * s) for x, y in POLY.exterior.coords]

def to_img(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def draw_boundary(im, w=3, color=LINEC, frac=1.0, offx=0):
    pts = boundary_px(offx=offx)
    if frac < 1:
        n = max(2, int(len(pts) * frac)); pts = pts[:n]
    ImageDraw.Draw(im).line(pts, fill=color, width=w, joint="curve")
    return im

def ease(t): t = min(max(t, 0), 1); return t * t * (3 - 2 * t)

class Writer:
    def __init__(self, name, w=W, h=H):
        self.name = name
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT + name + ".mp4"],
                                  stdin=subprocess.PIPE)
        self.last = None
    def add(self, im, n=1):
        b = im.convert("RGB").tobytes(); self.last = im
        for _ in range(n): self.p.stdin.write(b)
    def close(self, poster=None):
        self.p.stdin.close(); self.p.wait()
        (poster or self.last).convert("RGB").save(OUT + self.name + "_poster.jpg", quality=92)

def label(dr, xy, txt, sz=34, fill=TXT, b=True, anchor="la", bg=None, pad=10):
    f = font(sz, b)
    if bg:
        x0, y0, x1, y1 = dr.textbbox(xy, txt, font=f, anchor=anchor)
        dr.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=bg)
    dr.text(xy, txt, font=f, fill=fill, anchor=anchor)

PX = 650                                                # legend panel x
def frame(mapimg, title=None, sub=None, legend=None):
    fr = Image.new("RGB", (W, H), BG); fr.paste(mapimg, (0, 0))
    dr = ImageDraw.Draw(fr, "RGBA")
    y = 54
    if title:
        label(dr, (PX, y), title, 44, fill=DARK); y += 64
    if sub:
        for s_ in sub:
            label(dr, (PX, y), s_, 34, fill=GREY, b=False); y += 44
    if legend: legend(dr, PX, y + 30)
    L = 100 * S                                        # 100 m scale bar
    dr.rectangle((24, H - 40, 24 + L, H - 33), fill=(255, 255, 255, 235))
    label(dr, (24, H - 48), "100 м", 24, fill=WHITE, anchor="lb")
    return fr

def colorbar(dr, x, y, cm, vmin, vmax, ticks, unit, w=470, h=28):
    for i in range(w):
        c = tuple(int(255 * v) for v in cm(i / (w - 1))[:3]); dr.line((x + i, y, x + i, y + h), fill=c)
    for tv in ticks:
        px = x + (tv - vmin) / (vmax - vmin) * (w - 1)
        dr.line((px, y + h, px, y + h + 8), fill=GREY, width=2)
        label(dr, (px, y + h + 12), f"{tv:g}".replace(".", ","), 28, fill=TXT, b=False, anchor="ma")
    if unit: label(dr, (x + w + 14, y + h // 2), unit, 28, fill=TXT, b=False, anchor="lm")

def zone_legend(dr, x, y, alpha=1.0, areas=None):
    for k in (3, 2, 1):
        yy = y + (3 - k) * (110 if areas is not None else 62)
        dr.rectangle((x, yy, x + 38, yy + 38), fill=ZCOL[k] + (int(255 * alpha),))
        label(dr, (x + 56, yy + 19), ZNAME[k], 32, fill=TXT + (int(255 * alpha),), b=False, anchor="lm")
        if areas is not None:
            label(dr, (x + 56, yy + 62), f"{areas[k]:.1f} га".replace(".", ","), 34, fill=TXT, b=True, anchor="lm")

# ------------------------------------------------------------------ 1. zoom from space
def zoom_video():
    lv = [Image.open(f"data/zoom2/z{i}.png").convert("RGB") for i in range(5)]
    HW = [320, 80, 20, 5, 1.25]
    names = ["Україна", "Полтавщина", "Лубни", "Лубни", "дендропарк коледжу"]
    cx, cy = Transformer.from_crs(4326, 32636, always_xy=True).transform(32.990738, 49.984764)
    FEND = 1.5
    wv = Writer("zoom_from_space")
    seg, hold0 = 1.8, 0.8
    T1 = hold0 + 4 * seg; T2 = T1 + 2.2; T3 = T2 + 3.0
    for fi in range(int(T3 * FPS)):
        t = fi / FPS
        if t < hold0: z = 0.0
        elif t < T1: z = 4 * ease((t - hold0) / (4 * seg))
        else: z = 4 + np.log(FEND) / np.log(4) * ease((t - T1) / (T2 - T1))
        k = min(int(z), 4); f = 4 ** (z - k)
        A = lv[k]; cw, ch = 800 / f, 450 / f
        fr = A.crop((400 - cw / 2, 225 - ch / 2, 400 + cw / 2, 225 + ch / 2)).resize((W, H), Image.BICUBIC)
        if k < 4:
            sw, sh = W * f / 4, H * f / 4
            if sw > 4:
                Bm = lv[k + 1].resize((int(sw), int(sh)), Image.BICUBIC)
                fe = max(2, int(min(Bm.size) * 0.08))
                m = np.ones(Bm.size[::-1]); rmp = np.linspace(0, 1, fe)
                m[:fe] *= rmp[:, None]; m[-fe:] *= rmp[::-1][:, None]; m[:, :fe] *= rmp[None]; m[:, -fe:] *= rmp[::-1][None]
                fr.paste(Bm, (int(W / 2 - sw / 2), int(H / 2 - sh / 2)), Image.fromarray((m * 255).astype(np.uint8)))
        gw = 2 * HW[k] * 1000 / f                       # metres across the frame
        if z > 3.4:                                     # high-resolution image of the park area on top
            a = ease((z - 3.4) / 0.5)
            mpp = gw / W
            X, Y = view_xy(W, H, cx - gw / 2, cy + gw * H / W / 2, 1 / mpp)
            inside = (X > OX0 + 2) & (X < OX0 + 798) & (Y < OY1 - 2) & (Y > OY1 - 998)
            hi = ortho_view(X, Y)
            edge = np.minimum.reduce([X - OX0, OX0 + 800 - X, OY1 - Y, Y - (OY1 - 1000)]) / 60
            wgt = np.clip(edge, 0, 1)[..., None] * inside[..., None] * a
            fr = to_img(np.asarray(fr).astype(float) * (1 - wgt) + hi * wgt)
        dr = ImageDraw.Draw(fr, "RGBA")
        if t > T2 - 0.4:                                # boundary drawn as a growing line
            u = ease((t - T2 + 0.4) / 1.6)
            mpp = gw / W
            pts = [((x - cx) / mpp + W / 2, (cy - y) / mpp + H / 2) for x, y in POLY.exterior.coords]
            n = max(2, int(len(pts) * u)); dr.line(pts[:n], fill=LINEC + (255,), width=4, joint="curve")
        nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
        L = [n for n in nice if n <= gw / 5000][-1]; px = W * L * 1000 / gw
        dr.rectangle((40, H - 60, 40 + px, H - 52), fill=(255, 255, 255, 235))
        label(dr, (40, H - 70), f"{L:g} км" if L >= 1 else f"{int(L * 1000)} м", 30, fill=WHITE, anchor="lb")
        nm = names[min(int(round(z)), 4)]
        label(dr, (W - 40, 50), nm, 40, fill=WHITE, anchor="ra", bg=(12, 58, 52, 200), pad=12)
        wv.add(fr)
    wv.close()

# ------------------------------------------------------------------ 2. map scheme georeferencing
def scheme_video():
    """Vector quarter scheme (68 digitised quarters) morphs from the scheme onto the image."""
    import quarters_georef as QG
    from skimage.transform import ProjectiveTransform
    Ppx = {int(q): np.array(c) for q, c in json.load(open("data/lltk/q/quarters_px.json")).items()}
    Pb, Ht, _ = QG.transforms()
    final = QG.quarters_utm()
    tgt = {q: (Ht(c) if q in QG.TAIL else np.c_[c, np.ones(len(c))] @ Pb) for q, c in Ppx.items()}
    c0 = np.array([(VX0 + VX1) / 2, (VY0 + VY1) / 2])           # scheme starts north-up in the middle of the view
    s0 = 1.62
    src = {q: np.c_[(c[:, 0] - 220) * s0 + c0[0], -(c[:, 1] - 240) * s0 + c0[1]] for q, c in Ppx.items()}
    SS = 2                                                          # supersampling for smooth lines
    def to_px(xy):
        return [((x - VX0) * S * SS, (VY1 - y) * S * SS) for x, y in xy]
    base = Image.fromarray(BASE.astype(np.uint8))
    def render(t, fill_a=0.55, line=(255, 236, 160), lw=2, numbers=True, use_final=False, gcp=None, show=1.0):
        im = base.resize((MW * SS, MH * SS), Image.BILINEAR).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        polys = {q: (np.array(final[q].exterior.coords) if use_final else src[q] * (1 - t) + tgt[q] * t) for q in Ppx}
        for q, c in polys.items():
            dr.polygon(to_px(c), fill=(18, 70, 52, int(255 * fill_a * show)), outline=line + (int(255 * show),), width=lw * SS)
        if numbers:
            f = font(13 * SS, True)
            for q, c in polys.items():
                p = Polygon(c).representative_point(); x, y = to_px([(p.x, p.y)])[0]
                dr.text((x, y), str(q), font=f, fill=(255, 255, 255, int(255 * show)), anchor="mm")
        if gcp:
            for x, y in gcp:
                dr.ellipse((x - 10 * SS, y - 10 * SS, x + 10 * SS, y + 10 * SS), fill=(12, 58, 52, 255), outline=(255, 255, 255, 255), width=3 * SS)
        im = Image.alpha_composite(im, ov).convert("RGB")
        return im.resize((MW, MH), Image.LANCZOS)
    def leg_steps(k):
        def f(dr, x, y):
            steps = ["схема кварталів коледжу", "оцифрування: 68 кварталів", "опорні точки на знімку",
                     "афінне перетворення", "векторна межа парку"]
            for i, s_ in enumerate(steps):
                on = i <= k
                dr.rectangle((x, y + i * 60 + 6, x + 22, y + i * 60 + 28), fill=(DARK if on else (200, 199, 191)))
                label(dr, (x + 40, y + i * 60 + 17), s_, 33, fill=(TXT if on else (170, 170, 165)), b=False, anchor="lm")
        return f
    T, SUB = "Квартали і знімок", ["прив'язка (геореференсування)"]
    wv = Writer("scheme_georef")
    wv.add(frame(base, T, SUB, leg_steps(-1)), int(0.8 * FPS))
    for i in range(int(1.0 * FPS)):
        wv.add(frame(render(0, numbers=False, show=ease(i / FPS)), T, SUB, leg_steps(0)))
    for i in range(int(1.0 * FPS)):
        wv.add(frame(render(0, show=1, numbers=True), T, SUB, leg_steps(1)))
    g = json.load(open("data/lltk/gcp.json")); gt = json.load(open("data/lltk/q/gcp_tail.json"))
    gp = [((a[1][0] - VX0) * S * SS, (VY1 - a[1][1]) * S * SS) for a in g]
    gp += [((OX0 + a[1][0] * ORES - VX0) * S * SS, (VY1 - (OY1 - a[1][1] * ORES)) * S * SS) for a in gt[2:]]
    for i in range(int(1.2 * FPS)):
        n = int(len(gp) * min(1, (i + 1) / (0.9 * FPS)))
        wv.add(frame(render(0, gcp=gp[:n]), T, SUB, leg_steps(2)))
    for i in range(int(3.0 * FPS)):
        wv.add(frame(render(ease(i / (3.0 * FPS)), gcp=gp), T, SUB, leg_steps(3)))
    for i in range(int(1.0 * FPS)):
        u = ease(i / FPS)
        wv.add(frame(render(1, use_final=True, fill_a=0.55 * (1 - u) + 0.12 * u), T, SUB, leg_steps(3)))
    fin = draw_boundary(render(1, use_final=True, fill_a=0.12), w=4, color=WHITE)
    fin = frame(fin, T, SUB, leg_steps(4))
    wv.add(fin, int(2.5 * FPS)); wv.close(fin)

# ------------------------------------------------------------------ 3. layers
def layers_video():
    o = to_img(outside_dim(BASE, 0.35))
    chm = to_img(colorize(grid_view(CHM, 5, 1), CM_H, 0, 25))
    nd = to_img(colorize(grid_view(NDVI[7], 20, 0), CM_N, 0, 0.95))
    zs = zones_smooth(); zon = to_img(zone_rgb(zs))
    for im in (o, chm, nd, zon): draw_boundary(im)
    st = [(o, "Знімок", ["що бачить око"], None),
          (chm, "Висота крон", ["Meta і WRI, 1 м", "показано сітку 5 м"],
           lambda dr, x, y: colorbar(dr, x, y, CM_H, 0, 25, [0, 5, 10, 15, 20, 25], "м")),
          (nd, "NDVI, липень", ["Sentinel-2, 2024", "піксель 20 м"],
           lambda dr, x, y: colorbar(dr, x, y, CM_N, 0, .95, [0, .3, .6, .9], "")),
          (zon, "Зони ШІ", ["AlphaEarth, 64 ознаки", "k-середніх"], lambda dr, x, y: zone_legend(dr, x, y))]
    wv = Writer("layers_scan"); hold, sweep = 2.4, 1.2
    for si, (img, ti, sub, leg) in enumerate(st):
        if si:
            prev = st[si - 1][0]
            for k in range(int(sweep * FPS)):
                xc = int(ease(k / (sweep * FPS)) * MW)
                m = prev.copy(); m.paste(img.crop((0, 0, xc, MH)), (0, 0))
                dr = ImageDraw.Draw(m, "RGBA")
                for gg in range(16):
                    dr.line((xc - gg, 0, xc - gg, MH), fill=(255, 255, 255, int(210 * (1 - gg / 16) ** 2)))
                wv.add(frame(m, ti, sub, leg))
        wv.add(frame(img, ti, sub, leg), int(hold * FPS))
    poster = frame(chm, st[1][1], st[1][2], st[1][3]); wv.close(poster)

# ------------------------------------------------------------------ 4. NDVI seasons
MONTHS = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень", "вересень", "жовтень", "листопад", "грудень"]
def ndvi_video():
    valid = [m for m in range(1, 13) if np.isfinite(NDVI[m]).any()]
    mean = {m: float(np.nanmean(NDVI[m])) for m in valid}
    imgs = {m: draw_boundary(to_img(colorize(grid_view(NDVI[m], 20, 0), CM_N, 0, .95))) for m in valid}
    cx0, cw, chh = PX + 40, 520, 300
    def chart(dr, y0, upto, frac=1.0):
        dr.line((cx0, y0 + chh, cx0 + cw, y0 + chh), fill=GREY, width=2)
        for v in (0, .5, 1):
            y = y0 + chh - v * chh
            dr.line((cx0, y, cx0 + cw, y), fill=(214, 212, 204), width=1)
            label(dr, (cx0 - 12, y), f"{v:g}".replace(".", ","), 26, fill=GREY, b=False, anchor="rm")
        for m in range(1, 13):
            label(dr, (cx0 + (m - .5) / 12 * cw, y0 + chh + 10), MONTHS[m - 1][:3], 22, fill=GREY, b=False, anchor="ma")
        pts = [(cx0 + (m - .5) / 12 * cw, y0 + chh - mean[m] * chh, m) for m in valid if m <= upto]
        for i in range(1, len(pts)):
            (xa, ya, ma), (xb, yb, mb) = pts[i - 1], pts[i]
            if i == len(pts) - 1 and frac < 1:
                xb, yb = xa + (xb - xa) * frac, ya + (yb - ya) * frac
            if mb - ma > 1:
                for s_ in np.arange(0, 1, .1):
                    dr.line((xa + (xb - xa) * s_, ya + (yb - ya) * s_, xa + (xb - xa) * (s_ + .05), ya + (yb - ya) * (s_ + .05)), fill=GREY, width=3)
            else:
                dr.line((xa, ya, xb, yb), fill=TEAL, width=5)
        for (x, y, m) in pts[:len(pts) - (1 if frac < 1 else 0)]:
            dr.ellipse((x - 7, y - 7, x + 7, y + 7), fill=TEAL)
    wv = Writer("ndvi_seasons"); per, fade = 1.0, .5
    for i, m in enumerate(valid):
        nxt = valid[i + 1] if i + 1 < len(valid) else None
        for k in range(int(per * FPS)):
            t = k / FPS; img = imgs[m]
            if nxt and t > per - fade: img = Image.blend(imgs[m], imgs[nxt], ease((t - (per - fade)) / fade))
            fr = Image.new("RGB", (W, H), BG); fr.paste(img, (0, 0)); dr = ImageDraw.Draw(fr, "RGBA")
            label(dr, (PX, 54), f"{MONTHS[m - 1].capitalize()} 2024", 44, fill=DARK)
            label(dr, (PX, 118), f"середній NDVI {mean[m]:.2f}".replace(".", ","), 32, fill=GREY, b=False)
            colorbar(dr, PX + 40, 190, CM_N, 0, .95, [0, .3, .6, .9], "", w=420)
            chart(dr, 300, m, frac=min(1, t / per * 1.6) if i else 1)
            L = 100 * S; dr.rectangle((24, H - 40, 24 + L, H - 33), fill=(255, 255, 255, 235))
            label(dr, (24, H - 48), "100 м", 24, fill=WHITE, anchor="lb")
            wv.add(fr)
    wv.add(fr, int(2.0 * FPS)); wv.close(Image.open(OUT + "ndvi_seasons.mp4") if False else None)

# ------------------------------------------------------------------ 5. AI zones
def zones_video():
    zr = zones_raw(); zs = zones_smooth()
    area = [float(np.sum(zs == k)) / S ** 2 / 1e4 for k in range(4)]
    base = outside_dim(BASE, 0.35) * 0.8 + 0.2 * 255
    rng = np.random.default_rng(3)
    cells = np.argwhere(np.isfinite(ZONE)); order = rng.permutation(len(cells))
    reveal = np.full(ZONE.shape, np.inf)
    for i, j in enumerate(order): reveal[tuple(cells[j])] = i / len(order)
    rv = grid_view(reveal, 10, 0)
    wv = Writer("ai_zones")
    T, sub = "ШІ ділить парк на зони", ["без жодної підказки"]
    first = draw_boundary(to_img(base)); wv.add(frame(first, T, sub), FPS)
    n = int(3.2 * FPS)
    for s_ in range(n):
        u = ease((s_ + 1) / n)
        z = np.where(rv <= u, zr, np.nan)
        im = draw_boundary(to_img(zone_rgb(z, base=base, dim=0)))
        wv.add(frame(im, T, sub, lambda dr, x, y, a=min(1, u * 1.4): zone_legend(dr, x, y, alpha=a)))
    a_raw = zone_rgb(zr, base=base, dim=0); a_s = zone_rgb(zs, base=base, dim=0)
    for s_ in range(int(1.2 * FPS)):
        u = ease(s_ / (1.2 * FPS))
        im = draw_boundary(to_img(a_raw * (1 - u) + a_s * u))
        wv.add(frame(im, T, ["згладжування меж"], lambda dr, x, y: zone_legend(dr, x, y)))
    fin = to_img(a_s); dr = ImageDraw.Draw(fin)
    for k in range(4):                                    # thin outlines between zones
        e = (zs == k); edge = e & ~np.roll(e, 1, 0) | e & ~np.roll(e, 1, 1)
        ys, xs = np.where(edge & INSIDE)
        for x, y in zip(xs[::2], ys[::2]): dr.point((x, y), fill=WHITE)
    draw_boundary(fin)
    fin = frame(fin, "3 типи ділянок", ["площі у межах парку"], lambda dr, x, y: zone_legend(dr, x, y, areas=area))
    wv.add(fin, int(3 * FPS)); wv.close(fin)
    json.dump({"zone_area_ha": {ZNAME[k]: round(area[k], 2) for k in (1, 2, 3)}}, open("data/zones_area.json", "w"), ensure_ascii=False)

# ------------------------------------------------------------------ 6. 3D canopy
def canopy3d_video(nfr=200, ex=2.0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from scipy.ndimage import zoom as nzoom
    Hh = nzoom(np.nan_to_num(CHM, nan=0.0), 2, order=1)
    ny, nx = Hh.shape
    xs = GX0 + (np.arange(nx) + .5) * 2.5; ys = GY1 - (np.arange(ny) + .5) * 2.5
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(POLY, X, Y)
    Hh = np.where(inside, Hh, 0)
    tex = ortho_view(X, Y) / 255
    hs = LightSource(315, 35).hillshade(Hh, vert_exag=1.5, dx=2.5, dy=2.5)
    col = np.clip(tex * (0.45 + 0.85 * hs[..., None]) * 1.1, 0, 1)
    col = np.where(inside[..., None], col, col * 0.4 + 0.55)
    wv = Writer("canopy_3d")
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); fig.patch.set_facecolor(np.array(BG) / 255)
    ax = fig.add_axes([-0.3, -0.45, 1.6, 1.9], projection="3d")
    ax.plot_surface(X - X.mean(), Y - Y.mean(), Hh * ex, facecolors=col, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_axis_off(); ax.set_box_aspect((nx * 2.5, ny * 2.5, 25 * ex * 2.2)); ax.set_zlim(0, 25 * ex * 2.2)
    ax.set_facecolor(np.array(BG) / 255)
    poster = None
    for i in range(nfr):
        ax.view_init(elev=32 - 6 * np.sin(2 * np.pi * i / nfr), azim=-70 + 360 * i / nfr)
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        label(ImageDraw.Draw(im), (W - 30, H - 30), f"вертикаль × {ex:g}", 28, fill=GREY, b=False, anchor="rd")
        wv.add(im)
        if i == 10: poster = im.copy()
    plt.close(fig); wv.close(poster)

def stills():
    im = draw_boundary(to_img(outside_dim(BASE, 0.25)), w=4)
    im.save(OUT + "park_ortho_boundary.jpg", quality=92)
    big = Image.fromarray(ORTHO.astype(np.uint8)).crop((0, 0, 1600, 2000))
    dr = ImageDraw.Draw(big)
    dr.line([((x - OX0) * 2, (OY1 - y) * 2) for x, y in POLY.exterior.coords], fill=LINEC, width=7, joint="curve")
    big.save(OUT + "title_park.jpg", quality=90)
    thumbs = {"th_ortho": to_img(BASE), "th_chm": to_img(colorize(grid_view(CHM, 5, 1), CM_H, 0, 25)),
              "th_ndvi": to_img(colorize(grid_view(NDVI[7], 20, 0), CM_N, 0, .95)), "th_zones": to_img(zone_rgb(zones_smooth()))}
    for k, v in thumbs.items():
        draw_boundary(v, w=3).crop((40, 60, 560, 460)).save(OUT + k + ".jpg", quality=90)

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["stills", "zoom", "scheme", "layers", "ndvi", "zones", "3d"]):
        {"stills": stills, "zoom": zoom_video, "scheme": scheme_video, "layers": layers_video, "ndvi": ndvi_video,
         "zones": zones_video, "3d": canopy3d_video}[w](); print("done", w, flush=True)
