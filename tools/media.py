"""Animations for the Lubny dendropark deck.
Inputs (data/): park.npz (CHM 5 m, NDVI 20 m x 12 months, AI zones 10 m; all EPSG:32636),
esri/ortho_utm.png (Esri World Imagery resampled to 0.5 m UTM), zoom/z0..z5.png (Esri, Web Mercator, centred on the star).
Outputs (media/): MP4 (H.264) + poster JPG for each animation, PNG stills for the slides."""
import subprocess, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = f"{ROOT}/data/", f"{ROOT}/media/"
os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1280, 720, 25
BG = (251, 250, 247); DARK = (12, 58, 52); TEAL = (20, 82, 74); OCHRE = (154, 107, 47)
GREY = (90, 101, 96); TXT = (22, 33, 29); WHITE = (255, 255, 255); YEL = (240, 200, 80)
F = "/usr/share/fonts/truetype/crosextra/"
def font(sz, b=True): return ImageFont.truetype(F + ("Carlito-Bold.ttf" if b else "Carlito-Regular.ttf"), sz)

# ---------------------------------------------------------------- data
X0, Y1 = 498920, 5537280            # common UTM origin (upper left) of every grid
d = np.load(DATA + "park.npz")
CHM = d["H5"] / 2                                   # m, 5 m grid
NDVI = {m: d["N%02d" % m] / 40 - 0.1 for m in range(1, 13)}
for m in (2, 12):                                   # no cloud-free Sentinel-2 composite in Feb and Dec 2024
    NDVI[m] = np.where(np.isfinite(NDVI[m]), np.nan, np.nan)
ZONE = d["Z10"]
ORTHO = Image.open(DATA + "esri/ortho_utm.png").convert("RGB")   # 1360 x 1400, 0.5 m
PC = [[32.985841, 49.985279], [32.988085, 49.986922], [32.990524, 49.987288], [32.993484, 49.986535],
      [32.993256, 49.983729], [32.991256, 49.982243], [32.989223, 49.98284], [32.989744, 49.983782],
      [32.987109, 49.984671], [32.98589, 49.984724], [32.985841, 49.985279]]
T = Transformer.from_crs(4326, 32636, always_xy=True)
POLY = np.array([T.transform(*p) for p in PC])
STAR = T.transform(32.99069, 49.98555)
MS = 680 / 680                                      # map px per metre in the videos (map 680 x 700 px)
def to_px(x, y, s=MS): return ((x - X0) * s, (Y1 - y) * s)

def grid_img(arr, cell_m, cmap, vmin, vmax, s=MS):
    """Colour a masked grid and upscale with nearest neighbour to s px/m; returns RGBA."""
    from matplotlib import colormaps
    a = (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin)
    rgba = (colormaps[cmap](np.nan_to_num(a)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(arr), 255, 0)
    im = Image.fromarray(rgba, "RGBA")
    return im.resize((int(round(arr.shape[1] * cell_m * s)), int(round(arr.shape[0] * cell_m * s))), Image.NEAREST)

ZCOL = [(196, 150, 72), (124, 170, 104), (232, 222, 170), (24, 96, 72), (70, 120, 160)]   # a..e
def zone_img(s=MS):
    z = ZONE; rgba = np.zeros(z.shape + (4,), np.uint8)
    for k, c in enumerate(ZCOL):
        rgba[z == k] = c + (255,)
    im = Image.fromarray(rgba, "RGBA")
    return im.resize((int(z.shape[1] * 10 * s), int(z.shape[0] * 10 * s)), Image.NEAREST)

def paste_at(base, im, xy=(0, 0)):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0)); layer.paste(im, xy); return Image.alpha_composite(base, layer)

def poly_layer(size, s=MS, color=YEL, w=3, off=(0, 0)):
    L = Image.new("RGBA", size, (0, 0, 0, 0)); dr = ImageDraw.Draw(L)
    pts = [(to_px(x, y, s)[0] + off[0], to_px(x, y, s)[1] + off[1]) for x, y in POLY]
    dr.line(pts, fill=color + (255,), width=w, joint="curve"); return L

def ease(t): t = min(max(t, 0), 1); return t * t * (3 - 2 * t)

class Writer:
    def __init__(self, name, w=W, h=H):
        self.name = name
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT + name + ".mp4"],
                                  stdin=subprocess.PIPE)
        self.last = None
    def add(self, im, n=1):
        b = im.convert("RGB").tobytes(); self.last = im
        for _ in range(n): self.p.stdin.write(b)
    def close(self, poster=None):
        self.p.stdin.close(); self.p.wait()
        (poster or self.last).convert("RGB").save(OUT + self.name + "_poster.jpg", quality=92)

def label(dr, xy, txt, sz=34, fill=WHITE, bg=None, pad=10, b=True, anchor="la"):
    f = font(sz, b)
    if bg:
        x0, y0, x1, y1 = dr.textbbox(xy, txt, font=f, anchor=anchor)
        dr.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=bg)
    dr.text(xy, txt, font=f, fill=fill, anchor=anchor)

# ---------------------------------------------------------------- 1. zoom from space
def zoom_video():
    lv = [Image.open(DATA + f"zoom/z{i}.png").convert("RGB") for i in range(6)]
    names = ["Україна", "Полтавщина", "Лубни", "Лубни", "дендропарк коледжу", "центр дендропарку"]
    HW = [320, 80, 20, 5, 1.25, 0.32]                 # half-width of each image on the ground, km
    wv = Writer("zoom_from_space")
    seg, hold0, n_end = 2.0, 1.0, 3.2
    total = int((hold0 + 5 * seg + n_end) * FPS)
    for fi in range(total):
        t = fi / FPS
        if t < hold0: z = 0.0
        else:
            u = (t - hold0) / (5 * seg)
            z = 5 * ease(u) if u < 1 else 5 + 0.25 * ease((t - hold0 - 5 * seg) / n_end)   # z in "levels"
        k = min(int(z), 5); f = 4 ** (z - k)
        frame = Image.new("RGB", (W, H))
        A = lv[k]; cw, ch = 800 / f, 450 / f
        frame.paste(A.crop((400 - cw / 2, 225 - ch / 2, 400 + cw / 2, 225 + ch / 2)).resize((W, H), Image.BICUBIC))
        if k < 5:                                        # next level as a feathered centre inset
            sw, sh = W * f / 4, H * f / 4
            if sw > 4:
                B = lv[k + 1].resize((int(sw), int(sh)), Image.BICUBIC)
                m = Image.new("L", B.size, 0); ImageDraw.Draw(m).rectangle((0, 0, B.size[0], B.size[1]), fill=255)
                fe = max(2, int(min(B.size) * 0.06)); m = m.filter(ImageFilter.GaussianBlur(fe / 2))
                mi = np.array(m).astype(float); mi[:fe, :] *= np.linspace(0, 1, fe)[:, None]; mi[-fe:, :] *= np.linspace(1, 0, fe)[:, None]
                mi[:, :fe] *= np.linspace(0, 1, fe)[None]; mi[:, -fe:] *= np.linspace(1, 0, fe)[None]
                frame.paste(B, (int(W / 2 - sw / 2), int(H / 2 - sh / 2)), Image.fromarray(mi.astype(np.uint8)))
        dr = ImageDraw.Draw(frame, "RGBA")
        # scale bar: frame ground width
        gw = 2 * HW[k] / f                                # km across the frame
        nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
        L = [n for n in nice if n <= gw / 5][-1]
        px = W * L / gw
        dr.rectangle((40, H - 60, 40 + px, H - 52), fill=(255, 255, 255, 235))
        lab = (f"{L:g} км" if L >= 1 else f"{int(L * 1000)} м")
        label(dr, (40, H - 70), lab, 30, anchor="lb")
        nm = names[min(int(round(z)), 5)]
        label(dr, (W - 40, 50), nm, 40, anchor="ra", bg=(12, 58, 52, 200), pad=12)
        if z > 5.05:                                    # ring around the central clearing (star centre)
            a = int(255 * ease((z - 5.05) / 0.15))
            fz = 4 ** (z - 5); cx = W / 2 + (STAR_OFF[0]) * fz; cy = H / 2 + STAR_OFF[1] * fz
            r = 62 * fz
            dr.ellipse((cx - r, cy - r, cx + r, cy + r), outline=YEL + (a,), width=4)
        wv.add(frame)
    wv.close()

# offset of the star clearing from the centre of level-5 frame (px at 1280 wide); clearing sits at the centre by design
STAR_OFF = (11.0, -35.0)

# ---------------------------------------------------------------- base map frame (park, UTM)
MAPW, MAPH = 680, 700
def base_map(dim=0.0):
    o = ORTHO.resize((MAPW, MAPH), Image.LANCZOS).convert("RGBA")
    if dim:
        o = Image.alpha_composite(o, Image.new("RGBA", o.size, (0, 0, 0, int(255 * dim))))
    return o

def map_frame(mapimg, title, sub=None, legend=None, extra=None):
    fr = Image.new("RGB", (W, H), BG); fr.paste(mapimg.convert("RGB"), (10, 10))
    dr = ImageDraw.Draw(fr, "RGBA")
    x = 730
    label(dr, (x, 60), title, 46, fill=DARK)
    if sub:
        yy = 120
        for line in sub:
            label(dr, (x, yy), line, 34, fill=GREY, b=False); yy += 44
    if legend: legend(dr, x)
    if extra: extra(dr)
    # 100 m scale bar on the map
    dr.rectangle((40, H - 46, 40 + 100 * MS, H - 40), fill=(255, 255, 255, 240))
    label(dr, (40, H - 54), "100 м", 24, anchor="lb")
    return fr

def colorbar(dr, x, y, cmap, vmin, vmax, ticks, unit, w=440, h=26):
    from matplotlib import colormaps
    cm = colormaps[cmap]
    for i in range(w):
        c = tuple(int(255 * v) for v in cm(i / (w - 1))[:3]); dr.line((x + i, y, x + i, y + h), fill=c)
    for tv in ticks:
        px = x + (tv - vmin) / (vmax - vmin) * (w - 1)
        dr.line((px, y + h, px, y + h + 8), fill=GREY, width=2)
        label(dr, (px, y + h + 12), f"{tv:g}".replace(".", ","), 26, fill=TXT, b=False, anchor="ma")
    label(dr, (x + w + 14, y + h // 2), unit, 26, fill=TXT, b=False, anchor="lm")

def zone_legend(dr, x, y0=420, alpha=1.0, names=None):
    names = names or ZNAMES
    for k in ZORDER:
        y = y0 + ZORDER.index(k) * 56
        dr.rectangle((x, y, x + 34, y + 34), fill=ZCOL[k] + (int(255 * alpha),))
        label(dr, (x + 50, y + 17), names[k], 31, fill=TXT + (int(255 * alpha),), b=False, anchor="lm")

ZORDER = [2, 0, 1, 3, 4]
ZNAMES = {2: "відкриті ділянки, ~1 м", 0: "молоді й розріджені, ~9 м", 1: "середньовисокі, ~13 м",
          3: "високі, тип А, ~15 м", 4: "високі, тип Б, ~16 м"}

def layers_video():
    o = base_map()
    pl = poly_layer((MAPW, MAPH))
    chm = paste_at(base_map(0.35), grid_img(CHM, 5, "viridis", 0, 25))
    ndvi = paste_at(base_map(0.35), grid_img(NDVI[7], 20, "RdYlGn", 0.0, 0.95))
    zones = paste_at(base_map(0.35), zone_img())
    stages = [
        (o, "Знімок", ["що бачить око: кольори RGB"], None),
        (chm, "Висота крон", ["модель Meta і WRI, 1 м", "показано сітку 5 м"],
         lambda dr, x: colorbar(dr, x, 300, "viridis", 0, 25, [0, 5, 10, 15, 20, 25], "м")),
        (ndvi, "NDVI, липень 2024", ["Sentinel-2, піксель 20 м", "чим зеленіше, тим активніша листва"],
         lambda dr, x: colorbar(dr, x, 300, "RdYlGn", 0, 0.95, [0, 0.3, 0.6, 0.9], "")),
        (zones, "Зони, знайдені ШІ", ["AlphaEarth: 64 ознаки на піксель 10 м", "k-середніх, 5 груп, без підказок"],
         lambda dr, x: zone_legend(dr, x, 300)),
    ]
    wv = Writer("layers_scan")
    hold, sweep = 2.2, 1.2
    for si, (img, ti, sub, leg) in enumerate(stages):
        if si > 0:
            prev = stages[si - 1][0]
            for k in range(int(sweep * FPS)):
                u = ease(k / (sweep * FPS)); xcut = int(u * MAPW)
                m = Image.alpha_composite(prev, Image.new("RGBA", prev.size, (0, 0, 0, 0)))
                m.paste(img.crop((0, 0, xcut, MAPH)), (0, 0))
                m = Image.alpha_composite(m, pl)
                dr = ImageDraw.Draw(m, "RGBA")
                for g in range(14):                     # glowing scan line
                    dr.line((xcut - g, 0, xcut - g, MAPH), fill=(255, 255, 255, int(200 * (1 - g / 14) ** 2)))
                wv.add(map_frame(m, ti, sub, leg))
        m = Image.alpha_composite(img, pl)
        wv.add(map_frame(m, ti, sub, leg), int(hold * FPS))
    poster = map_frame(Image.alpha_composite(chm, pl), stages[1][1], stages[1][2], stages[1][3])
    wv.close(poster)

# ---------------------------------------------------------------- 3. NDVI seasons
MONTHS = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень", "вересень",
          "жовтень", "листопад", "грудень"]
def ndvi_video():
    valid = [m for m in range(1, 13) if np.isfinite(NDVI[m]).any()]
    mean = {m: float(np.nanmean(NDVI[m])) for m in valid}
    imgs = {m: Image.alpha_composite(paste_at(base_map(0.45), grid_img(NDVI[m], 20, "RdYlGn", 0.0, 0.95)), poly_layer((MAPW, MAPH)))
            for m in valid}
    # chart area
    cx0, cy0, cw, chh = 740, 250, 500, 330
    def chart(dr, upto):
        dr.line((cx0, cy0 + chh, cx0 + cw, cy0 + chh), fill=GREY, width=2)
        for v in (0, 0.5, 1.0):
            y = cy0 + chh - v * chh
            dr.line((cx0, y, cx0 + cw, y), fill=(200, 199, 191), width=1)
            label(dr, (cx0 - 10, y), f"{v:g}".replace(".", ","), 26, fill=GREY, b=False, anchor="rm")
        for m in range(1, 13):
            x = cx0 + (m - 0.5) / 12 * cw
            label(dr, (x, cy0 + chh + 10), MONTHS[m - 1][:3], 23, fill=GREY, b=False, anchor="ma")
        pts = [(cx0 + (m - 0.5) / 12 * cw, cy0 + chh - max(mean[m], 0) * chh) for m in valid if m <= upto]
        mm = [m for m in valid if m <= upto]
        for i in range(1, len(pts)):
            dash = mm[i] - mm[i - 1] > 1
            if dash:
                (xa, ya), (xb, yb) = pts[i - 1], pts[i]
                for s in np.arange(0, 1, 0.1):
                    dr.line((xa + (xb - xa) * s, ya + (yb - ya) * s, xa + (xb - xa) * (s + 0.05), ya + (yb - ya) * (s + 0.05)), fill=GREY, width=3)
            else:
                dr.line((pts[i - 1], pts[i]), fill=TEAL, width=5)
        for (x, y) in pts:
            dr.ellipse((x - 7, y - 7, x + 7, y + 7), fill=TEAL)
    wv = Writer("ndvi_seasons")
    seq = valid + [valid[0]]
    per, fade = 1.0, 0.45
    for i, m in enumerate(valid):
        nxt = valid[i + 1] if i + 1 < len(valid) else None
        for k in range(int(per * FPS)):
            t = k / FPS
            img = imgs[m]
            if nxt and t > per - fade:
                img = Image.blend(imgs[m], imgs[nxt], ease((t - (per - fade)) / fade))
            def leg(dr, x, m=m):
                colorbar(dr, x, 140, "RdYlGn", 0, 0.95, [0, 0.3, 0.6, 0.9], "NDVI", w=380)
                chart(dr, m)
                label(dr, (x, 620), f"{MONTHS[m - 1]} 2024 · середнє {mean[m]:.2f}".replace(".", ","), 32, fill=DARK)
                if m == 1:
                    label(dr, (x, 665), "лютий і грудень: безхмарних знімків немає", 27, fill=GREY, b=False)
            wv.add(map_frame(img, "Як парк «дихає»", None, leg))
    def leg(dr, x):
        colorbar(dr, x, 140, "RdYlGn", 0, 0.95, [0, 0.3, 0.6, 0.9], "NDVI", w=380); chart(dr, 12)
        label(dr, (x, 620), "пік – червень, 0,89", 32, fill=DARK)
    last = map_frame(imgs[6], "Як парк «дихає»", None, leg)
    wv.add(last, int(2.5 * FPS)); wv.close(last)

# ---------------------------------------------------------------- 4. AI zones reveal
def zones_video():
    rng = np.random.default_rng(7)
    z = ZONE; idx = np.argwhere(np.isfinite(z)); order = rng.permutation(len(idx))
    zi = zone_img(); base = base_map(0.2); pl = poly_layer((MAPW, MAPH))
    wv = Writer("ai_zones")
    wv.add(map_frame(Image.alpha_composite(base, pl), "ШІ шукає схожі ділянки", ["кожен піксель 10 м – 64 ознаки", "AlphaEarth, 2024"]), FPS)
    n = len(idx); steps = int(3.5 * FPS)
    mask = np.zeros(z.shape, bool)
    for s in range(steps):
        k1 = int(n * ease((s + 1) / steps)); k0 = int(n * ease(s / steps))
        for j in order[k0:k1]: mask[tuple(idx[j])] = True
        mimg = Image.fromarray((mask * 255).astype(np.uint8)).resize(zi.size, Image.NEAREST)
        zz = zi.copy(); a = np.array(zz.split()[3]) * (np.array(mimg) > 0); zz.putalpha(Image.fromarray(a.astype(np.uint8)))
        m = Image.alpha_composite(paste_at(base, zz), pl)
        al = s / steps
        wv.add(map_frame(m, "ШІ шукає схожі ділянки", ["кожен піксель 10 м – 64 ознаки", "AlphaEarth, 2024"],
                         lambda dr, x, al=al: zone_legend(dr, x, 300, alpha=ease(al * 1.3))))
    fin = map_frame(Image.alpha_composite(paste_at(base, zi), pl), "Результат: 5 зон", ["висоти – з карти висот крон"],
                    lambda dr, x: (zone_legend(dr, x, 300), label(dr, (x, 590), "А чи Б: які це породи?", 32, fill=OCHRE),
                                   label(dr, (x, 632), "перевірте в полі", 32, fill=OCHRE)))
    wv.add(fin, int(3 * FPS)); wv.close(fin)

# ---------------------------------------------------------------- 5. 3D canopy
def canopy3d_video(nfr=200, ex=2.0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from scipy.ndimage import zoom as nzoom
    Hh = nzoom(np.nan_to_num(CHM, nan=0.0), 2, order=1)          # 2.5 m for a smoother surface
    inside = nzoom(np.isfinite(CHM).astype(float), 2, order=1) > 0.5
    ny, nx = Hh.shape
    tex = np.asarray(ORTHO.resize((nx, ny), Image.BOX)).astype(float) / 255
    hs = LightSource(315, 35).hillshade(Hh, vert_exag=1.5, dx=2.5, dy=2.5)
    col = np.clip(tex * (0.45 + 0.85 * hs[..., None]) * 1.1, 0, 1)
    col = np.where(inside[..., None], col, col * 0.35 + 0.55)
    X, Y = np.meshgrid(np.arange(nx) * 2.5, (ny - 1 - np.arange(ny)) * 2.5)
    wv = Writer("canopy_3d")
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(np.array(BG) / 255)
    ax = fig.add_axes([-0.4, -0.55, 1.8, 2.05], projection="3d")
    ax.plot_surface(X, Y, Hh * ex, facecolors=col, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_axis_off(); ax.set_box_aspect((nx * 2.5, ny * 2.5, 25 * ex * 2.2)); ax.set_zlim(0, 25 * ex * 2.2)
    ax.set_facecolor(np.array(BG) / 255)
    poster = None
    for i in range(nfr):
        az = -60 + 360 * i / nfr; el = 34 - 6 * np.sin(2 * np.pi * i / nfr)
        ax.view_init(elev=el, azim=az)
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        dr = ImageDraw.Draw(im)
        label(dr, (W - 30, H - 30), f"вертикаль × {ex:g}", 28, fill=GREY, b=False, anchor="rd")
        wv.add(im)
        if i == min(12, nfr - 1): poster = im.copy()
    plt.close(fig); wv.close(poster)

if __name__ == "__main__":
    which = sys.argv[1:] or ["zoom", "layers", "ndvi", "zones", "3d"]
    for w in which:
        {"zoom": zoom_video, "layers": layers_video, "ndvi": ndvi_video, "zones": zones_video, "3d": canopy3d_video}[w]()
        print("done", w)
