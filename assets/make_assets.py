"""Generates the pixel-art README assets (SVG) and PNG previews."""
import sys
from PIL import Image
OUT, PREV = sys.argv[1], sys.argv[2]  # usage: python assets/make_assets.py assets <preview_dir>
PAL = {"k": "#2b2f4a", "c": "#2ec4ff", "C": "#8fe0ff", "g": "#3ddc84", "y": "#ffd23f", "o": "#ff8c42", "r": "#ff4d6d", "R": "#ff8fa3",
       "p": "#9b5de5", "w": "#eef0f6", "W": "#ffffff", "s": "#aab2c0", "b": "#3a86ff", "B": "#7fb0ff", "d": "#5b6178", "t": "#14b8a6"}
FONT = {
"A":"01110 10001 10001 11111 10001 10001 10001","B":"11110 10001 10001 11110 10001 10001 11110","C":"01111 10000 10000 10000 10000 10000 01111",
"D":"11110 10001 10001 10001 10001 10001 11110","E":"11111 10000 10000 11110 10000 10000 11111","F":"11111 10000 10000 11110 10000 10000 10000",
"G":"01111 10000 10000 10011 10001 10001 01111","H":"10001 10001 10001 11111 10001 10001 10001","I":"11111 00100 00100 00100 00100 00100 11111",
"J":"00111 00010 00010 00010 00010 10010 01100","K":"10001 10010 10100 11000 10100 10010 10001","L":"10000 10000 10000 10000 10000 10000 11111",
"M":"10001 11011 10101 10101 10001 10001 10001","N":"10001 11001 10101 10011 10001 10001 10001","O":"01110 10001 10001 10001 10001 10001 01110",
"P":"11110 10001 10001 11110 10000 10000 10000","Q":"01110 10001 10001 10001 10101 10010 01101","R":"11110 10001 10001 11110 10100 10010 10001",
"S":"01111 10000 10000 01110 00001 00001 11110","T":"11111 00100 00100 00100 00100 00100 00100","U":"10001 10001 10001 10001 10001 10001 01110",
"V":"10001 10001 10001 10001 10001 01010 00100","W":"10001 10001 10001 10101 10101 11011 10001","X":"10001 10001 01010 00100 01010 10001 10001",
"Y":"10001 10001 01010 00100 00100 00100 00100","Z":"11111 00001 00010 00100 01000 10000 11111",
"0":"01110 10001 10011 10101 11001 10001 01110","1":"00100 01100 00100 00100 00100 00100 01110","2":"01110 10001 00001 00110 01000 10000 11111",
"3":"11110 00001 00001 01110 00001 00001 11110","4":"00010 00110 01010 10010 11111 00010 00010","5":"11111 10000 11110 00001 00001 10001 01110",
"6":"00110 01000 10000 11110 10001 10001 01110","7":"11111 00001 00010 00100 01000 01000 01000","8":"01110 10001 10001 01110 10001 10001 01110",
"9":"01110 10001 10001 01111 00001 00010 01100",".":"00000 00000 00000 00000 00000 01100 01100","%":"11001 11010 00010 00100 01000 01011 10011",
"-":"00000 00000 00000 11111 00000 00000 00000"," ":"00000 00000 00000 00000 00000 00000 00000","+":"00000 00100 00100 11111 00100 00100 00000",
"(":"00010 00100 01000 01000 01000 00100 00010",")":"01000 00100 00010 00010 00010 00100 01000","/":"00001 00001 00010 00100 01000 10000 10000",
":":"00000 01100 01100 00000 01100 01100 00000","_":"00000 00000 00000 00000 00000 00000 11111",
}
class Canvas:
    def __init__(s, w, h, bg=None):
        s.w, s.h, s.bg, s.px = w, h, bg, {}
    def set(s, x, y, col):
        if 0 <= x < s.w and 0 <= y < s.h and col: s.px[(x, y)] = col
    def rect(s, x, y, w, h, col):
        for j in range(h):
            for i in range(w): s.set(x + i, y + j, col)
    def art(s, x, y, rows, scale=1):
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch != ".": s.rect(x + i * scale, y + j * scale, scale, scale, PAL[ch])
    def text(s, x, y, txt, col, scale=1, gap=1):
        for ch in txt:
            g = FONT[ch].split()
            for j, row in enumerate(g):
                for i, b in enumerate(row):
                    if b == "1": s.rect(x + i * scale, y + j * scale, scale, scale, col)
            x += (5 + gap) * scale
        return x
    def save(s, name, scale, radius=0):
        W, H = s.w * scale, s.h * scale
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s.w} {s.h}" width="{W}" height="{H}" shape-rendering="crispEdges">']
        if s.bg: out.append(f'<rect width="{s.w}" height="{s.h}" rx="{radius}" fill="{s.bg}"/>')
        for y in range(s.h):  # run-length encode rows
            x = 0
            while x < s.w:
                c = s.px.get((x, y))
                if c is None: x += 1; continue
                x0 = x
                while x < s.w and s.px.get((x, y)) == c: x += 1
                out.append(f'<rect x="{x0}" y="{y}" width="{x - x0}" height="1" fill="{c}"/>')
        out.append("</svg>")
        open(f"{OUT}/{name}.svg", "w").write("\n".join(out) + "\n")
        for tag, back in (("dark", "#0d1117"), ("light", "#ffffff")):
            im = Image.new("RGB", (s.w, s.h), s.bg or back)
            for (x, y), c in s.px.items(): im.putpixel((x, y), tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)))
            im.resize((W, H), Image.NEAREST).save(f"{PREV}/{name.replace('/', '_')}_{tag}.png")

ICONS = {
"lidar": """
................
c......y......c.
.c.....y.....c..
..c....y....c...
...c.......c....
................
....kkkkkkkk....
...kWwwwwwwsk...
...kBbbbbbbbk...
...kWwwwwwwsk...
...kBbbbbbbbk...
...kWwwwwwwsk...
....kkkkkkkk....
.....kssdk......
...kkkkkkkkkk...
................""",
"car": """
................
................
.......cc.......
......kkkk......
....kkkkkkkk....
...kCcckkCcck...
..kCccckkCccck..
.kkkkkkkkkkkkkk.
kRRRRRRRRRRRRRRk
kyrrrrrrrrrrrrok
krrrrrrrrrrrrrrk
kkkkkrrrrrrkkkkk
.kdsdkkkkkkdsdk.
.kdddk....kdddk.
..kkk......kkk..
................""",
"clock": """
................
....kkkkkkkk....
...kWWWWWWWWk...
..kWWwwkwwwwwk..
.kWWwwwkwwwwwwk.
.kWwwwwkwwwwwsk.
.kWwwwwkwwwwwsk.
.kWwwwwrrrrwwsk.
.kWwwwwwwwwwwsk.
.kwwwwwwwwwwssk.
.kwwwwwwwwwwssk.
..kwwwwwwwsssk..
...kwwssssssk...
....kkkkkkkk....
................
................""",
"layers": """
................
.......kk.......
.....kkCCkk.....
...kkCCccccckk..
.kkCcccccccccck.
...kkcccccckk...
.kk..kkcckk..kk.
.kkgg..kk..ggkk.
...kkgg..ggkk...
.kk..kkggkk..kk.
.kkyy..kk..yykk.
...kkyy..yykk...
.....kkyykk.....
.......kk.......
................
................""",
"gear": """
................
......kkkk......
..kk..kWsk..kk..
.kWsk.kssk.kssk.
.ksssksssskssk..
..kssssssssssk..
.kkWsskkkksssdk.
kWsssk....kssdk.
kssssk....kssdk.
.kkssskkkkssdkk.
..kssssssssddk..
.ksssksssdkddk..
.kssk.ksdk.kddk.
..kk..kddk..kk..
......kkkk......
................""",
"disk": """
................
...kkkkkkkkkk...
..kBBBBBBBBbbk..
.kBbbbbbbbbbbbk.
.kkbbbbbbbbbbkk.
.kbkkkkkkkkkkbk.
.kbbbbbbbbbbbbk.
.kkbbbbbbbbbbkk.
.kckkkkkkkkkkck.
.kCcccccccccgck.
.kkcccccccccckk.
.ktkkkkkkkkkktk.
.kttttttttttgtk.
..kttttttttttk..
...kkkkkkkkkk...
................""",
"chart": """
................
.k..............
.k..........kkk.
.k..........kgk.
.k......kkk.kgk.
.k......kyk.kgk.
.k......kyk.kgk.
.k..kkk.kyk.kgk.
.k..kok.kyk.kgk.
.k..kok.kyk.kgk.
.k..kok.kyk.kgk.
.k..kok.kyk.kgk.
.k..kkk.kkk.kkk.
.kkkkkkkkkkkkkk.
................
................""",
"book": """
................
.kkkkkk..kkkkkk.
.kWwwwwkkWwwwwk.
.kwddwwkkwwddwk.
.kwwwwwkkwwwwwk.
.kwdddwkkwdddwk.
.kwwwwwkkwwwwwk.
.kwddwwkkwwddwk.
.kwwwwwkkwwwwwk.
.kwdddwkkwdddwk.
.kwwwwwkkwwwwwk.
.kpppppkkpppppk.
.kkkkkkkkkkkkkk.
.......kk.......
................
................""",
"rocket": """
................
.......kk.......
......kWwk......
.....kWwwsk.....
.....kwCcsk.....
....kWwccwsk....
....kWwwwwsk....
....kWwwwwsk....
...kkWwwwwskk...
..krkWwwwwskrk..
..krrkwwwskrrk..
..kkkkkkkkkkkk..
......kyyk......
......koyk......
.......ok.......
................""",
"flag": """
................
..kk............
..kkRRRR........
..kkrrrrrrrr....
..kkrrrrrrrrrr..
..kkrrrrrrrrrr..
..kkrrrrrrrr....
..kkrrrr........
..kk............
..kk............
..kk............
..kk............
..kk............
.kkkk...........
kkkkkk..........
................""",
"scroll": """
................
..kkkkkkkkkkk...
.kyyyyyyyyyyyk..
.kykkkkkkkkkyk..
..kWwwwwwwwwk...
..kwdddddddwk...
..kwwwwwwwwwk...
..kwdddddwwwk...
..kwwwwwwwwwk...
..kwdddddddwk...
..kwwwwwwwwwk...
.kkkkkkkkkkkkk..
.kyyyyyyyyyyyk..
..kkkkkkkkkkk...
................
................""",
"note": """
................
.......kk.......
......kyyk......
......kyyk......
.....kyyyyk.....
.....kykkyk.....
....kyykkyyk....
....kyykkyyk....
...kyyykkyyyk...
...kyyykkyyyk...
..kyyyyyyyyyyk..
..kyyyykkyyyyk..
.kyyyyykkyyyyyk.
.kyyyyyyyyyyyyk.
..kkkkkkkkkkkk..
................""",
"heart": """
................
................
..kkkk...kkkk...
.kRRrrk.krrrrk..
kRRrrrrkrrrrrrk.
kRrrrrrrrrrrrrk.
krrrrrrrrrrrrrk.
krrrrrrrrrrrrrk.
.krrrrrrrrrrrk..
..krrrrrrrrrk...
...krrrrrrrk....
....krrrrrk.....
.....krrrk......
......krk.......
.......k........
................""",
"map": """
................
................
.kkkk....kkkkk..
.kgggkkkkgggggk.
.kgggkcccgggrgk.
.kggkccckgggggk.
.kgkccckggyyggk.
.kgkcckgggyyggk.
.kggkcckggggggk.
.kgggkccckggggk.
.kggggkccckgggk.
.kgggggkccckggk.
.kkkkkkkkkkkkkk.
................
................
................""",
}
for name, art in ICONS.items():
    rows = [r.ljust(16, ".")[:16] for r in art.strip("\n").split("\n")]
    assert len(rows) == 16, (name, len(rows))
    c = Canvas(16, 16); c.art(0, 0, rows); c.save(f"icons/{name}", 4)

def turbo(t):  # near (0) red -> far (1) blue
    stops = ["#ff4d6d", "#ff8c42", "#ffd23f", "#3ddc84", "#2ec4ff", "#3a86ff", "#9b5de5"]
    return stops[min(len(stops) - 1, int(t * len(stops)))]


# ---------------------------------------------------------------- logo
car = [r.ljust(16, ".")[:16] for r in ICONS["car"].strip("\n").split("\n")]
c = Canvas(56, 16)
c.art(0, 0, car)
for k, (tx, ty) in enumerate([(55, 1), (55, 6), (55, 11)]):
    ox, oy = 8, 1
    n = tx - ox
    for i in range(16, n, 3):
        c.set(ox + i, oy + round((ty - oy) * i / n), turbo((i - 16) / (n - 16)))
c.save("logo", 6)

# ---------------------------------------------------------------- results chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
ROWS = [("MVSAnywhere", 10.29, "#9aa3b2"), ("DriveMVS · no prompt", 7.16, "#9b5de5"), ("DriveMVS · source only", 4.90, "#3a86ff"),
        ("DriveMVS · current only", 3.67, "#2ec4ff"), ("DriveMVS · current + source", 3.32, "#3ddc84")]
for tag, fg, sub, grid in (("light", "#1f2328", "#59636e", "#d1d9e0"), ("dark", "#f0f6fc", "#9198a1", "#3d444d")):
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    names = [r[0] for r in ROWS][::-1]; vals = [r[1] for r in ROWS][::-1]; cols = [r[2] for r in ROWS][::-1]
    ax.set_facecolor("none")
    bars = ax.barh(names, vals, color=cols, height=0.58)
    for b, v in zip(bars, vals):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", ha="left", color=fg, fontsize=11, fontweight="bold")
        if v < 10:
            ax.text(v + 1.3, b.get_y() + b.get_height() / 2, f"−{100 * (1 - v / 10.29):.0f}%", va="center", ha="left", color=sub, fontsize=9.5)
    ax.set_xlim(0, 12); ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=fg); ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=sub)
    ax.xaxis.grid(True, color=grid, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title("KITTI, zero-shot · AbsRel % (lower is better)", loc="left", color=fg, fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/results_kitti_{tag}.svg", transparent=True)
    fig.savefig(f"{PREV}/results_kitti_{tag}.png", dpi=110, facecolor="#0d1117" if tag == "dark" else "#ffffff")
    plt.close(fig)
