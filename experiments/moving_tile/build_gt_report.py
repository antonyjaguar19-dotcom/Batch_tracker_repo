# -*- coding: utf-8 -*-
"""Build the ground-truth deviation report (manual vs tracker)."""
import os, base64, cv2

_HERE = os.path.dirname(os.path.abspath(__file__))

def jpg_uri(png, q=80, w=1280):
    img = cv2.imread(png)
    h0, w0 = img.shape[:2]
    if w0 > w:
        img = cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# per-track mean deviation (px vs manual)
TRK = [
 ("01",2.95,0.36,0.72),("02",3.35,0.49,1.04),("03",7.09,0.78,1.24),
 ("04",1.57,0.53,0.81),("05",2.49,1.05,1.11),("06",3.28,0.69,0.76),
 ("07",3.56,1.15,1.01),("08",6.74,0.96,0.35),("09",6.71,0.59,0.63),
 ("10",4.45,1.22,0.89),("11",3.27,1.81,0.45),("12",3.67,0.95,0.59),
 ("13",7.62,11.43,2.32),("14",3.51,10.10,4.29),("15",5.07,1.70,2.26),
 ("17",5.00,2.17,1.06),("18",11.88,5.13,1.43),
]
OVERALL = {"BL":(4.88,3.77,9.50,27.44),"MT":(2.46,0.78,6.94,25.95),"CB":(1.30,0.96,2.86,5.91)}

AXMAX = 12.0
def bar(v, cls):
    w = min(100.0, v / AXMAX * 100.0)
    cap = " over" if v > AXMAX else ""
    return f'<div class="bar {cls}{cap}" style="width:{w:.1f}%"><span>{v:.2f}</span></div>'

rows = ""
for nm, bl, mt, cb in TRK:
    rows += f"""<div class="trkrow">
      <div class="tname">{nm}</div>
      <div class="bars">
        {bar(bl,'bl')}{bar(mt,'mt')}{bar(cb,'cb')}
      </div>
    </div>\n"""

ov = jpg_uri(os.path.join(_HERE, "out", "gt", "_ov90.png"))

def tile(tag, label, sub, cls):
    m = OVERALL[tag]
    return f"""<div class="{cls}"><dt>{label}</dt>
      <dd>{m[0]:.2f}<span class="u">px</span></dd>
      <div class="d">{sub}</div>
      <div class="mini">med {m[1]:.2f} &middot; p90 {m[2]:.2f} &middot; max {m[3]:.1f}</div></div>"""

STYLE = """<title>Ground-truth deviation &mdash; manual vs tracker</title>
<style>
:root{ --bg:#0d1012;--panel:#151a1d;--panel2:#1b2226;--line:#283136;--ink:#d7dee0;--ink2:#9aa7ab;
 --faint:#6b787d;--brand:#38cdba;--good:#4fbf7b;--bad:#e0a24a;--bl:#7d8a90;--mt:#e0a24a;--cb:#38cdba;
 --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;--sans:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;}
@media (prefers-color-scheme:light){:root{--bg:#eef1f0;--panel:#fff;--panel2:#f4f6f6;--line:#dce2e2;
 --ink:#1b2427;--ink2:#566065;--faint:#8a969b;--brand:#0f9e8e;--good:#2f9d5b;--bad:#c07c1e;--bl:#9aa5aa;--mt:#c07c1e;--cb:#0f9e8e;}}
:root[data-theme="dark"]{--bg:#0d1012;--panel:#151a1d;--panel2:#1b2226;--line:#283136;--ink:#d7dee0;--ink2:#9aa7ab;--faint:#6b787d;--brand:#38cdba;--good:#4fbf7b;--bad:#e0a24a;--bl:#7d8a90;--mt:#e0a24a;--cb:#38cdba;}
:root[data-theme="light"]{--bg:#eef1f0;--panel:#fff;--panel2:#f4f6f6;--line:#dce2e2;--ink:#1b2427;--ink2:#566065;--faint:#8a969b;--brand:#0f9e8e;--good:#2f9d5b;--bad:#c07c1e;--bl:#9aa5aa;--mt:#c07c1e;--cb:#0f9e8e;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased}
article{max-width:1000px;margin:0 auto;padding:clamp(20px,4vw,56px) clamp(16px,4vw,40px) 80px}
h1,h2{text-wrap:balance;letter-spacing:-.02em;line-height:1.12;margin:0}
section{margin-top:clamp(38px,6vw,64px)}
h2{font-size:clamp(1.2rem,2.4vw,1.55rem);padding-bottom:.5em;border-bottom:1px solid var(--line)}
.lede{color:var(--ink2);margin:.7em 0 1.4em;max-width:66ch}
.hero{border:1px solid var(--line);border-radius:14px;background:radial-gradient(120% 140% at 85% -10%,#38cdba14,transparent 60%),var(--panel);padding:clamp(22px,4vw,40px)}
.kick{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--brand);margin-bottom:1em}
.hero h1{font-size:clamp(1.8rem,4.6vw,2.9rem)}
.sub{color:var(--ink2);max-width:62ch;margin:.8em 0 0}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;margin-top:clamp(22px,4vw,32px);background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.tiles>div{background:var(--panel);padding:16px 18px}
.tiles dt{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.tiles dd{margin:.3em 0 .1em;font-size:clamp(1.6rem,4vw,2.3rem);font-family:var(--mono);font-variant-numeric:tabular-nums}
.tiles .u{font-size:.85rem;color:var(--ink2);margin-left:2px}
.tiles .d{font-size:.8rem;color:var(--ink2)}
.tiles .mini{font-family:var(--mono);font-size:.7rem;color:var(--faint);margin-top:.5em}
.tiles .win dd{color:var(--brand)} .tiles .win{outline:1px solid var(--brand);outline-offset:-1px}
@media(max-width:620px){.tiles{grid-template-columns:1fr}}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:.82rem;color:var(--ink2);margin:0 0 14px}
.legend b{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.chart{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:16px 18px}
.trkrow{display:grid;grid-template-columns:34px 1fr;gap:12px;align-items:center;padding:5px 0}
.tname{font-family:var(--mono);color:var(--ink2);font-size:.8rem;text-align:right}
.bars{display:flex;flex-direction:column;gap:3px}
.bar{height:15px;border-radius:3px;position:relative;min-width:30px;transition:width .3s}
.bar span{position:absolute;right:6px;top:0;line-height:15px;font-family:var(--mono);font-size:.66rem;color:#0009}
.bar.cb span,.bar.bl span{color:#fff9}
.bar.bl{background:var(--bl)} .bar.mt{background:var(--mt)} .bar.cb{background:var(--cb)}
.bar.over{background-image:repeating-linear-gradient(45deg,#0000,#0000 5px,#fff3 5px,#fff3 10px)}
.axis{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.66rem;color:var(--faint);margin:8px 0 0;padding-left:46px}
figure{margin:0}
.card{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--panel)}
.card img{width:100%;display:block;background:#000}
.card .cap{margin:0;padding:11px 14px;font-size:.82rem;color:var(--ink2)}
.verdict{border:1px solid var(--brand);border-radius:12px;background:#38cdba0d;padding:clamp(18px,3vw,26px)}
.verdict h2{border:0;padding:0}
.verdict ul{list-style:none;padding:0;margin:1em 0 0;display:grid;gap:10px}
.verdict li{padding-left:22px;position:relative;color:var(--ink2)}
.verdict li::before{content:"\\2713";position:absolute;left:0;color:var(--brand);font-weight:700}
.verdict li b{color:var(--ink)}
.note{font-size:.78rem;color:var(--faint);margin-top:1em}
</style>
"""

html = STYLE + f"""<article>
  <header class="hero">
    <div class="kick">Ground truth &middot; Shot_01 &middot; 3840&times;2160 &middot; 180f &middot; 17 manual tracks</div>
    <h1>How close does it get to a human?</h1>
    <p class="sub">Every method seeded at the artist's own start points, then scored by mean
    euclidean deviation from the manual track &mdash; in native 4K pixels. Lower is better.</p>
    <dl class="tiles">
      {tile('BL','Baseline (256 whole-frame)','current GPU fallback','')}
      {tile('MT','Moving-tile','native crop, no refine','')}
      {tile('CB','Winning combo','tile + NCC sub-pixel','win')}
    </dl>
  </header>

  <section>
    <h2>Per-track mean deviation</h2>
    <p class="lede">17 tracks. The combo is not just lower on average &mdash; it removes the
    moving-tile's two blow-ups (trk 13, 14) and the baseline's big drifts (03, 08, 09, 18).</p>
    <div class="legend">
      <span><b style="background:var(--bl)"></b>Baseline</span>
      <span><b style="background:var(--mt)"></b>Moving-tile</span>
      <span><b style="background:var(--cb)"></b>Combo</span>
      <span style="color:var(--faint)">hatched = off the 12px scale</span>
    </div>
    <div class="chart">
{rows}    </div>
    <div class="axis"><span>0</span><span>3</span><span>6</span><span>9</span><span>12&nbsp;px</span></div>
  </section>

  <section>
    <h2>Why the combo wins: the tails</h2>
    <p class="lede">Averages hide the risk. A single 26px drift ruins a camera solve. The combo's
    worst frame across all 17 tracks is <b>5.9px</b>; the others spike past 25px.</p>
    <div class="chart">
      <div class="trkrow"><div class="tname">p90</div><div class="bars">
        {bar(9.50,'bl')}{bar(6.94,'mt')}{bar(2.86,'cb')}</div></div>
      <div class="trkrow"><div class="tname">max</div><div class="bars">
        {bar(27.44,'bl')}{bar(25.95,'mt')}{bar(5.91,'cb')}</div></div>
    </div>
    <p class="note">p90 = 90th-percentile frame error &middot; max = worst single frame, all tracks pooled.
    Scale capped at 12px; hatched bars run past it.</p>
  </section>

  <section>
    <h2>Overlay &mdash; frame 91</h2>
    <figure class="card">
      <img src="{ov}" alt="manual vs combo overlay" />
      <p class="cap">RED circle = manual &middot; BLUE = baseline &middot; GREEN = combo. Green sits on
      the red circles across the timer, tube rack and form; blue drifts off.</p>
    </figure>
  </section>

  <section class="verdict">
    <h2>Verdict &mdash; wire the combo</h2>
    <p class="lede" style="margin-top:.8em">Against a real human track on a 4K plate:</p>
    <ul>
      <li><b>3.75&times; closer than baseline</b> &mdash; 1.30px vs 4.88px mean.</li>
      <li><b>Consistent</b> &mdash; p90 2.86px, worst frame 5.9px. No solve-breaking drifts.</li>
      <li><b>Refine fixes the tile's failures</b> &mdash; trk 13: 11.4&rarr;2.3px, trk 18: 5.1&rarr;1.4px.</li>
      <li><b>Sub-pixel where it matters</b> &mdash; median under 1px on the strong features.</li>
    </ul>
    <p class="note">Combo = guided moving-tile + NCC translation refine, anchor at each track's
    sharpest frame. Affine stays OFF (regressed on pan-only plates). FB-cull / spread-select are
    the culling layers on top &mdash; they trim count, not accuracy of survivors.</p>
  </section>
</article>
"""

open(os.path.join(_HERE, "out", "gt_report.html"), "w", encoding="utf-8").write(html)
print("wrote gt_report.html", len(html))
