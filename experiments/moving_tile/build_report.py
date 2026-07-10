# -*- coding: utf-8 -*-
"""Build the comparative HTML report (embeds stage frames as data URIs)."""
import os, base64, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
_ST = os.path.join(_HERE, "out", "stages")


def jpg_uri(png, q=78, w=900):
    img = cv2.imread(png)
    h0, w0 = img.shape[:2]
    if w0 > w:
        img = cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

# stage data: label, note, N, len, jit, ncc, fb
STAGES = [
    ("v0", "guided moving-tile baseline", 36, 97.6, 0.680, 0.670, None),
    ("v1", "+ NCC sub-pixel refine",       31, 85.5, 2.161, 0.809, None),
    ("v2", "+ forward/backward cull",       18, 61.8, 0.527, 0.890, 0.74),
    ("v3", "+ sub-pixel corner seeds",      22, 65.5, 0.749, 0.901, 0.96),
    ("v4", "+ quality-score drop",          21, 66.6, 0.725, 0.920, 0.96),
    ("v5", "+ affine pattern lock",         21, 66.6, 1.016, 0.914, 0.96),
    ("v6", "+ sharpest-frame anchor",       21, 66.6, 0.597, 0.912, 0.96),
    ("v7", "+ spread / quality select",     20, 69.1, 0.609, 0.920, 0.90),
]

def dirn(prev, cur, lower_better):
    if prev is None or cur is None:
        return "flat"
    if abs(cur - prev) < 1e-6:
        return "flat"
    better = (cur < prev) if lower_better else (cur > prev)
    return "good" if better else "bad"

rows = ""
prev = None
for s in STAGES:
    label, note, N, ln, jit, ncc, fb = s
    jc = dirn(prev[4] if prev else None, jit, True)
    nc = dirn(prev[5] if prev else None, ncc, False)
    fc = dirn(prev[6] if prev else None, fb, True)
    ncc_bar = int(ncc * 100)
    fb_txt = f"{fb:.2f}" if fb is not None else "&mdash;"
    rows += f"""<tr>
      <td class="stg">{label}</td>
      <td class="note">{note}</td>
      <td class="num">{N}</td>
      <td class="num">{ln:.0f}</td>
      <td class="num {jc}">{jit:.3f}</td>
      <td class="num {nc}"><span class="barwrap"><span class="bar" style="width:{ncc_bar}%"></span></span>{ncc:.3f}</td>
      <td class="num {fc}">{fb_txt}</td>
    </tr>\n"""
    prev = s

cards = ""
CARDNOTE = {
 "v0":"Raw guided tile. Crosses sit near the contrast edge but carry the learned-point offset; weak sky/water points still present.",
 "v1":"NCC snaps each point onto the full-res patch (ncc 0.67&rarr;0.81) but per-frame matching adds wobble (jit 2.16) &mdash; refinement alone is not enough.",
 "v2":"Backward pass rejects tracks whose forward/backward paths disagree. Wobble collapses (jit 0.53), lock climbs (0.89) &mdash; but half the points were junk and got culled.",
 "v3":"cornerSubPix + min-eigen gate seeds only real corners (no water/haze). More good points survive (N 18&rarr;22).",
 "v4":"Drop any track whose lock stays below 0.60. ncc &rarr; 0.920.",
 "v5":"Affine ECC adds rotation/scale DoF &mdash; but this is a pan-dominant plate, so the extra freedom wobbles (jit 0.73&rarr;1.02). A regression here.",
 "v6":"Anchoring the reference patch at each track's sharpest frame tames the affine wobble (jit 1.02&rarr;0.60). Sharpest-anchor rescues affine.",
 "v7":"Farthest-point spread keeps the 20 best-locked, evenly-spread points. Final: ncc 0.920, fb 0.90px, jit 0.61.",
}
for label, note, *_ in STAGES:
    uri = jpg_uri(os.path.join(_ST, f"{label}.png"))
    cards += f"""<figure class="card">
      <figcaption><span class="tag">{label}</span> {note}</figcaption>
      <img src="{uri}" alt="{label} zoomed tracks" loading="lazy" />
      <p class="cap">{CARDNOTE[label]}</p>
    </figure>\n"""

STYLE = """<title>Moving-tile track quality &mdash; v0 to v7</title>
<style>
:root{
  --bg:#0d1012; --panel:#151a1d; --panel2:#1b2226; --line:#283136;
  --ink:#d7dee0; --ink2:#9aa7ab; --faint:#6b787d;
  --brand:#38cdba; --good:#4fbf7b; --bad:#e0a24a;
  --shadow:0 1px 0 #ffffff08, 0 12px 30px -18px #000a;
  --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;
  --sans:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;
}
@media (prefers-color-scheme:light){
  :root{ --bg:#eef1f0; --panel:#ffffff; --panel2:#f4f6f6; --line:#dce2e2;
    --ink:#1b2427; --ink2:#566065; --faint:#8a969b; --brand:#0f9e8e;
    --good:#2f9d5b; --bad:#c07c1e; --shadow:0 1px 0 #fff, 0 10px 24px -18px #0004; }
}
:root[data-theme="dark"]{ --bg:#0d1012; --panel:#151a1d; --panel2:#1b2226; --line:#283136;
  --ink:#d7dee0; --ink2:#9aa7ab; --faint:#6b787d; --brand:#38cdba; --good:#4fbf7b; --bad:#e0a24a;
  --shadow:0 1px 0 #ffffff08, 0 12px 30px -18px #000a; }
:root[data-theme="light"]{ --bg:#eef1f0; --panel:#ffffff; --panel2:#f4f6f6; --line:#dce2e2;
  --ink:#1b2427; --ink2:#566065; --faint:#8a969b; --brand:#0f9e8e; --good:#2f9d5b; --bad:#c07c1e;
  --shadow:0 1px 0 #fff, 0 10px 24px -18px #0004; }

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.55;-webkit-font-smoothing:antialiased}
article{max-width:1000px;margin:0 auto;padding:clamp(20px,4vw,56px) clamp(16px,4vw,40px) 80px}
h1,h2{text-wrap:balance;letter-spacing:-.02em;line-height:1.1;margin:0}
section{margin-top:clamp(40px,7vw,72px)}
h2{font-size:clamp(1.25rem,2.4vw,1.6rem);padding-bottom:.5em;border-bottom:1px solid var(--line)}
.lede{color:var(--ink2);margin:.7em 0 1.4em;max-width:64ch}

/* hero */
.hero{border:1px solid var(--line);border-radius:14px;background:
  radial-gradient(120% 140% at 88% -10%, #38cdba14, transparent 60%),var(--panel);
  padding:clamp(22px,4vw,40px);box-shadow:var(--shadow)}
.kick{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--brand);margin-bottom:1em}
.hero h1{font-size:clamp(1.8rem,4.6vw,3rem)}
.sub{color:var(--ink2);max-width:60ch;margin:.8em 0 0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;
  margin:clamp(22px,4vw,34px) 0 0;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.stats>div{background:var(--panel);padding:16px 18px}
.stats dt{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.stats dd{margin:.35em 0 .1em;font-size:1.5rem;font-family:var(--mono);font-variant-numeric:tabular-nums}
.stats dd b{color:var(--brand)}
.stats .u{font-size:.9rem;color:var(--ink2)}
.stats .arr{color:var(--faint);font-size:1rem;padding:0 .1em}
.stats .d{font-size:.78rem;color:var(--ink2)}

/* table */
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{width:100%;border-collapse:collapse;font-size:.9rem;min-width:620px}
th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line)}
thead th{font-family:var(--mono);font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--faint);font-weight:600;background:var(--panel2)}
tbody tr:last-child td{border-bottom:0}
tbody tr:last-child{background:#38cdba0d}
.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
td.stg{font-family:var(--mono);color:var(--brand);font-weight:700}
td.note{color:var(--ink2)}
td.good{color:var(--good)} td.bad{color:var(--bad)}
td.good::before{content:"\\25B2 ";font-size:.6em;vertical-align:1px}
td.bad::before{content:"\\25BC ";font-size:.6em;vertical-align:1px}
.barwrap{display:inline-block;width:46px;height:5px;border-radius:3px;background:var(--line);
  margin-right:8px;vertical-align:middle;overflow:hidden}
.bar{display:block;height:100%;background:var(--brand)}
.foot{font-size:.76rem;color:var(--faint);margin-top:.9em}
b.good{color:var(--good)} b.bad{color:var(--bad)}

/* stage grid */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--panel);
  box-shadow:var(--shadow);display:flex;flex-direction:column}
.card figcaption{font-size:.86rem;padding:12px 14px 10px;color:var(--ink);border-bottom:1px solid var(--line)}
.card .tag{font-family:var(--mono);font-weight:700;color:var(--brand);margin-right:.5em}
.card img{width:100%;display:block;background:#000}
.card .cap{margin:0;padding:12px 14px 16px;font-size:.82rem;color:var(--ink2);line-height:1.5}

/* takeaways */
.takeaways ul{list-style:none;padding:0;margin:1.2em 0 0;display:grid;gap:12px}
.takeaways li{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--brand);
  border-radius:8px;padding:13px 16px;font-size:.92rem;color:var(--ink2)}
.takeaways li b{color:var(--ink)}
</style>
"""

html = STYLE + f"""<article>
  <header class="hero">
    <div class="kick">TAPNext&#43;&#43; &middot; moving-tile experiment &middot; isolated</div>
    <h1>Track quality, feature by feature</h1>
    <p class="sub">Seven refinements stacked on the guided moving-tile tracker, each measured
    end-to-end on shot <b>SH011</b> (1920&times;1080, 121f). Metrics need no ground truth &mdash;
    they read the footage itself.</p>
    <dl class="stats">
      <div><dt>NCC lock</dt><dd>0.67 <span class="arr">&rarr;</span> <b>0.92</b></dd><div class="d">+37% tighter to contrast</div></div>
      <div><dt>Fwd/bwd error</dt><dd><b>0.90<span class="u">px</span></b></dd><div class="d">sub-pixel, trustworthy</div></div>
      <div><dt>Jitter</dt><dd>0.68 <span class="arr">&rarr;</span> <b>0.61<span class="u">px</span></b></dd><div class="d">smoother, wobble removed</div></div>
      <div><dt>Points</dt><dd>36 <span class="arr">&rarr;</span> <b>20</b></dd><div class="d">junk culled, spread kept</div></div>
    </dl>
  </header>

  <section>
    <h2>Cumulative metrics</h2>
    <p class="lede">Each row adds one feature on top of the previous. <b class="good">Green</b> = that
    metric improved vs the row above, <b class="bad">amber</b> = regressed. Lower jitter &amp; fb are better; higher NCC is better.</p>
    <div class="tablewrap">
    <table>
      <thead><tr><th>stage</th><th>feature</th><th class="num">N</th><th class="num">len</th>
        <th class="num">jitter&nbsp;px</th><th class="num">NCC lock</th><th class="num">fb&nbsp;px</th></tr></thead>
      <tbody>
{rows}      </tbody>
    </table>
    </div>
    <p class="foot">N surviving tracks &middot; len mean visible frames &middot; jitter = mean per-frame
    acceleration (px) &middot; NCC lock = template correlation at the tracked point &middot;
    fb = forward/backward disagreement (px).</p>
  </section>

  <section>
    <h2>What each update did</h2>
    <p class="lede">Same six feature locations, same frame (61), 6&times; zoom. Green cross = tracked position.
    Watch points get culled or snap tighter to the edge as features stack.</p>
    <div class="grid">
{cards}    </div>
  </section>

  <section class="takeaways">
    <h2>Read on the results</h2>
    <ul>
      <li><b>FB-cull is the workhorse.</b> It, not raw NCC, produced the biggest quality jump &mdash; by deleting tracks that can't survive a round trip.</li>
      <li><b>NCC alone jitters.</b> Per-frame independent matching (v1) tightened lock but spiked wobble; it needs the consistency cull behind it.</li>
      <li><b>Affine is shot-dependent.</b> On this pan-only plate the extra rotation/scale freedom regressed (v5). Keep it off for translation-dominant shots, on for roll/zoom.</li>
      <li><b>Sharpest anchor matters.</b> Anchoring the pattern at each track's crispest frame (v6) recovered the smoothness affine cost.</li>
      <li><b>Fewer, better.</b> 36&rarr;20 is the point: the survivors are sub-pixel, consistent, and evenly spread &mdash; the tracks an artist keeps.</li>
    </ul>
  </section>
</article>
"""

with open(os.path.join(_HERE, "out", "report.html"), "w", encoding="utf-8") as f:
    f.write(html)
print("wrote report.html", len(html))
