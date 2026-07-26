from __future__ import annotations

import html
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from clip_model import encode_texts, load_clip
from index_io import load_index
from nero import NEGATION_MARKERS, route
from retrieval import topk_search
from utils import resolve_device

# --------------------------------------------------------------------------- #
# Server state — model + index are loaded once and shared across requests.
# CLIP inference is serialized with a lock (MPS/CUDA contexts are not
# guaranteed thread-safe under ThreadingHTTPServer).
# --------------------------------------------------------------------------- #


class _State:
    model = None
    processor = None
    device = "cpu"
    images_dir = Path(".")
    image_embeddings = None
    image_names: list[str] = []
    name_to_idx: dict[str, int] = {}
    lock = threading.Lock()


# Reproduced on this machine (HF CLIP ViT-B/32, held-out COCO MCQ-Neg, n=1775).
# CLIP baseline matches the paper's 39.66 exactly; NeRo 54.99 essentially
# matches the paper's open_clip result of 54.70.
NERO_REPRO = {"clip": 39.66, "nero": 54.99, "n": 1775}


# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #

PAGE_CSS = """
:root{
  --bg:#fbfbfd; --panel:#fff; --panel-2:#f4f5f8; --line:#e4e5ec; --line-2:#eceef3;
  --tx:#14141b; --tx-2:#5a5c6b; --tx-3:#8b8d9c;
  --acc:#4f46e5; --acc-2:#6366f1; --acc-soft:#eef1ff; --acc-line:#c9cffc;
  --pos:#0d8a4f; --pos-soft:#e7f6ee; --pos-line:#a9dfc4;
  --neg:#d4342a; --neg-soft:#fdedeb; --neg-line:#f3bfb9;
  --shadow:0 1px 2px rgba(18,18,32,.05), 0 10px 28px -14px rgba(18,18,32,.22);
  --shadow-sm:0 1px 2px rgba(18,18,32,.06);
  --r:14px; --r-sm:9px;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0a0a0e; --panel:#131318; --panel-2:#1b1b22; --line:#282830; --line-2:#20202a;
    --tx:#edeef4; --tx-2:#a3a5b4; --tx-3:#72747f;
    --acc:#8f8cff; --acc-2:#a5a2ff; --acc-soft:#1a1a35; --acc-line:#37356a;
    --pos:#4ade80; --pos-soft:#10261a; --pos-line:#1f4d33;
    --neg:#f87171; --neg-soft:#291313; --neg-line:#5c2626;
    --shadow:0 1px 2px rgba(0,0,0,.5), 0 10px 28px -14px rgba(0,0,0,.8);
    --shadow-sm:0 1px 2px rgba(0,0,0,.5);
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--tx); line-height:1.62;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Apple SD Gothic Neo",
              "Pretendard","Noto Sans KR",system-ui,sans-serif;
  font-size:15.5px; -webkit-font-smoothing:antialiased;
}
.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,monospace}

/* ---------- layout ---------- */
.shell{max-width:1080px; margin-inline:auto; padding:0 22px 90px}
.topbar{
  position:sticky; top:0; z-index:20; backdrop-filter:saturate(180%) blur(14px);
  background:color-mix(in srgb, var(--bg) 82%, transparent);
  border-bottom:1px solid var(--line);
}
.topbar .inner{max-width:1080px; margin-inline:auto; padding:11px 22px;
  display:flex; align-items:center; gap:18px; flex-wrap:wrap}
.brand{display:flex; align-items:center; gap:9px; font-weight:680; letter-spacing:-.015em;
  font-size:.95rem; color:var(--tx); text-decoration:none; white-space:nowrap}
.brand .dot{width:19px; height:19px; border-radius:6px; flex:none;
  background:linear-gradient(135deg,var(--acc),#c084fc)}
nav{display:flex; gap:3px; flex-wrap:wrap; margin-left:auto}
nav a{
  padding:6px 13px; border-radius:999px; font-size:.855rem; font-weight:520;
  color:var(--tx-2); text-decoration:none; transition:background .14s,color .14s;
  display:flex; align-items:center; gap:7px; white-space:nowrap;
}
nav a:hover{background:var(--panel-2); color:var(--tx)}
nav a.active{background:var(--acc); color:#fff}
nav a .n{
  font-size:.68rem; font-weight:700; width:16px; height:16px; border-radius:5px;
  display:grid; place-items:center; background:var(--panel-2); color:var(--tx-3);
}
nav a.active .n{background:rgba(255,255,255,.24); color:#fff}

/* ---------- typography ---------- */
h1{font-size:1.92rem; line-height:1.24; letter-spacing:-.032em; font-weight:730; margin:34px 0 10px}
h2{font-size:1.2rem; letter-spacing:-.022em; font-weight:670; margin:46px 0 12px;
   display:flex; align-items:center; gap:10px}
h2 .idx{font-size:.72rem; font-weight:700; color:var(--acc); background:var(--acc-soft);
   border:1px solid var(--acc-line); border-radius:6px; padding:2px 7px; letter-spacing:0}
h3{font-size:1rem; font-weight:640; margin:26px 0 8px; letter-spacing:-.014em}
p{margin:12px 0}
.lede{font-size:1.06rem; color:var(--tx-2); max-width:66ch; margin-bottom:6px}
.sub{color:var(--tx-3); font-size:.87rem}
.eyebrow{font-size:.74rem; font-weight:700; letter-spacing:.10em; text-transform:uppercase;
  color:var(--acc); margin-top:34px}
a{color:var(--acc); text-decoration:none}
a:hover{text-decoration:underline}
b,strong{font-weight:640}
hr{border:0; border-top:1px solid var(--line); margin:44px 0}
ul{padding-left:20px; margin:12px 0}
li{margin:7px 0}

/* ---------- surfaces ---------- */
.card{background:var(--panel); border:1px solid var(--line); border-radius:var(--r);
  padding:20px 22px; box-shadow:var(--shadow-sm); margin:18px 0}
.card.pad-lg{padding:26px 28px}
.formula{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.885rem;
  background:var(--panel-2); border:1px solid var(--line-2); border-left:3px solid var(--acc);
  border-radius:var(--r-sm); padding:14px 16px; margin:16px 0; overflow-x:auto;
  line-height:1.85; white-space:pre-wrap;
}
.formula .cm{color:var(--tx-3)}
.note{border-left:3px solid var(--acc); background:var(--acc-soft); border-radius:var(--r-sm);
  padding:13px 16px; margin:18px 0; font-size:.92rem}
.note.bad{border-left-color:var(--neg); background:var(--neg-soft)}
.note.good{border-left-color:var(--pos); background:var(--pos-soft)}
.note .hd{font-weight:680; margin-bottom:2px}

/* two-column feature grid */
.cols{display:grid; grid-template-columns:repeat(auto-fit,minmax(258px,1fr)); gap:14px; margin:18px 0}
.cols .c{background:var(--panel); border:1px solid var(--line); border-radius:var(--r);
  padding:17px 19px; box-shadow:var(--shadow-sm)}
.cols .c h4{margin:0 0 6px; font-size:.95rem; font-weight:660; display:flex; gap:8px; align-items:center}
.cols .c p{margin:0; font-size:.895rem; color:var(--tx-2)}
.ico{width:24px; height:24px; border-radius:7px; display:grid; place-items:center; flex:none;
  background:var(--acc-soft); color:var(--acc); font-size:.8rem; border:1px solid var(--acc-line)}

/* steps */
.steps{counter-reset:s; margin:18px 0}
.steps .s{position:relative; padding:0 0 20px 42px; border-left:2px solid var(--line); margin-left:13px}
.steps .s:last-child{border-left-color:transparent; padding-bottom:0}
.steps .s::before{
  counter-increment:s; content:counter(s); position:absolute; left:-14px; top:-2px;
  width:26px; height:26px; border-radius:9px; display:grid; place-items:center;
  background:var(--acc); color:#fff; font-size:.78rem; font-weight:700;
}
.steps .s h4{margin:0 0 4px; font-size:.97rem; font-weight:645}
.steps .s p{margin:0; color:var(--tx-2); font-size:.9rem}

/* ---------- search ---------- */
form.search{display:flex; gap:9px; flex-wrap:wrap; margin:20px 0 12px}
.field{flex:1; min-width:250px; position:relative; display:flex; align-items:center}
.field svg{position:absolute; left:14px; width:16px; height:16px; color:var(--tx-3)}
input[type=text]{
  width:100%; padding:13px 15px 13px 40px; font-size:.98rem; font-family:inherit;
  border:1px solid var(--line); border-radius:11px; background:var(--panel); color:inherit;
  box-shadow:var(--shadow-sm); transition:border-color .14s, box-shadow .14s;
}
.field.plain input[type=text]{padding-left:15px}
input[type=text]:focus{outline:0; border-color:var(--acc);
  box-shadow:0 0 0 3.5px color-mix(in srgb,var(--acc) 17%,transparent)}
input[type=number]{width:78px; padding:13px 10px; font-size:.98rem; font-family:inherit;
  border:1px solid var(--line); border-radius:11px; background:var(--panel); color:inherit; text-align:center}
input[type=number]:focus{outline:0; border-color:var(--acc)}
button{padding:13px 22px; font-size:.95rem; font-weight:600; font-family:inherit; border:0;
  border-radius:11px; background:var(--acc); color:#fff; cursor:pointer; transition:filter .14s}
button:hover{filter:brightness(1.1)}
.chips{display:flex; gap:7px; flex-wrap:wrap; margin:2px 0 22px; align-items:center}
.chips .lbl{font-size:.8rem; color:var(--tx-3); margin-right:2px}
.chips a{font-size:.815rem; padding:5px 12px; border:1px solid var(--line); border-radius:999px;
  color:var(--tx-2); background:var(--panel); transition:all .14s}
.chips a:hover{border-color:var(--acc); color:var(--acc); text-decoration:none; background:var(--acc-soft)}

.resbar{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin:22px 0 14px;
  padding-bottom:12px; border-bottom:1px solid var(--line)}
.resbar .q{font-weight:640; font-size:1.02rem}
.resbar .m{font-size:.83rem; color:var(--tx-3)}

.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(212px,1fr)); gap:15px}
.card-img{background:var(--panel); border:1px solid var(--line); border-radius:var(--r);
  overflow:hidden; box-shadow:var(--shadow-sm); transition:transform .16s, box-shadow .16s, border-color .16s}
.card-img:hover{transform:translateY(-3px); box-shadow:var(--shadow); border-color:var(--acc-line)}
.card-img .ph{position:relative; display:block; line-height:0}
.card-img img{width:100%; height:168px; object-fit:cover; display:block}
.rank{position:absolute; top:9px; left:9px; min-width:24px; height:24px; padding:0 7px;
  border-radius:8px; background:rgba(10,10,16,.76); color:#fff; font-size:.75rem; font-weight:700;
  display:grid; place-items:center; backdrop-filter:blur(4px)}
.card-img .ft{padding:10px 12px 12px}
.card-img .nm{font-size:.74rem; color:var(--tx-3); font-family:ui-monospace,Menlo,monospace;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.card-img .sc{display:flex; align-items:center; justify-content:space-between; gap:8px; margin-top:5px}
.card-img .sc .v{font-size:.83rem; font-weight:660; font-family:ui-monospace,Menlo,monospace}
.bar{flex:1; height:4px; border-radius:99px; background:var(--panel-2); overflow:hidden}
.bar i{display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,var(--acc),#c084fc)}
.empty{color:var(--tx-3); padding:56px 0; text-align:center}

/* ---------- negation compare ---------- */
.cmp{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:14px}
@media (max-width:720px){ .cmp{grid-template-columns:1fr} }
.cmp .col{background:var(--panel); border:1px solid var(--line); border-radius:var(--r);
  padding:16px 17px; box-shadow:var(--shadow-sm)}
.cmp .col.a{border-top:3px solid var(--pos)}
.cmp .col.b{border-top:3px solid var(--neg)}
.cmp .col h3{margin:0 0 9px; font-size:.86rem; text-transform:uppercase; letter-spacing:.07em;
  color:var(--tx-3); font-weight:700}
.qtext{font-size:.94rem; font-weight:560; margin-bottom:8px}
.pill{display:inline-block; padding:3px 10px; border-radius:999px; font-size:.735rem; font-weight:660}
.pill.neg{background:var(--neg-soft); color:var(--neg); border:1px solid var(--neg-line)}
.pill.aff{background:var(--pos-soft); color:var(--pos); border:1px solid var(--pos-line)}
.row5{display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-top:11px}
.row5 .t{border:1px solid var(--line); border-radius:8px; overflow:hidden; background:var(--panel-2)}
.row5 .t.hit{border-color:var(--acc); box-shadow:0 0 0 2px var(--acc-soft)}
.row5 .t img{width:100%; height:68px; object-fit:cover; display:block}
.row5 .t .s{font-size:.63rem; text-align:center; padding:3px 0; color:var(--tx-3);
  font-family:ui-monospace,Menlo,monospace}

/* ---------- tables ---------- */
.twrap{overflow-x:auto; margin:16px 0; border:1px solid var(--line); border-radius:var(--r);
  background:var(--panel); box-shadow:var(--shadow-sm)}
table{border-collapse:collapse; width:100%; font-size:.865rem; min-width:520px}
th,td{padding:9px 14px; text-align:right; border-bottom:1px solid var(--line-2)}
th{font-size:.73rem; text-transform:uppercase; letter-spacing:.055em; color:var(--tx-3);
   font-weight:700; background:var(--panel-2); white-space:nowrap}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:0}
tr.hl td{background:var(--acc-soft); font-weight:660}
tr.grp td{background:var(--panel-2); font-size:.72rem; text-transform:uppercase;
  letter-spacing:.06em; color:var(--tx-3); font-weight:700; padding:6px 14px}
td.num{font-family:ui-monospace,Menlo,monospace}
.best{color:var(--pos); font-weight:700}
.worse{color:var(--neg)}

.foot{margin-top:56px; padding-top:22px; border-top:1px solid var(--line);
  font-size:.83rem; color:var(--tx-3); display:flex; gap:14px; flex-wrap:wrap;
  justify-content:space-between}
.nextlink{display:inline-flex; align-items:center; gap:8px; margin-top:26px; padding:12px 18px;
  border:1px solid var(--line); border-radius:11px; background:var(--panel); font-weight:600;
  font-size:.92rem; box-shadow:var(--shadow-sm); transition:all .15s}
.nextlink:hover{border-color:var(--acc); text-decoration:none; transform:translateX(3px)}
svg.fig{width:100%; height:auto; display:block; margin:6px 0}
.figwrap{background:var(--panel); border:1px solid var(--line); border-radius:var(--r);
  padding:18px; margin:18px 0; box-shadow:var(--shadow-sm); overflow-x:auto}
.figcap{font-size:.82rem; color:var(--tx-3); margin-top:10px; text-align:center}
.kv{display:grid; grid-template-columns:auto 1fr; gap:6px 16px; font-size:.9rem; margin:14px 0}
.kv dt{color:var(--tx-3)}
.kv dd{margin:0; font-family:ui-monospace,Menlo,monospace; font-size:.86rem}
"""


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #

_NAV = [
    ("/", "1", "CLIP이란?"),
    ("/search", "2", "검색 데모"),
    ("/negation", "3", "Negation 실패"),
    ("/nero", "4", "NeRo-CLIP"),
]


def _page(title: str, body: str, active: str) -> bytes:
    nav = "".join(
        f"<a href='{href}' class='{'active' if href == active else ''}'>"
        f"<span class='n'>{n}</span>{label}</a>"
        for href, n, label in _NAV
    )
    return (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)} · CLIP Demo</title>"
        f"<style>{PAGE_CSS}</style></head><body>"
        "<div class='topbar'><div class='inner'>"
        "<a class='brand' href='/'><span class='dot'></span>CLIP &amp; Negation</a>"
        f"<nav>{nav}</nav>"
        "</div></div>"
        f"<div class='shell'>{body}"
        "<div class='foot'><span>CLIP ViT-B/32 · Flickr30k retrieval demo</span>"
        "<span>NeRo-CLIP — Lee · Kwak · Park</span></div>"
        "</div></body></html>"
    ).encode("utf-8")


def _next(href: str, label: str) -> str:
    return f"<a class='nextlink' href='{href}'>{label} <span>→</span></a>"


# --------------------------------------------------------------------------- #
# Page 1 — what CLIP is
# --------------------------------------------------------------------------- #

def _fig_two_tower() -> str:
    """Two-tower architecture: image encoder + text encoder -> shared space."""
    return """
<svg class="fig" viewBox="0 0 900 260" font-family="-apple-system,system-ui,sans-serif" font-size="12.5">
  <defs><marker id="a1" markerWidth="8" markerHeight="8" refX="6.5" refY="2.6" orient="auto">
    <path d="M0,0 L6.5,2.6 L0,5.2 Z" fill="currentColor"/></marker></defs>
  <g fill="currentColor" color="currentColor">
    <rect x="10" y="26" width="120" height="62" rx="9" fill="#8882" stroke="#8886"/>
    <rect x="52" y="38" width="36" height="26" rx="3" fill="none" stroke="currentColor"
          stroke-width="1.3" opacity=".75"/>
    <circle cx="62" cy="46" r="3" fill="currentColor" opacity=".6"/>
    <path d="M55,61 L67,50 L75,58 L81,53 L85,61 Z" fill="currentColor" opacity=".55"/>
    <text x="70" y="80" text-anchor="middle" font-size="11.5" opacity=".72">이미지</text>

    <rect x="10" y="172" width="120" height="62" rx="9" fill="#8882" stroke="#8886"/>
    <text x="70" y="199" text-anchor="middle" font-size="11.5">“a dog running</text>
    <text x="70" y="215" text-anchor="middle" font-size="11.5">through grass”</text>

    <line x1="132" y1="57" x2="172" y2="57" stroke="currentColor" stroke-width="1.4" marker-end="url(#a1)"/>
    <line x1="132" y1="203" x2="172" y2="203" stroke="currentColor" stroke-width="1.4" marker-end="url(#a1)"/>

    <rect x="174" y="24" width="168" height="66" rx="9" fill="#4f46e51e" stroke="#6366f1"/>
    <text x="258" y="49" text-anchor="middle" font-weight="600">Image Encoder</text>
    <text x="258" y="70" text-anchor="middle" font-size="11" opacity=".72">ViT-B/32 (Transformer)</text>

    <rect x="174" y="170" width="168" height="66" rx="9" fill="#4f46e51e" stroke="#6366f1"/>
    <text x="258" y="195" text-anchor="middle" font-weight="600">Text Encoder</text>
    <text x="258" y="216" text-anchor="middle" font-size="11" opacity=".72">Transformer, [EOS] 토큰</text>

    <line x1="344" y1="57" x2="392" y2="57" stroke="currentColor" stroke-width="1.4" marker-end="url(#a1)"/>
    <line x1="344" y1="203" x2="392" y2="203" stroke="currentColor" stroke-width="1.4" marker-end="url(#a1)"/>

    <rect x="394" y="34" width="116" height="46" rx="8" fill="#8882" stroke="#8886"/>
    <text x="452" y="54" text-anchor="middle" font-size="11.5">linear proj.</text>
    <text x="452" y="70" text-anchor="middle" font-size="11" opacity=".7">→ 512-d</text>
    <rect x="394" y="180" width="116" height="46" rx="8" fill="#8882" stroke="#8886"/>
    <text x="452" y="200" text-anchor="middle" font-size="11.5">linear proj.</text>
    <text x="452" y="216" text-anchor="middle" font-size="11" opacity=".7">→ 512-d</text>

    <path d="M512,57 L560,57 L560,118 L604,118" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#a1)"/>
    <path d="M512,203 L560,203 L560,142 L604,142" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#a1)"/>

    <rect x="606" y="78" width="176" height="104" rx="11" fill="#c084fc22" stroke="#a855f7"/>
    <text x="694" y="104" text-anchor="middle" font-weight="600">공유 임베딩 공간</text>
    <text x="694" y="124" text-anchor="middle" font-size="11" opacity=".72">L2 정규화된 512차원 벡터</text>
    <text x="694" y="150" text-anchor="middle" font-size="11.5" font-family="ui-monospace,monospace">cos(u, v)</text>
    <text x="694" y="168" text-anchor="middle" font-size="11" opacity=".72">= 유사도 점수</text>

    <text x="838" y="118" text-anchor="middle" font-size="26">↔</text>
    <text x="838" y="146" text-anchor="middle" font-size="11" opacity=".72">가까움</text>
  </g>
</svg>"""


def _fig_contrastive() -> str:
    """N x N contrastive similarity matrix with the diagonal highlighted."""
    cells = []
    for r in range(5):
        for c in range(5):
            x, y = 150 + c * 46, 40 + r * 32
            if r == c:
                cells.append(
                    f'<rect x="{x}" y="{y}" width="42" height="28" rx="5" '
                    f'fill="#16a34a33" stroke="#16a34a" stroke-width="1.3"/>'
                    f'<text x="{x + 21}" y="{y + 19}" text-anchor="middle" font-size="11" '
                    f'fill="#16a34a" font-weight="700">↑</text>'
                )
            else:
                cells.append(
                    f'<rect x="{x}" y="{y}" width="42" height="28" rx="5" '
                    f'fill="#dc262614" stroke="#8884" stroke-width="1"/>'
                    f'<text x="{x + 21}" y="{y + 19}" text-anchor="middle" font-size="10" '
                    f'opacity=".45">↓</text>'
                )
    heads = "".join(
        f'<text x="{150 + c * 46 + 21}" y="30" text-anchor="middle" font-size="11" '
        f'opacity=".7" font-family="ui-monospace,monospace">T{c + 1}</text>'
        for c in range(5)
    )
    rows = "".join(
        f'<text x="140" y="{40 + r * 32 + 19}" text-anchor="end" font-size="11" '
        f'opacity=".7" font-family="ui-monospace,monospace">I{r + 1}</text>'
        for r in range(5)
    )
    return f"""
<svg class="fig" viewBox="0 0 640 230" font-family="-apple-system,system-ui,sans-serif">
  <g fill="currentColor">
    <text x="14" y="20" font-size="12" font-weight="600">배치 안의 N×N 유사도 행렬</text>
    {heads}{rows}{''.join(cells)}
    <rect x="410" y="52" width="14" height="14" rx="4" fill="#16a34a33" stroke="#16a34a"/>
    <text x="432" y="64" font-size="11.5">정답 쌍 (대각선) — 유사도를 <tspan font-weight="700">올린다</tspan></text>
    <rect x="410" y="82" width="14" height="14" rx="4" fill="#dc262614" stroke="#8884"/>
    <text x="432" y="94" font-size="11.5">나머지 N²−N 쌍 — 유사도를 <tspan font-weight="700">내린다</tspan></text>
    <text x="410" y="128" font-size="11.5" opacity=".72">한 배치에서 정답 1개 vs 오답 N−1개를</text>
    <text x="410" y="145" font-size="11.5" opacity=".72">맞히는 N-way 분류 문제로 학습.</text>
    <text x="410" y="162" font-size="11.5" opacity=".72">CLIP은 N=32,768 로 학습했습니다.</text>
  </g>
</svg>"""


def _learn_body() -> str:
    return (
        "<div class='eyebrow'>Chapter 1</div>"
        "<h1>CLIP — 이미지와 문장을<br>같은 공간에 담는 모델</h1>"
        "<p class='lede'>CLIP(Contrastive Language–Image Pre-training)은 2021년 OpenAI가 공개한 "
        "모델입니다. 사진과 문장을 <b>하나의 공통 벡터 공간</b>에 올려놓아, 둘 사이의 거리를 "
        "재는 것만으로 검색·분류를 할 수 있게 만듭니다.</p>"

        "<h2><span class='idx'>1.1</span>왜 필요했나</h2>"
        "<p>CLIP 이전의 이미지 분류기는 <b>미리 정해진 클래스 목록</b>에 갇혀 있었습니다. "
        "ImageNet 모델은 1,000개 클래스만 알고, 새 개념 하나를 추가하려면 라벨링된 데이터를 "
        "다시 모아 재학습해야 했죠. 라벨은 사람이 붙여야 하니 확장이 느리고 비쌉니다.</p>"
        "<p>CLIP의 발상은 <b>라벨 대신 인터넷의 캡션을 쓰자</b>는 것입니다. 웹에는 이미 사진과 "
        "그 설명글이 쌍으로 널려 있고, 문장은 열린 어휘라서 어떤 개념이든 표현할 수 있습니다. "
        "OpenAI는 이렇게 웹에서 <b>4억 쌍</b>의 (이미지, 캡션)을 모아 학습시켰습니다.</p>"

        "<h2><span class='idx'>1.2</span>구조: 두 개의 인코더</h2>"
        "<p>CLIP은 서로 다른 두 인코더를 나란히 둡니다. 하나는 이미지를, 하나는 텍스트를 "
        "받아 각각 같은 차원의 벡터를 뱉습니다. 이 데모가 쓰는 ViT-B/32 기준으로 <b>512차원</b>입니다.</p>"
        f"<div class='figwrap'>{_fig_two_tower()}"
        "<div class='figcap'>두 인코더는 서로 가중치를 공유하지 않습니다. 마지막 linear projection이 "
        "둘을 같은 차원으로 맞추고, L2 정규화 후 내적하면 그게 곧 cosine similarity입니다.</div></div>"
        "<div class='cols'>"
        "<div class='c'><h4><span class='ico'>🖼️</span>Image Encoder</h4>"
        "<p>ViT-B/32는 224×224 이미지를 32×32 패치 <b>49개</b>로 자르고, 각 패치를 토큰처럼 "
        "취급해 Transformer에 넣습니다. 맨 앞 [CLS] 토큰의 최종 출력이 이미지 대표 벡터가 됩니다.</p></div>"
        "<div class='c'><h4><span class='ico'>💬</span>Text Encoder</h4>"
        "<p>문장을 BPE로 토크나이즈해 최대 77개 토큰으로 자른 뒤 Transformer에 넣습니다. "
        "문장 끝 <span class='mono'>[EOS]</span> 토큰의 최종 출력이 문장 대표 벡터입니다. "
        "<b>NeRo-CLIP이 손대는 지점이 바로 여기</b>입니다.</p></div>"
        "</div>"

        "<h2><span class='idx'>1.3</span>학습: 대조 학습(Contrastive Learning)</h2>"
        "<p>CLIP에는 \"고양이=3번 클래스\" 같은 정답 라벨이 없습니다. 대신 <b>배치 안에서 짝 맞추기</b> "
        "문제를 풉니다. N개의 (이미지, 캡션) 쌍을 한 배치로 가져와 N×N 유사도 행렬을 만들고, "
        "실제로 짝인 대각선은 높이고 나머지는 낮추도록 학습합니다.</p>"
        f"<div class='figwrap'>{_fig_contrastive()}</div>"
        "<p>손실 함수는 이미지→텍스트, 텍스트→이미지 양방향 cross-entropy의 평균(InfoNCE)입니다. "
        "여기서 τ(temperature)는 학습되는 스칼라로, 유사도 분포를 얼마나 뾰족하게 만들지를 조절합니다.</p>"
        "<div class='formula'>"
        "s<sub>ij</sub> = cos(v<sub>i</sub><span class='cm'>·이미지</span>, "
        "t<sub>j</sub><span class='cm'>·텍스트</span>) / τ\n\n"
        "L = ½ · [ CE(s, 정답=대각선, 행 방향) + CE(s, 정답=대각선, 열 방향) ]"
        "</div>"
        "<div class='note'><div class='hd'>핵심 직관</div>"
        "여기서 배우는 것은 \"이 사진이 무엇인가\"가 아니라 <b>\"이 사진과 이 문장이 서로 어울리는가\"</b>입니다. "
        "그래서 학습이 끝나면 한 번도 본 적 없는 개념도 문장으로 설명만 하면 찾아낼 수 있습니다.</div>"

        "<h2><span class='idx'>1.4</span>추론: 학습 없이 두 가지 일을</h2>"
        "<p>학습이 끝난 CLIP으로 할 수 있는 일은 결국 <b>cosine similarity 하나</b>입니다. "
        "그런데 무엇을 벡터로 만드느냐에 따라 전혀 다른 작업이 됩니다.</p>"
        "<div class='formula'>cos(u, v) = (u · v) / (‖u‖ ‖v‖)  ∈ [−1, 1]</div>"
        "<div class='cols'>"
        "<div class='c'><h4><span class='ico'>🔍</span>Retrieval (검색)</h4>"
        "<p>문장 <b>하나</b>를 벡터로 만들고, 이미지 <b>여러 개</b>의 벡터와 비교해 가장 가까운 "
        "사진을 고릅니다. → <b>이 데모가 하는 일</b></p></div>"
        "<div class='c'><h4><span class='ico'>🏷️</span>Classification (분류)</h4>"
        "<p>이미지 <b>하나</b>를 벡터로 만들고, 클래스 이름을 문장으로 바꾼 "
        "(<span class='mono'>\"a photo of a dog\"</span> …) 벡터 <b>여러 개</b>와 비교해 "
        "가장 가까운 클래스를 답으로 냅니다.</p></div>"
        "</div>"
        "<p>두 경우 모두 <b>추가 학습이 전혀 없습니다</b>. 이것을 zero-shot이라고 부르고, "
        "CLIP이 화제가 된 가장 큰 이유입니다.</p>"

        "<h2><span class='idx'>1.5</span>이 데모의 파이프라인</h2>"
        "<div class='steps'>"
        "<div class='s'><h4>이미지 인덱스 미리 만들기 <span class='sub'>(딱 한 번)</span></h4>"
        "<p>Flickr30k 사진 31,783장을 전부 image encoder에 통과시켜 512차원 벡터로 바꾸고 "
        "파일에 저장합니다. 검색할 때마다 사진을 다시 읽지 않기 위해서입니다.</p></div>"
        "<div class='s'><h4>쿼리 인코딩</h4>"
        "<p>검색창에 입력한 문장을 text encoder에 넣어 512차원 벡터 하나를 얻습니다. "
        "이 부분만 요청마다 실행되므로 검색이 빠릅니다.</p></div>"
        "<div class='s'><h4>내적 한 번으로 전체 비교</h4>"
        "<p>모든 벡터가 L2 정규화되어 있으므로, (31783×512) 행렬과 (512,) 벡터를 곱하면 "
        "31,783개의 cosine similarity가 한 번에 나옵니다. 여기서 상위 k개를 고르면 끝입니다.</p></div>"
        "</div>"

        "<h2><span class='idx'>1.6</span>그런데 잘 안 되는 것들</h2>"
        "<p>CLIP은 강력하지만, 문장을 <b>단어들의 주머니(bag-of-words)</b>처럼 읽는 경향이 "
        "있습니다. 어떤 단어가 있는지는 잘 보지만, 그 단어들이 <b>어떤 관계로 엮여 있는지</b>는 "
        "잘 못 봅니다. 대조 학습 목표 자체가 어순을 굳이 볼 이유를 주지 않기 때문입니다.</p>"
        "<div class='cols'>"
        "<div class='c'><h4><span class='ico'>🚫</span>Negation (부정)</h4>"
        "<p><span class='mono'>\"a beach with no people\"</span> → 사람이 가득한 해변을 반환. "
        "<b>이 데모의 주제</b>입니다.</p></div>"
        "<div class='c'><h4><span class='ico'>🔢</span>Counting (개수)</h4>"
        "<p><span class='mono'>\"three dogs\"</span>와 <span class='mono'>\"five dogs\"</span>를 "
        "거의 구분하지 못합니다.</p></div>"
        "<div class='c'><h4><span class='ico'>🧭</span>Spatial (공간 관계)</h4>"
        "<p><span class='mono'>\"A on B\"</span>와 <span class='mono'>\"B on A\"</span>가 "
        "비슷한 점수를 받습니다.</p></div>"
        "<div class='c'><h4><span class='ico'>🔗</span>Binding (속성 결합)</h4>"
        "<p>\"빨간 차와 파란 집\"에서 어떤 색이 어디에 붙는지 헷갈립니다.</p></div>"
        "</div>"
        "<div class='note bad'><div class='hd'>왜 이런 일이?</div>"
        "원인은 크게 둘로 봅니다. <b>Perception failure</b> — vision encoder가 애초에 객체·속성을 "
        "제대로 못 잡은 경우. <b>Binding failure</b> — 두 모달리티를 cosine similarity 하나로 "
        "묶는 과정에서 정렬이 어긋난 경우. 부정 문제는 후자에 가깝습니다. 부정어에 해당하는 "
        "<b>시각적 증거가 이미지에 없기 때문</b>이죠 — \"모자가 없다\"를 그림으로 그릴 수는 없으니까요.</div>"

        + _next("/search", "2. 실제로 검색해보기")
    )


# --------------------------------------------------------------------------- #
# Page 2 — retrieval demo
# --------------------------------------------------------------------------- #

_SEARCH_ICON = (
    "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' "
    "stroke-linecap='round'><circle cx='11' cy='11' r='7'/><path d='M20 20l-3.5-3.5'/></svg>"
)

_EXAMPLES = [
    "a dog running through the grass",
    "two men playing guitar on stage",
    "a child in a red jacket on a swing",
    "a spaceship on mars",
]


def _search_form(query: str = "", k: int = 12) -> str:
    return (
        "<form class='search' action='/search' method='get'>"
        f"<div class='field'>{_SEARCH_ICON}"
        "<input type='text' name='q' autocomplete='off' "
        "placeholder='영어 문장으로 검색… (예: a dog running through the grass)' "
        f"value=\"{html.escape(query, quote=True)}\" autofocus></div>"
        f"<input type='number' name='k' min='1' max='50' value='{k}' title='결과 개수'>"
        "<button type='submit'>검색</button></form>"
    )


def _examples() -> str:
    links = "".join(
        "<a href='/search?" + urllib.parse.urlencode({"q": e}) + f"'>{html.escape(e)}</a>"
        for e in _EXAMPLES
    )
    return f"<div class='chips'><span class='lbl'>예시</span>{links}</div>"


def _retrieve(query: str, k: int):
    """Encode a query with CLIP and return top-k (indices, scores)."""
    with _State.lock:
        text_emb = encode_texts(_State.model, _State.processor, [query], 1, _State.device)
    idx, sc = topk_search(text_emb, _State.image_embeddings, k)
    return idx[0], sc[0]


def _search_intro() -> str:
    return (
        "<div class='eyebrow'>Chapter 2</div>"
        "<h1>검색 데모</h1>"
        "<p class='lede'>문장을 입력하면 CLIP이 그 문장의 벡터와 가장 가까운 사진을 "
        "Flickr30k 31,783장 중에서 찾아옵니다. 영어 문장이 가장 잘 동작합니다.</p>"
    )


def _do_search(query: str, k: int) -> str:
    idx, scores = _retrieve(query, k)
    vals = [float(s) for s in scores]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0

    cards = []
    for rank, (i, score) in enumerate(zip(idx, vals), start=1):
        name = _State.image_names[int(i)]
        img_url = "/img?name=" + urllib.parse.quote(name)
        # Bar length is relative *within these results* — CLIP scores live in a
        # narrow absolute band, so an absolute scale would look identical always.
        pct = 26 + 74 * (score - lo) / span
        cards.append(
            f"<div class='card-img'>"
            f"<a class='ph' href='{img_url}' target='_blank' rel='noopener'>"
            f"<img src='{img_url}' loading='lazy' alt=''>"
            f"<span class='rank'>{rank}</span></a>"
            f"<div class='ft'><div class='nm'>{html.escape(name)}</div>"
            f"<div class='sc'><span class='bar'><i style='width:{pct:.0f}%'></i></span>"
            f"<span class='v'>{score:.3f}</span></div></div></div>"
        )

    grid = f"<div class='grid'>{''.join(cards)}</div>" if cards else "<div class='empty'>결과 없음</div>"
    return (
        _search_intro()
        + _search_form(query, k)
        + _examples()
        + "<div class='resbar'>"
        f"<span class='q'>“{html.escape(query)}”</span>"
        f"<span class='m mono'>top {len(cards)} · score {lo:.3f} – {hi:.3f}</span></div>"
        + grid
        + "<div class='note'><div class='hd'>점수를 읽는 법</div>"
        "CLIP의 cosine similarity는 확률이 아니라 <b>상대적</b> 값입니다. 전혀 관계없는 쿼리도 "
        "~0.25쯤 나오고, 잘 맞는 쿼리가 ~0.35 정도라 대역이 아주 좁습니다. 절대값보다 "
        "<b>순위</b>를 보세요. 위 막대도 이번 결과 안에서의 상대 길이입니다.</div>"
        "<div class='note bad'><div class='hd'>직접 확인해보기</div>"
        "<span class='mono'>a spaceship on mars</span> 처럼 Flickr30k에 <b>없는</b> 것을 검색해보세요. "
        "CLIP은 \"없다\"고 말하지 못하고, 언제나 가장 가까운 무언가를 반환합니다. "
        "이것이 다음 장에서 볼 실패의 예고편입니다.</div>"
        + _next("/negation", "3. Negation 실패 보기")
    )


# --------------------------------------------------------------------------- #
# Page 3 — negation failure
# --------------------------------------------------------------------------- #

_NEG_PRESETS = [
    ("a man wearing a hat", "a man without a hat"),
    ("a dog on a leash", "a dog without a leash"),
    ("a child holding a toy", "a child holding nothing"),
    ("a street with cars", "a street with no cars"),
]


def _pill(negated: bool, markers: list[str]) -> str:
    if negated:
        return f"<span class='pill neg'>부정어 감지: {html.escape(', '.join(markers))}</span>"
    return "<span class='pill aff'>긍정문</span>"


def _thumbs(indices, scores, mark_first_of: int | None = None) -> str:
    cells = []
    for i, s in zip(indices, scores):
        name = _State.image_names[int(i)]
        url = "/img?name=" + urllib.parse.quote(name)
        hit = " hit" if mark_first_of is not None and int(i) == mark_first_of else ""
        cells.append(
            f"<div class='t{hit}'><img src='{url}' loading='lazy' alt=''>"
            f"<div class='s'>{float(s):.3f}</div></div>"
        )
    return f"<div class='row5'>{''.join(cells)}</div>"


def _neg_column(kind: str, title: str, query: str, mark: int | None = None):
    idx, sc = _retrieve(query, 5)
    negated, markers = route(query)
    col = (
        f"<div class='col {kind}'><h3>{title}</h3>"
        f"<div class='qtext'>“{html.escape(query)}”</div>"
        f"{_pill(negated, markers)}"
        f"{_thumbs(idx, sc, mark)}"
        "</div>"
    )
    return col, [int(x) for x in idx]


def _negation_body(qa: str, qb: str) -> str:
    presets = "<div class='chips'><span class='lbl'>프리셋</span>" + "".join(
        "<a href='/negation?" + urllib.parse.urlencode({"qa": a, "qb": b}) + "'>"
        f"{html.escape(a.replace('a ', '', 1))} ↔ {html.escape(b.replace('a ', '', 1))}</a>"
        for a, b in _NEG_PRESETS
    ) + "</div>"

    form = (
        "<form class='search' action='/negation' method='get'>"
        "<div class='field plain'><input type='text' name='qa' placeholder='긍정 쿼리' "
        f"value=\"{html.escape(qa, quote=True)}\"></div>"
        "<div class='field plain'><input type='text' name='qb' "
        "placeholder='부정 쿼리 (no / without / nothing …)' "
        f"value=\"{html.escape(qb, quote=True)}\"></div>"
        "<button type='submit'>비교</button></form>"
    )

    body = (
        "<div class='eyebrow'>Chapter 3</div>"
        "<h1>CLIP은 부정을 읽지 못합니다</h1>"
        "<p class='lede'>같은 문장에 <span class='mono'>no · without · nothing</span> 만 "
        "붙여도 CLIP은 거의 같은 이미지를 반환합니다. 부정한 바로 그 대상이 가득 담긴 사진을요.</p>"
        "<div class='note'><div class='hd'>왜 그럴까</div>"
        "① 웹에서 모은 학습 캡션 중 부정어가 들어간 것은 <b>1% 미만</b>입니다 — 배울 기회가 "
        "거의 없었습니다. ② 대조 학습 목표는 문장을 순서 없는 개념 덩어리로 읽어도 "
        "벌점을 주지 않습니다. 그 결과 <b>affirmation bias</b>(긍정 편향) — 문장에 등장한 "
        "content word는 부정되든 말든 전부 \"있다\"로 읽는 습관이 생깁니다.</div>"
        + presets
        + form
    )

    if qa and qb:
        col_a, rank_a = _neg_column("a", "긍정 쿼리", qa)
        top_a = rank_a[0]
        col_b, rank_b = _neg_column("b", "부정 쿼리", qb, mark=top_a)
        overlap = len(set(rank_a) & set(rank_b))
        body += f"<div class='cmp'>{col_a}{col_b}</div>"

        if rank_a[0] == rank_b[0]:
            body += (
                "<div class='note bad'><div class='hd'>🚨 top-1이 완전히 동일합니다</div>"
                "CLIP이 부정어를 <b>통째로 무시</b>했습니다. 부정한 바로 그 대상이 담긴 사진을 "
                f"1등으로 골랐고, 상위 5개 중 <b>{overlap}장</b>이 두 쿼리에서 겹칩니다.</div>"
            )
        elif overlap:
            marked = " (오른쪽의 파란 테두리가 긍정 쿼리의 1등 이미지입니다)" if top_a in rank_b else ""
            body += (
                "<div class='note bad'><div class='hd'>top-1은 달라졌지만…</div>"
                f"상위 5개 중 <b>{overlap}장</b>이 여전히 겹칩니다{marked}. 점수 변화도 소수점 "
                "둘째 자리 수준이라, CLIP은 부정을 <b>제대로 이해한 게 아니라 살짝 흔들린 것</b>에 "
                "가깝습니다.</div>"
            )
        else:
            body += (
                "<div class='note good'><div class='hd'>이번엔 상위 5개가 모두 달라졌습니다</div>"
                "부정어가 검색 결과를 실제로 움직인 드문 경우입니다. 다만 점수 대역은 거의 그대로이고, "
                "다른 프리셋에서는 대부분 실패합니다 — CLIP의 부정 처리는 <b>일관성이 없습니다</b>.</div>"
            )
        body += _next("/nero", "4. 이 약점을 고치는 NeRo-CLIP")
    return body


# --------------------------------------------------------------------------- #
# Page 4 — NeRo-CLIP
# --------------------------------------------------------------------------- #

def _nero_diagram() -> str:
    return """
<svg class="fig" viewBox="0 0 920 290" font-family="-apple-system,system-ui,sans-serif" font-size="12.5">
  <defs><marker id="a2" markerWidth="8" markerHeight="8" refX="6.5" refY="2.6" orient="auto">
    <path d="M0,0 L6.5,2.6 L0,5.2 Z" fill="currentColor"/></marker></defs>
  <g fill="currentColor">
    <rect x="8" y="122" width="150" height="48" rx="9" fill="#8882" stroke="#8886"/>
    <text x="83" y="142" text-anchor="middle" font-size="12">caption c</text>
    <text x="83" y="159" text-anchor="middle" font-size="10.5" opacity=".7">“a beach with no people”</text>

    <line x1="160" y1="146" x2="190" y2="146" stroke="currentColor" stroke-width="1.4" marker-end="url(#a2)"/>

    <rect x="192" y="116" width="132" height="60" rx="9" fill="#4f46e51e" stroke="#6366f1"/>
    <text x="258" y="140" text-anchor="middle" font-size="12">❄ frozen</text>
    <text x="258" y="158" text-anchor="middle" font-size="12">CLIP text enc.</text>

    <line x1="326" y1="146" x2="362" y2="146" stroke="currentColor" stroke-width="1.4" marker-end="url(#a2)"/>
    <text x="344" y="138" text-anchor="middle" font-size="11" opacity=".7">z</text>

    <polygon points="412,116 456,146 412,176 368,146" fill="#a855f722" stroke="#a855f7"/>
    <text x="412" y="143" text-anchor="middle" font-size="11">router</text>
    <text x="412" y="157" text-anchor="middle" font-size="11">ϕ(c)?</text>

    <path d="M456,146 L500,146 L500,66 L548,66" fill="none" stroke="#dc2626" stroke-width="1.6" marker-end="url(#a2)"/>
    <text x="545" y="55" text-anchor="end" font-size="11" fill="#dc2626">부정어 있음 · 1.04%</text>
    <rect x="550" y="38" width="310" height="58" rx="9" fill="#dc26261a" stroke="#dc2626"/>
    <text x="705" y="61" text-anchor="middle" font-size="12">rank-8 residual adapter · 8,192 params</text>
    <text x="705" y="80" text-anchor="middle" font-size="11.5" font-family="ui-monospace,monospace">g(z) = z + W_up·tanh(W_down·z)</text>

    <path d="M456,146 L500,146 L500,226 L548,226" fill="none" stroke="#16a34a" stroke-width="1.6" marker-end="url(#a2)"/>
    <text x="545" y="266" text-anchor="end" font-size="11" fill="#16a34a">긍정문 · 98.96%</text>
    <rect x="550" y="202" width="196" height="48" rx="9" fill="#16a34a1a" stroke="#16a34a"/>
    <text x="648" y="231" text-anchor="middle" font-size="12">z 그대로 — 교정 없음</text>

    <path d="M860,67 L888,67 L888,146 L866,146" fill="none" stroke="currentColor" stroke-width="1.3" marker-end="url(#a2)"/>
    <path d="M746,226 L888,226 L888,146 L866,146" fill="none" stroke="currentColor" stroke-width="1.3"/>
    <rect x="756" y="122" width="110" height="48" rx="9" fill="#8882" stroke="#8886"/>
    <text x="811" y="142" text-anchor="middle" font-size="12">cosine sim</text>
    <text x="811" y="159" text-anchor="middle" font-size="10.5" opacity=".7">↔ 이미지 인덱스</text>
  </g>
</svg>"""


def _table_main() -> str:
    groups = [
        ("Baselines", [
            ("CLIP", "—", "39.66", "44.66", "45.6", "0.50", False),
            ("ConCLIP", "백본 전체 · 228K쌍", "25.07", "44.86", "31.8", "0.70", False),
            ("NegCLIP", "백본 전체 · 566K쌍", "26.40", "60.77", "47.8", "0.20", False),
            ("CLIP-CC12M", "백본 전체 · 12M쌍", "55.13", "49.58", "55.3", "0.40", False),
            ("Layerwise steering", "추론 시 보정", "41.60", "36.17", "52.6", "0.80", False),
        ]),
        ("NeRo-CLIP (ours)", [
            ("NeRo-CLIP λ=0.75", "8K · 백본 동결", "54.70", "50.08", "49.7", "0.60", True),
        ]),
    ]
    out = (
        "<div class='twrap'><table><thead><tr><th>Method</th><th>학습 규모</th>"
        "<th>MCQ-Neg ↑</th><th>Retrieval-Neg R@1 ↑</th>"
        "<th>SimpleNeg Top-1 ↑</th><th>N-COCO R@1 ↑</th></tr></thead><tbody>"
    )
    for title, rows in groups:
        out += f"<tr class='grp'><td colspan='6'>{title}</td></tr>"
        for name, scale, *nums, hl in rows:
            cls = " class='hl'" if hl else ""
            cells = "".join(f"<td class='num'>{n}</td>" for n in nums)
            out += f"<tr{cls}><td>{name}</td><td class='sub'>{scale}</td>{cells}</tr>"
    return out + "</tbody></table></div>"


def _table_retention() -> str:
    return (
        "<div class='twrap'><table><thead><tr><th>Method</th>"
        "<th>COCO val2017 R@1 ↑</th><th>Δ vs CLIP</th><th>개입 범위</th></tr></thead><tbody>"
        "<tr><td>CLIP (기준)</td><td class='num'>30.36</td><td class='num'>—</td>"
        "<td class='sub'>없음</td></tr>"
        "<tr><td>Layerwise steering</td><td class='num'>26.06</td>"
        "<td class='num worse'>−4.30</td><td class='sub'>모든 문장 · 12개 층 전부</td></tr>"
        "<tr class='hl'><td>NeRo-CLIP (ours)</td><td class='num'>30.37</td>"
        "<td class='num best'>+0.01</td><td class='sub'>캡션의 1.04%만</td></tr>"
        "</tbody></table></div>"
    )


def _table_lambda() -> str:
    rows = [
        ("0.0", "검색만", "37.80", "49.06", "검색은 오르지만 MCQ가 CLIP 아래로"),
        ("0.25", "", "42.08", "51.66", ""),
        ("0.5", "", "42.82", "50.80", ""),
        ("0.75", "권장", "54.70", "50.08", "둘 다 CLIP 위 — 배포 설정"),
        ("1.0", "MCQ만", "56.85", "25.68", "MCQ 최고, 검색 붕괴 (−18.98pp)"),
    ]
    out = (
        "<div class='twrap'><table><thead><tr><th>λ</th><th>MCQ-Neg ↑</th>"
        "<th>Retrieval-Neg R@1 ↑</th><th>해석</th></tr></thead><tbody>"
    )
    for lam, tag, mcq, ret, note in rows:
        cls = " class='hl'" if lam == "0.75" else ""
        label = f"λ = {lam}" + (f" <span class='sub'>({tag})</span>" if tag else "")
        out += (
            f"<tr{cls}><td>{label}</td><td class='num'>{mcq}</td>"
            f"<td class='num'>{ret}</td><td class='sub' style='text-align:left'>{note}</td></tr>"
        )
    return out + "</tbody></table></div>"


def _nero_body() -> str:
    r = NERO_REPRO
    markers = ", ".join(NEGATION_MARKERS)
    return (
        "<div class='eyebrow'>Chapter 4</div>"
        "<h1>NeRo-CLIP<br>부정만 골라 고치는 어댑터</h1>"
        "<p class='lede'>Negation-Routed adapter for frozen CLIP — Dasol Lee · Jaehyun Kwak · "
        "Seungwon Park. CLIP을 <b>전혀 건드리지 않고</b>, 부정문일 때만 8,192개짜리 작은 "
        "보정기를 끼워 넣습니다.</p>"

        "<h2><span class='idx'>4.1</span>기존 해법의 문제</h2>"
        "<p>부정 문제를 고치려는 시도는 크게 둘이었습니다. <b>재학습</b>(NegCLIP, ConCLIP, "
        "CLIP-CC12M)은 부정 데이터 56만~1200만 쌍으로 백본 전체를 다시 학습합니다. "
        "<b>활성값 스티어링</b>은 텍스트 인코더 12개 층 전부에서 임베딩을 \"부정 방향\"으로 밀어냅니다.</p>"
        "<div class='note bad'><div class='hd'>공통 약점: 개입 범위가 너무 넓다</div>"
        "실제 검색 트래픽에서 부정문은 드물고 평범한 긍정문이 대부분입니다. 그런데 두 방법 모두 "
        "<b>모든 문장</b>에 영향을 줍니다. 그 결과 재학습 3개 중 2개는 오히려 MCQ-Neg가 CLIP보다 "
        "떨어졌고(25.07 / 26.40), 스티어링은 일반 COCO 검색을 <b>4.30pp</b> 깎아먹었습니다.</div>"

        "<h2><span class='idx'>4.2</span>발상의 전환: 선택적 개입</h2>"
        "<p>NeRo-CLIP은 부정 교정을 <b>모델 전체를 바꾸는 문제</b>가 아니라 "
        "<b>선택적 개입(selective intervention) 문제</b>로 다시 정의합니다. 좋은 교정이라면 "
        "두 가지를 동시에 만족해야 한다는 것이죠 — 부정문에서의 <b>정확도</b>, 그리고 "
        "긍정문을 건드리지 않는 <b>선택성</b>.</p>"
        f"<div class='figwrap'>{_nero_diagram()}</div>"

        "<h2><span class='idx'>4.3</span>부품 세 개</h2>"
        "<div class='steps'>"
        "<div class='s'><h4>정규식 라우터</h4>"
        f"<p>부정어 11개 — <span class='mono'>{html.escape(markers)}</span> — 중 하나라도 "
        "있는지만 봅니다. 있으면 고치고, 없으면 CLIP 원본 그대로 통과. COCO 캡션 25,000개 중 "
        "<b>260개(1.04%)</b>에만 반응합니다.</p></div>"
        "<div class='s'><h4>랭크-8 잔차 어댑터</h4>"
        "<p>CLIP 텍스트 인코더의 <b>맨 마지막 [EOS] 임베딩</b> 한 곳에만 붙는 보정기입니다. "
        "W_up을 0으로 초기화해 학습 시작 시점엔 아무것도 바꾸지 않는 항등함수로 출발합니다. "
        "CLIP 인코더는 이미지·텍스트 모두 완전히 동결.</p></div>"
        "<div class='s'><h4>MCQ + 검색 혼합 손실</h4>"
        "<p>계수 λ 하나로 두 목표를 섞습니다. MCQ 손실은 <b>문장 vs 문장</b>을 가르는 힘을, "
        "검색 손실은 <b>이미지 vs 이미지</b>를 가르는 힘을 줍니다.</p></div>"
        "</div>"
        "<div class='formula'>"
        "부정이면:  g(z) = z + W_up · tanh(W_down · z)   "
        "<span class='cm'># W_down: 8×512, W_up: 512×8</span>\n"
        "아니면:    z 그대로                              "
        "<span class='cm'># 교정 없음</span>\n\n"
        "학습:     L = λ · L_MCQ + (1−λ) · L_retrieval"
        "</div>"

        "<h2><span class='idx'>4.4</span>결과 — 부정 벤치마크 4종</h2>"
        "<p class='sub'>CLIP ViT-B/32, 동일 프로토콜로 재측정. 논문 Table 1.</p>"
        + _table_main()
        + "<div class='note good'><div class='hd'>CLIP 대비 4개 전부 개선</div>"
        "특히 1200만 쌍으로 백본 전체를 재학습한 CLIP-CC12M(55.13)에, "
        "<b>8천 파라미터 · 약 2.4만 쌍</b>으로 54.70까지 따라붙습니다.</div>"

        "<h2><span class='idx'>4.5</span>결과 — 일반 검색은 그대로</h2>"
        "<p>부정 교정은 부정문 점수만으로 평가하면 안 됩니다. 긍정문이 대부분인 실제 트래픽에서 "
        "<b>잘 되던 검색을 망치지 않는지</b>가 똑같이 중요합니다.</p>"
        + _table_retention()
        + "<div class='note good'><div class='hd'>이것이 라우팅의 핵심 효과</div>"
        "스티어링은 모든 문장을 건드려 4.30pp를 잃는 반면, NeRo-CLIP은 1.04%에만 개입하므로 "
        "일반 검색이 <b>사실상 그대로</b>입니다 (30.36 → 30.37).</div>"

        "<h2><span class='idx'>4.6</span>왜 λ = 0.75 인가</h2>"
        + _table_lambda()
        + "<p>λ=1(MCQ만)로 학습하면 검색이 붕괴합니다. 4개 후보만 구분하면 되니 상관없는 "
        "캡션끼리 뭉개져도 벌점이 없거든요. λ=0.75는 <b>MCQ를 포기하지 않으면서 검색을 살릴 수 "
        "있는 가장 큰 값</b>입니다 — MCQ는 2.15pp만 손해 보고, 검색은 24.40pp를 되찾습니다.</p>"

        "<h2><span class='idx'>4.7</span>이 데모에서의 재현</h2>"
        "<div class='card'>"
        "<dl class='kv'>"
        "<dt>백본</dt><dd>HuggingFace CLIP ViT-B/32</dd>"
        "<dt>평가</dt><dd>held-out COCO MCQ-Neg · "
        f"n = {r['n']}</dd>"
        f"<dt>CLIP</dt><dd>{r['clip']:.2f}</dd>"
        f"<dt>NeRo-CLIP</dt><dd>{r['nero']:.2f} &nbsp;(<span class='best'>"
        f"+{r['nero'] - r['clip']:.2f}</span>)</dd>"
        "</dl>"
        "<p class='sub' style='margin:0'>CLIP 기준값이 논문의 39.66과 정확히 일치하고, "
        "NeRo도 논문의 open_clip 결과 54.70과 사실상 동일하게 재현됩니다.</p>"
        "</div>"

        "<h2><span class='idx'>4.8</span>한계</h2>"
        "<ul>"
        "<li><b>백본 커버리지</b> — ViT-B/32 하나에서만 검증했습니다. B/16 · L/14에서도 같은 "
        "트레이드오프가 성립하는지는 미확인입니다.</li>"
        "<li><b>라우터 범위</b> — 정규식이라 <span class='mono'>devoid of</span>, "
        "<span class='mono'>lacking</span> 같은 우회 표현은 놓칩니다. 일부러 보수적으로 잡은 "
        "것으로, 라우터를 넓히면 개입이 잦아져 일반 검색 보존이라는 장점이 사라집니다.</li>"
        "<li><b>적용 범위</b> — \"명시적 부정어 + 동결 CLIP 검색 파이프라인\"이라는 특정 환경을 "
        "위한 사후 보정입니다.</li>"
        "</ul>"

        "<hr>"
        "<p class='sub'>논문 전문은 저장소의 <span class='mono'>assets/NeRo_CLIP_final.pdf</span>, "
        "논문 코드는 <a href='https://github.com/Algorythmsz/261RCOSE46101' target='_blank' "
        "rel='noopener'>github.com/Algorythmsz/261RCOSE46101</a> 에 있습니다.</p>"
    )


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quieter console
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)
        q = (params.get("q", [""])[0]).strip()

        try:
            if path == "/":
                self._send(_page("CLIP이란?", _learn_body(), "/"), "text/html; charset=utf-8")

            elif path == "/search":
                if not q:
                    body = _search_intro() + _search_form() + _examples()
                else:
                    try:
                        k = max(1, min(50, int(params.get("k", ["12"])[0])))
                    except ValueError:
                        k = 12
                    body = _do_search(q, k)
                self._send(_page(q or "검색 데모", body, "/search"), "text/html; charset=utf-8")

            elif path == "/negation":
                qa = params.get("qa", [""])[0].strip()
                qb = params.get("qb", [""])[0].strip()
                self._send(
                    _page("Negation 실패", _negation_body(qa, qb), "/negation"),
                    "text/html; charset=utf-8",
                )

            elif path == "/nero":
                self._send(_page("NeRo-CLIP", _nero_body(), "/nero"), "text/html; charset=utf-8")

            elif path == "/img":
                name = params.get("name", [""])[0]
                if name not in _State.name_to_idx:
                    self._send(b"not found", "text/plain", 404)
                    return
                fpath = _State.images_dir / name
                if not fpath.exists():
                    self._send(b"not found", "text/plain", 404)
                    return
                self._send(fpath.read_bytes(), "image/jpeg")

            else:
                self._send(b"not found", "text/plain", 404)
        except BrokenPipeError:
            pass


def run_server(
    index_dir: str,
    images_dir: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str | None = None,
    cpu: bool = False,
) -> None:
    device = resolve_device(cpu)
    image_embeddings, image_names, _, _, meta, name_to_idx = load_index(Path(index_dir))
    resolved_model = model_name if model_name else meta["model_name"]

    print(f"Loading CLIP model '{resolved_model}' on {device} …")
    model, processor = load_clip(resolved_model, device)

    _State.model = model
    _State.processor = processor
    _State.device = device
    _State.images_dir = Path(images_dir)
    _State.image_embeddings = image_embeddings
    _State.image_names = image_names
    _State.name_to_idx = name_to_idx

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Index: {len(image_names)} images | serving on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
