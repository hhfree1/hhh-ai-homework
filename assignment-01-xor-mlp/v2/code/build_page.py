"""把 data/metrics.json 的数字写进 template.html，生成 index.html 与静态图。

三条硬规矩：
1. 页面上的每一个数据数字都必须从这里来。template.html 里只允许出现结构性的坐标、
   字号这类设计参数，不许出现损失、概率、参数值。
2. 同一份数据连跑两次，输出逐字节一致（生成的 SVG/PNG 里不写时间、不写随机 id）。
3. 占位符写错就报错，不许静默留白：模板里有未知占位符、或生成后还残留 `{{`，都算失败。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COL = {
    "bg": "#0B0D10",
    "panel": "#12151A",
    "code": "#08090B",
    "text": "#E8E6E1",
    "muted": "#9AA0A6",
    "line": "rgba(232,230,225,0.10)",
    "signal": "#3DDC97",
}

PARAM_MEANING = {
    "W00": "输入 x₁ → 隐藏 h₁",
    "W01": "输入 x₂ → 隐藏 h₁",
    "W10": "输入 x₁ → 隐藏 h₂",
    "W11": "输入 x₂ → 隐藏 h₂",
    "bh0": "隐藏 h₁ 偏置",
    "bh1": "隐藏 h₂ 偏置",
    "v0": "隐藏 h₁ → 输出",
    "v1": "隐藏 h₂ → 输出",
    "bo": "输出层偏置",
}


# ---------------------------------------------------------------- 数字格式
def f4(x: float) -> str:
    return f"{x:.4f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed4(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.4f}"


def pct(a: float, b: float) -> str:
    if a == 0:
        return "—"
    return f"{(b / a * 100):.1f}%"


# ---------------------------------------------------------------- 前向
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(theta, x):
    W = np.array([[theta[0], theta[1]], [theta[2], theta[3]]], dtype=np.float64)
    bh = np.array([theta[4], theta[5]], dtype=np.float64)
    v = np.array([[theta[6]], [theta[7]]], dtype=np.float64)
    h = sigmoid(x @ W.T + bh)
    p = sigmoid(h @ v + float(theta[8]))
    return h, p


def prob_grid(theta, n=241):
    """在 [0,1]^2 上算概率网格，返回 (n, n) 数组。"""
    t = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(t, t)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    _, p = forward(theta, pts)
    return p.reshape(n, n)


# ---------------------------------------------------------------- 静态图
def fig_net(theta) -> str:
    """2-2-1 结构图，线宽随 |w|，亮色为正、灰色为负。

    标签一律离边线走，并且用 paint-order 描一圈底色当晕，压在交叉点上也不糊。
    """
    W = [[theta[0], theta[1]], [theta[2], theta[3]]]
    bh = [theta[4], theta[5]]
    v = [theta[6], theta[7]]
    bo = theta[8]

    xs = [96, 300, 504]
    ys_in = [120, 264]
    ys_h = [120, 264]
    y_out = 192
    r = 26
    maxw = max(abs(w) for row in W for w in row) or 1.0
    maxv = max(abs(x) for x in v) or 1.0

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 380" width="600" height="380" '
        'role="img" aria-label="2-2-1 网络结构图">',
        f'<rect width="600" height="380" fill="{COL["panel"]}"/>',
    ]
    mono = 'font-family="ui-monospace,SFMono-Regular,Consolas,monospace"'

    def line(x1, y1, x2, y2, w, wmax):
        t = abs(w) / wmax
        color = COL["signal"] if w > 0 else COL["muted"]
        op = 0.35 + 0.65 * t
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                     f'stroke-width="{0.8 + 4.0 * t:.2f}" stroke-opacity="{op:.2f}" stroke-linecap="round"/>')

    def label(x1, y1, x2, y2, frac, off, text):
        """沿边取 frac 处，向法线方向让开 off 像素写标签。"""
        mx = x1 + (x2 - x1) * frac
        my = y1 + (y2 - y1) * frac
        dx, dy = x2 - x1, y2 - y1
        ln = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / ln * off, dx / ln * off
        parts.append(
            f'<text x="{mx + nx:.1f}" y="{my + ny + 3.5:.1f}" text-anchor="middle" fill="{COL["muted"]}" '
            f'{mono} font-size="10" stroke="{COL["panel"]}" stroke-width="3" paint-order="stroke">{text}</text>'
        )

    # 四条输入→隐藏的线，标签都靠输入端，避开中间的交叉
    line(xs[0] + r, ys_in[0], xs[1] - r, ys_h[0], W[0][0], maxw)
    line(xs[0] + r, ys_in[1], xs[1] - r, ys_h[0], W[0][1], maxw)
    line(xs[0] + r, ys_in[0], xs[1] - r, ys_h[1], W[1][0], maxw)
    line(xs[0] + r, ys_in[1], xs[1] - r, ys_h[1], W[1][1], maxw)
    line(xs[1] + r, ys_h[0], xs[2] - r, y_out, v[0], maxv)
    line(xs[1] + r, ys_h[1], xs[2] - r, y_out, v[1], maxv)

    label(xs[0] + r, ys_in[0], xs[1] - r, ys_h[0], 0.5, -11, f"W₀₀ {f3(W[0][0])}")
    label(xs[0] + r, ys_in[1], xs[1] - r, ys_h[0], 0.30, -11, f"W₀₁ {f3(W[0][1])}")
    label(xs[0] + r, ys_in[0], xs[1] - r, ys_h[1], 0.30, 13, f"W₁₀ {f3(W[1][0])}")
    label(xs[0] + r, ys_in[1], xs[1] - r, ys_h[1], 0.5, 13, f"W₁₁ {f3(W[1][1])}")
    label(xs[1] + r, ys_h[0], xs[2] - r, y_out, 0.44, -12, f"v₁ {f3(v[0])}")
    label(xs[1] + r, ys_h[1], xs[2] - r, y_out, 0.44, 14, f"v₂ {f3(v[1])}")

    def node(x, y, text, accent=False):
        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{COL["bg"]}" '
                     f'stroke="{COL["signal"] if accent else COL["line"]}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x}" y="{y + 4}" text-anchor="middle" fill="{COL["text"]}" {mono} '
                     f'font-size="11">{text}</text>')

    node(xs[0], ys_in[0], "x₁")
    node(xs[0], ys_in[1], "x₂")
    node(xs[1], ys_h[0], "h₁")
    node(xs[1], ys_h[1], "h₂")
    node(xs[2], y_out, "p", True)

    for x, y, text in (
        (xs[1], ys_h[0] - r - 14, f"b₁ {f3(bh[0])}"),
        (xs[1], ys_h[1] + r + 22, f"b₂ {f3(bh[1])}"),
        (xs[2], y_out - r - 14, f"b₀ {f3(bo)}"),
    ):
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" fill="{COL["muted"]}" {mono} '
                     f'font-size="10">{text}</text>')

    parts.append("</svg>")
    return "".join(parts)


def fig_loss(frames) -> str:
    losses = [max(f[1], 1e-6) for f in frames]
    lo, hi = math.log10(min(losses)), math.log10(max(losses))
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    w, h = 720, 300
    padl, padr, padt, padb = 58, 18, 18, 34

    def X(i):
        return padl + (w - padl - padr) * i / (len(losses) - 1)

    def Y(v):
        t = (math.log10(v) - lo) / (hi - lo)
        return padt + (h - padt - padb) * (1 - t)

    pts = " ".join(f"{X(i):.2f},{Y(v):.2f}" for i, v in enumerate(losses))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-label="损失曲线，对数纵轴">',
        f'<rect width="{w}" height="{h}" fill="{COL["panel"]}"/>',
    ]
    for tick in (1.0, 0.5, 0.1, 0.05, 0.02, 0.01):
        if not (min(losses) * 0.9 <= tick <= max(losses) * 1.05):
            continue
        y = Y(tick)
        parts.append(f'<line x1="{padl}" y1="{y:.2f}" x2="{w - padr}" y2="{y:.2f}" stroke="{COL["line"]}" stroke-width="1"/>')
        parts.append(f'<text x="{padl - 8}" y="{y + 3.5:.2f}" text-anchor="end" fill="{COL["muted"]}" '
                     f'font-family="ui-monospace,monospace" font-size="10">{tick:g}</text>')
    for e in (0, 100, 200, 300):
        if e >= len(losses):
            continue
        x = X(e)
        parts.append(f'<line x1="{x:.2f}" y1="{padt}" x2="{x:.2f}" y2="{h - padb}" stroke="{COL["line"]}" stroke-width="1" stroke-dasharray="2 4"/>')
        parts.append(f'<text x="{x:.2f}" y="{h - padb + 16}" text-anchor="middle" fill="{COL["muted"]}" '
                     f'font-family="ui-monospace,monospace" font-size="10">{e}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{COL["signal"]}" stroke-width="1.6"/>')
    parts.append(f'<circle cx="{X(len(losses) - 1):.2f}" cy="{Y(losses[-1]):.2f}" r="3.2" fill="{COL["signal"]}"/>')
    parts.append(f'<text x="{padl}" y="{h - 2}" fill="{COL["muted"]}" font-family="ui-monospace,monospace" font-size="10">轮次 →（纵轴对数）</text>')
    parts.append("</svg>")
    return "".join(parts)


def load_font(size: int):
    """静态图上的小字用系统 UI 字体，保证 ₂ ↑ 这类符号不是豆腐块。"""
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fig_boundary(theta, labels, out: Path, size=460) -> None:
    """决策边界的静态快照：两类用明暗区分，p=0.5 画白线，四个样本点标出来。"""
    grid = prob_grid(theta, 231)
    n = grid.shape[0]
    bg = np.array([11, 13, 16], dtype=np.float64)
    pos = np.array([61, 220, 151], dtype=np.float64)     # 信号色：判 1
    neg = np.array([58, 66, 78], dtype=np.float64)       # 中性灰蓝：判 0
    conf = np.clip(np.abs(grid - 0.5) * 2.0, 0, 1) ** 0.6
    rgb = np.zeros((n, n, 3), dtype=np.float64)
    hi = grid >= 0.5
    rgb[hi] = bg + (pos - bg) * (0.10 + 0.55 * conf[hi])[:, None]
    rgb[~hi] = bg + (neg - bg) * (0.10 + 0.60 * conf[~hi])[:, None]
    near = np.abs(grid - 0.5) < 0.010
    rgb[near] = np.array([232, 230, 225], dtype=np.float64)
    rgb = rgb[::-1]  # x₂ 向上
    im = Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize((size, size), Image.LANCZOS)

    d = ImageDraw.Draw(im)
    f = load_font(15)
    pad = 34
    span = size - pad * 2
    for k in range(5):
        t = pad + span * k / 4
        d.line([(pad, t), (size - pad, t)], fill=(48, 54, 62), width=1)
        d.line([(t, pad), (t, size - pad)], fill=(48, 54, 62), width=1)
    for i, s in enumerate(([0, 0], [0, 1], [1, 0], [1, 1])):
        cx = pad + span * s[0]
        cy = size - pad - span * s[1]
        y = labels[i]
        fill = (61, 220, 151) if y == 1 else (232, 230, 225)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=fill, width=2)
        tx = cx - 16 if s[0] >= 1 else cx + 8
        d.text((tx, cy - 7), str(y), fill=fill, font=f)
    d.text((6, pad - 28), "x2", fill=(150, 156, 164), font=f)
    d.text((size - 26, size - 30), "x1", fill=(150, 156, 164), font=f)
    im.save(out, format="PNG", optimize=True)


# ---------------------------------------------------------------- 表格行
def rows_params(m) -> str:
    theta0 = m["init"]["theta"]
    theta1 = m["final"]["theta"]
    out = []
    for k, a, b in zip(m["meta"]["param_order"], theta0, theta1):
        out.append(
            f'<tr><td class="mono">{k}</td><td class="muted">{PARAM_MEANING[k]}</td>'
            f'<td class="num mono">{signed4(a)}</td><td class="num mono">{signed4(b)}</td>'
            f'<td class="num mono muted">{signed4(b - a)}</td></tr>'
        )
    return "\n              ".join(out)


def rows_samples(m) -> str:
    out = []
    for i, (s, y) in enumerate(zip(m["samples"], m["labels"])):
        h = m["final"]["hidden"][i]
        p = m["final"]["pred"][i]
        ok = (p >= 0.5) == (y == 1)
        mark = '<span class="signal">判对</span>' if ok else '<span class="muted">判错</span>'
        out.append(
            f'<tr><td class="mono">({int(s[0])}, {int(s[1])}) → {y}</td>'
            f'<td class="num mono">{f4(h[0])}</td><td class="num mono">{f4(h[1])}</td>'
            f'<td class="num mono">{f4(p)}</td><td>{mark}</td></tr>'
        )
    return "\n                ".join(out)


def rows_key(m) -> str:
    l0, l1 = m["init"]["loss"], m["final"]["loss"]
    p0, p1 = m["init"]["pred"], m["final"]["pred"]
    acc0 = sum(1 for p, y in zip(p0, m["labels"]) if (p >= 0.5) == (y == 1))
    acc1 = m["final"]["correct"]
    rng0 = f"{f4(min(p0))} – {f4(max(p0))}"
    rng1 = f"{f4(min(p1))} – {f4(max(p1))}"
    mg0 = min(abs(p - 0.5) for p in p0)
    mg1 = m["final"]["min_margin"]
    return "\n                ".join([
        f'<tr><td>交叉熵损失</td><td class="num mono">{f4(l0)}</td><td class="num mono">{f4(l1)}</td>'
        f'<td class="muted">降到原来的 {pct(l0, l1)}</td></tr>',
        f'<tr><td>判对数（4 个样本）</td><td class="num mono">{acc0}/4</td><td class="num mono">{acc1}/4</td>'
        f'<td class="muted">阈值为 0.5</td></tr>',
        f'<tr><td>输出概率范围</td><td class="num mono">{rng0}</td><td class="num mono">{rng1}</td>'
        f'<td class="muted">越接近 0 或 1 越有把握</td></tr>',
        f'<tr><td>最小判对边界 |p−0.5|</td><td class="num mono">{f4(mg0)}</td><td class="num mono">{f4(mg1)}</td>'
        f'<td class="muted">离 0.5 最近的那个样本有多远</td></tr>',
    ])


def rows_grad(m) -> str:
    first = m["frames"][0]
    # 帧布局：e(0) loss(1) q0..q3(2..5) h0..h7(6..13) theta0..8(14..22) g0..8(23..31)
    grads = first[23:32]
    out = []
    for k, g in zip(m["meta"]["param_order"], grads):
        d = "正 · 该变小" if g > 0 else ("负 · 该变大" if g < 0 else "零")
        out.append(f'<tr><td class="mono">{k}</td><td class="num mono">{signed4(g)}</td>'
                   f'<td class="muted">{d}</td></tr>')
    return "\n                ".join(out)


# ---------------------------------------------------------------- 组装
def build() -> None:
    metrics = json.loads((DATA / "metrics.json").read_text(encoding="utf-8"))
    rnd = json.loads((DATA / "random_init.json").read_text(encoding="utf-8"))

    raw = (DATA / "metrics.json").read_bytes()
    build_id = hashlib.sha256(raw).hexdigest()[:12]

    frames = metrics["frames"]
    theta_final = metrics["final"]["theta"]
    theta_init = metrics["init"]["theta"]

    ok = rnd["success"]
    bad = [str(r["seed"]) for r in rnd["runs"] if r["correct"] != 4]

    K = {
        "loss_init": f4(metrics["init"]["loss"]),
        "loss_end": f4(metrics["final"]["loss"]),
        "acc": str(metrics["final"]["correct"]),
        "margin": f4(metrics["final"]["min_margin"]),
        "epochs": str(metrics["meta"]["epochs"]),
        "frames": str(len(frames)),
        "lr": str(metrics["meta"]["lr"]),
        "params": str(len(metrics["meta"]["param_order"])),
        "seed": str(metrics["meta"]["seed"]),
        "dtype": metrics["meta"]["dtype"],
        "rand_ok": str(ok),
        "rand_fail": str(len(rnd["runs"]) - ok),
        "rand_bad": ", ".join(bad),
        "build": build_id,
    }

    tpl = (ROOT / "template.html").read_text(encoding="utf-8")
    html = re.sub(r"\{\{K:(\w+)\}\}", lambda mm: K.get(mm.group(1), mm.group(0)), tpl)
    blocks = {
        "ROWS_PARAMS": rows_params(metrics),
        "ROWS_SAMPLES": rows_samples(metrics),
        "ROWS_KEY": rows_key(metrics),
        "ROWS_GRAD": rows_grad(metrics),
    }
    for name, body in blocks.items():
        html = html.replace("{{" + name + "}}", body)

    leftover = re.findall(r"\{\{[^}]*\}\}", html.replace("{{DATA}}", ""))
    if leftover:
        raise SystemExit("模板里还有没替换的占位符：" + str(leftover[:5]))
    if "0.5620" in tpl:
        raise SystemExit("模板里出现了数据数字，必须改成占位符")

    # 回放数据内联进页面：file:// 下 fetch 会被 CORS 挡住，所以不额外请求 json 文件
    payload = {
        "meta": {
            "epochs": metrics["meta"]["epochs"],
            "lr": metrics["meta"]["lr"],
            "dtype": metrics["meta"]["dtype"],
            "param_order": metrics["meta"]["param_order"],
            "frame_layout": metrics["meta"]["frame_layout"],
        },
        "samples": metrics["samples"],
        "labels": metrics["labels"],
        "frames": [[round(v, 4) for v in f] for f in frames],
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    html = html.replace("{{DATA}}", data_json)

    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figures" / "fig_net.svg").write_text(fig_net(theta_final), encoding="utf-8")
    (ROOT / "figures" / "fig_loss.svg").write_text(fig_loss(frames), encoding="utf-8")
    fig_boundary(theta_init, metrics["labels"], ROOT / "figures" / "fig_boundary_init.png")
    fig_boundary(theta_final, metrics["labels"], ROOT / "figures" / "fig_boundary_final.png")

    (ROOT / "index.html").write_text(html, encoding="utf-8")
    print("已生成 index.html 与 4 张静态图")
    print(f"数据指纹 {build_id} · 帧数 {len(frames)} · 占位符 {len(K)} 个 · 表格块 {len(blocks)} 个")


if __name__ == "__main__":
    build()
    sys.exit(0)
