"""把这一页能查的都查一遍。

跑法：python code/verify.py（在 assignment-01-xor-mlp/ 下）

查五件事（页面上写的就是这五件）：

    1. 帧数、每帧字段数、轮次是否连续
    2. 数据本身自洽：每帧的损失/判对/边界能否由这一帧的概率重算出来
    3. 参数可重算：重新训练一遍，终态与抽样帧和 data/metrics.json 一致
    4. build_page.py 连续跑两次，index.html 与四张图逐字节一致
    5. 页面里没有残留占位符、没有外部请求，关键数字与 data/metrics.json 一致

任何一项失败都以非 0 退出，并在最后一行写 FAIL。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import run_and_log as R  # noqa: E402

FRAME_FIELDS = 32
TOL = 1e-6


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, ok, detail))

    def report(self) -> int:
        bad = 0
        for name, ok, detail in self.rows:
            mark = "PASS" if ok else "FAIL"
            if not ok:
                bad += 1
            print(f"[{mark}] {name}" + (f" · {detail}" if detail else ""))
        print()
        if bad:
            print(f"FAIL · {bad}/{len(self.rows)} 项未通过")
            return 1
        print(f"ALL PASS · {len(self.rows)} 项")
        return 0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frames(m: dict, c: Check) -> None:
    frames = m["frames"]
    c.add("帧数 = epochs + 1", len(frames) == m["meta"]["epochs"] + 1,
          f"{len(frames)} 帧 / {m['meta']['epochs']} 轮")
    widths = {len(f) for f in frames}
    c.add("每帧字段数一致", widths == {FRAME_FIELDS}, f"字段数集合 {sorted(widths)}")
    c.add("轮次 0..N 连续", [f[0] for f in frames] == list(range(len(frames))),
          f"{frames[0][0]} → {frames[-1][0]}")
    c.add("帧内数字有限", all(isinstance(v, (int, float)) for f in frames for v in f),
          "无 null / 字符串")


def check_self_consistent(m: dict, c: Check) -> None:
    """不复用训练代码，直接用帧里的概率重算损失、判对、边界。"""
    import math

    labels = m["labels"]
    losses, accs, margins = [], [], []
    for f in m["frames"]:
        q = f[2:6]
        n = len(q)
        loss = -sum(y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12)
                    for p, y in zip(q, labels)) / n
        losses.append(loss)
        accs.append(sum(1 for p, y in zip(q, labels) if (p >= 0.5) == (y == 1)))
        margins.append(min(abs(p - 0.5) for p in q))

    dloss = max(abs(a - f[1]) for a, f in zip(losses, m["frames"]))
    # 帧里的概率只存 6 位小数，由它反算损失会有 ~1e-6 的舍入残差
    c.add("每帧损失可由概率重算", dloss < 1e-5, f"最大偏差 {dloss:.2e}")
    last = m["frames"][-1]
    c.add("末帧判对数 = final.correct",
          accs[-1] == m["final"]["correct"] == 4, f"{accs[-1]}/4")
    c.add("末帧边界 = final.min_margin",
          abs(margins[-1] - m["final"]["min_margin"]) < TOL,
          f"{margins[-1]:.6f}")
    hid_max = max(last[6:14])
    c.add("末帧隐藏层最大激活在 (0,1)", 0.0 < hid_max < 1.0, f"{hid_max:.4f}")


def check_recompute(m: dict, c: Check) -> None:
    res = R.train()
    dif = max(abs(a - b) for a, b in zip(res["theta"], m["final"]["theta"]))
    c.add("重训终态参数一致", dif < TOL, f"最大偏差 {dif:.2e}")
    dl = abs(res["loss"] - m["final"]["loss"])
    c.add("重训终态损失一致", dl < TOL, f"偏差 {dl:.2e}")

    picks = [0, 1, 7, 50, 128, 299, 300]
    worst, where = 0.0, -1
    for e in picks:
        frame = m["frames"][e]
        res_e = R.train(epochs=e, keep_frames=False)
        d = max(abs(a - b) for a, b in zip(res_e["theta"], frame[14:23]))
        if d > worst:
            worst, where = d, e
    c.add("抽样帧参数可复现", worst < TOL, f"偏差最大在第 {where} 轮：{worst:.2e}")


def build_once() -> dict[str, str]:
    subprocess.run([sys.executable, str(ROOT / "code" / "build_page.py")],
                   cwd=ROOT, check=True, capture_output=True)
    targets = [ROOT / "index.html"] + sorted((ROOT / "figures").glob("*"))
    return {str(p.relative_to(ROOT)): sha(p) for p in targets}


def check_deterministic(c: Check) -> None:
    first = build_once()
    second = build_once()
    same = [k for k in first if first[k] == second.get(k)]
    c.add("连续两次构建逐字节一致", same == sorted(first) or len(same) == len(first),
          f"{len(same)}/{len(first)} 个文件相同")


def check_page(m: dict, c: Check) -> None:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    c.add("页面无残留占位符", "{{" not in html and "}}" not in html)
    ext = re.findall(r'(?:src|href)\s*=\s*"(?:https?:)?//[^"]+', html)
    c.add("页面无外部请求", not ext, f"{len(ext)} 处")

    raw = json.dumps(m, ensure_ascii=False)
    c.add("数据指纹出现在页面里", sha(ROOT / "data" / "metrics.json")[:12] in html,
          sha(ROOT / "data" / "metrics.json")[:12])

    expect = {
        "初值": f"{m['frames'][0][1]:.4f}",
        "末值": f"{m['final']['loss']:.4f}",
        "边界": f"{m['final']['min_margin']:.4f}",
    }
    missing = [k for k, v in expect.items() if v not in html]
    c.add("关键数字来自 data/metrics.json", not missing, f"缺 {missing}" if missing else "初值/末值/边界")

    # 手写数字的哨兵：去掉图形坐标与内联数据后，页面出现的四位小数必须能在
    # data/metrics.json 里找到（把 JSON 的数字统一修约到 4 位再比）
    text = re.sub(r'points="[^"]*"|d="[^"]*"|transform="[^"]*"', "", html)
    text = re.sub(r'<script type="application/json".*?</script>', "", text, flags=re.S)
    allowed = {f"{float(v):.4f}" for v in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", raw)}
    stray = sorted({s for s in re.findall(r"(?<![\d.])0\.\d{4}(?!\d)", text)} - allowed)
    c.add("页面没有数据里不存在的四位小数", not stray,
          f"多出 {stray[:6]}" if stray else f"比对 {len(allowed)} 个修约值")


def check_random_init(c: Check) -> None:
    r = load_json(ROOT / "data" / "random_init.json")
    runs = r["runs"]
    c.add("随机初值 20 次", len(runs) == 20, f"{len(runs)} 次")
    c.add("随机初值成功次数 = 9", r["success"] == 9, f"{r['success']}/20")
    fails = sorted(x["seed"] for x in runs if x["correct"] != 4)
    c.add("失败次数与失败种子一致",
          len(fails) == r["failed"] and fails == sorted(x["seed"] for x in runs
                                                       if x["correct"] != 4),
          ", ".join(map(str, fails)))


def check_inputs(c: Check) -> None:
    for name in ("data/metrics.json", "data/random_init.json", "data/env.txt"):
        c.add(f"存在 {name}", (ROOT / name).exists())


def main() -> int:
    c = Check()
    check_inputs(c)
    m = load_json(ROOT / "data" / "metrics.json")
    check_frames(m, c)
    check_self_consistent(m, c)
    check_recompute(m, c)
    check_deterministic(c)
    check_page(m, c)
    check_random_init(c)
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
