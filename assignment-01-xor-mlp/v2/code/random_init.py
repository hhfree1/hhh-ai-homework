"""随机初值实验：20 次随机初始化，同样的结构与优化器，看几次能学会 XOR。

规则写死在这里，页面上的「20 次里 N 次」只能来自本脚本的输出：

    第 s 次（s = 0..19）用 numpy 的 default_rng(s) 抽样
        W  ~ N(0, 2)  形状 (2,2)
        bh ~ N(0, 1)  形状 (2,)
        v  ~ N(0, 2)  形状 (2,1)
        bo ~ N(0, 1)  形状 (1,)
    其余与 run_and_log.py 完全一致：BCE、Adam lr=0.03、300 轮、全批量、float64。
    判定「学会」= 4 个样本全部判对（p >= 0.5 与标签一致），不看 loss 阈值。

失败样本要保留，不许为了提高成功率而换种子或改轮数。
输出 data/random_init.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from run_and_log import DTYPE, LABELS, PARAM_ORDER, SAMPLES, ROUND, train

N_RUNS = 20
SCALE_W = 2.0
SCALE_B = 1.0
SCALE_V = 2.0


def draw(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, SCALE_W, size=(2, 2))
    bh = rng.normal(0.0, SCALE_B, size=(2,))
    v = rng.normal(0.0, SCALE_V, size=(2, 1))
    bo = rng.normal(0.0, SCALE_B, size=(1,))
    return np.array([W[0, 0], W[0, 1], W[1, 0], W[1, 1], bh[0], bh[1], v[0, 0], v[1, 0], bo[0]],
                    dtype=DTYPE)


def main(argv) -> int:
    root = Path(__file__).resolve().parents[1]
    out = Path(argv[1]).resolve() if len(argv) > 1 else root / "data" / "random_init.json"

    labels = [int(v[0]) for v in LABELS]
    runs = []
    success = 0
    for seed in range(N_RUNS):
        theta0 = draw(seed)
        res = train(theta0=theta0, keep_frames=False)
        pred = [float(p) for p in res["pred"]]
        correct = sum(1 for p, t in zip(pred, labels) if (p >= 0.5) == (t == 1))
        if correct == 4:
            success += 1
        runs.append({
            "seed": seed,
            "loss": round(float(res["loss"]), ROUND),
            "pred": [round(p, ROUND) for p in pred],
            "correct": int(correct),
            "theta_init": [round(float(v), ROUND) for v in theta0],
        })

    payload = {
        "meta": {
            "generated_by": "assignment-01-xor-mlp/code/random_init.py",
            "rule": (
                f"numpy default_rng(seed) 抽样：W~N(0,{SCALE_W}), bh~N(0,{SCALE_B}), "
                f"v~N(0,{SCALE_V}), bo~N(0,{SCALE_B})；BCE + Adam lr=0.03，300 轮，全批量，float64"
            ),
            "success_rule": "4 个样本全部判对（p>=0.5 与标签一致）",
            "n": N_RUNS,
            "param_order": PARAM_ORDER,
        },
        "success": success,
        "failed": N_RUNS - success,
        "runs": runs,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8")

    print(f"写入 {out}")
    print(f"{N_RUNS} 次随机初值：学会 {success} 次，失败 {N_RUNS - success} 次")
    bad = [f"seed={r['seed']}({r['correct']}/4)" for r in runs if r["correct"] < 4]
    print("失败样本 " + ("、".join(bad) if bad else "无"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
