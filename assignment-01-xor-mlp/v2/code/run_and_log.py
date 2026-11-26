"""2-2-1 网络学习 XOR：训练并逐帧记录。

约定（全部写死；改任何一项都要同时改页面上标注的配置，verify.py 会比对）：

    样本      [[0,0],[0,1],[1,0],[1,1]]，标签 [0,1,1,0]
    结构      2 -> 2 -> 1，隐藏层 sigmoid，输出层 sigmoid
    初值      W=[[2,-2],[-2,2]]  b=(-1,-1)  v=(3,3)  b_out=-2.5
    损失      交叉熵（BCE），对 4 个样本取平均
    梯度      dL/dz_out = (p - y)，链式到隐藏层时乘 sigmoid 的导数 h(1-h)
    优化器    Adam  lr=0.03  betas=(0.9,0.999)  eps=1e-8  float64
    轮数      300，全批量（每轮 1 次参数更新）

每帧记录优化器更新前的梯度，顺序 W00,W01,W10,W11,bh0,bh1,v0,v1,bo。这是 C7
「梯度箭头」的数据来源：箭头来自梯度本身，不是靠相邻两帧的位置差反推。

输出 data/metrics.json。脚本不写时间戳，同一环境连跑两次输出逐字节一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EPOCHS = 300
LR = 0.03
BETAS = (0.9, 0.999)
EPS = 1e-8
SEED = 7
DTYPE = np.float64
ROUND = 6

SAMPLES = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
LABELS = [[0.0], [1.0], [1.0], [0.0]]

W_INIT = [[2.0, -2.0], [-2.0, 2.0]]
B_INIT = [-1.0, -1.0]
V_INIT = [[3.0], [3.0]]
BO_INIT = [-2.5]

PARAM_ORDER = ["W00", "W01", "W10", "W11", "bh0", "bh1", "v0", "v1", "bo"]
FRAME_LAYOUT = (
    "e,loss,q0..q3,h0..h7,theta0..theta8,g0..g8 —— q 是输出概率，h 是隐藏层激活，"
    "theta/g 的 9 个分量顺序见 param_order；每帧 32 个数"
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def forward(theta, x):
    W = np.array([[theta[0], theta[1]], [theta[2], theta[3]]], dtype=DTYPE)
    bh = np.array([theta[4], theta[5]], dtype=DTYPE)
    v = np.array([[theta[6]], [theta[7]]], dtype=DTYPE)
    bo = float(theta[8])
    hidden = sigmoid(x @ W.T + bh)
    prob = sigmoid(hidden @ v + bo)
    return hidden, prob


def bce(prob, y) -> float:
    eps = 1e-12
    return float(-(y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps)).mean())


def backward(theta, x, y, hidden, prob) -> np.ndarray:
    """dL/dtheta，L = mean(BCE)。输出层的 dL/dz 恰好是 p - y。"""
    n = x.shape[0]
    v = np.array([[theta[6]], [theta[7]]], dtype=DTYPE)
    dz_out = (prob - y) / n                      # dL/dz_out
    dv = hidden.T @ dz_out                       # (2,1)
    dbo = dz_out.sum(axis=0)                     # (1,)
    dh = dz_out @ v.T
    dz_hidden = dh * (hidden * (1.0 - hidden))
    dW = dz_hidden.T @ x                         # (2,2)
    dbh = dz_hidden.sum(axis=0)                  # (2,)
    return np.array([
        dW[0, 0], dW[0, 1], dW[1, 0], dW[1, 1],
        dbh[0], dbh[1], dv[0, 0], dv[1, 0], dbo[0],
    ], dtype=DTYPE)


def train(epochs: int = EPOCHS, lr: float = LR, theta0=None, x=None, y=None,
          keep_frames: bool = True):
    x = np.array(SAMPLES if x is None else x, dtype=DTYPE)
    y = np.array(LABELS if y is None else y, dtype=DTYPE)
    if theta0 is None:
        theta0 = [W_INIT[0][0], W_INIT[0][1], W_INIT[1][0], W_INIT[1][1],
                  B_INIT[0], B_INIT[1], V_INIT[0][0], V_INIT[1][0], BO_INIT[0]]
    theta = np.array(theta0, dtype=DTYPE)

    m = np.zeros(9, dtype=DTYPE)
    s = np.zeros(9, dtype=DTYPE)
    frames = []
    history = []

    for epoch in range(epochs + 1):
        hidden, prob = forward(theta, x)
        loss = bce(prob, y)
        grad = backward(theta, x, y, hidden, prob)

        if keep_frames:
            frames.append(
                [epoch, loss]
                + [float(v) for v in prob.ravel()]
                + [float(v) for v in hidden.ravel()]
                + [float(v) for v in theta]
                + [float(v) for v in grad]
            )
        if epoch % 10 == 0:
            history.append({
                "e": epoch,
                "l": loss,
                "theta": [float(v) for v in theta],
                "p": [float(v) for v in prob.ravel()],
            })
        if epoch == epochs:
            break

        m = BETAS[0] * m + (1 - BETAS[0]) * grad
        s = BETAS[1] * s + (1 - BETAS[1]) * grad * grad
        mhat = m / (1 - BETAS[0] ** (epoch + 1))
        shat = s / (1 - BETAS[1] ** (epoch + 1))
        theta = theta - lr * mhat / (np.sqrt(shat) + EPS)

    hidden, prob = forward(theta, x)
    return {
        "theta": theta,
        "loss": bce(prob, y),
        "pred": prob.ravel(),
        "hidden": hidden,
        "frames": frames,
        "history": history,
    }


def rounded_frames(frames):
    return [[f[0]] + [round(v, ROUND) for v in f[1:]] for f in frames]


def build_metrics() -> dict:
    res = train()
    theta = res["theta"]
    pred = res["pred"]
    labels = [int(v[0]) for v in LABELS]
    correct = int(sum(1 for p, t in zip(pred, labels) if (p >= 0.5) == (t == 1)))
    margins = [abs(float(p) - 0.5) for p in pred]

    theta_init = np.array([W_INIT[0][0], W_INIT[0][1], W_INIT[1][0], W_INIT[1][1],
                           B_INIT[0], B_INIT[1], V_INIT[0][0], V_INIT[1][0], BO_INIT[0]],
                          dtype=DTYPE)
    y = np.array(LABELS, dtype=DTYPE)
    _, prob_init = forward(theta_init, np.array(SAMPLES, dtype=DTYPE))

    return {
        "meta": {
            "generated_by": "assignment-01-xor-mlp/code/run_and_log.py",
            "env": "data/env.txt",
            "epochs": EPOCHS,
            "lr": LR,
            "betas": list(BETAS),
            "eps": EPS,
            "seed": SEED,
            "dtype": "float64",
            "optimizer": "Adam",
            "loss": "BCE(mean)",
            "batch": "full(4 samples)",
            "param_order": PARAM_ORDER,
            "frame_layout": FRAME_LAYOUT,
        },
        "samples": SAMPLES,
        "labels": labels,
        "init": {
            "W": W_INIT,
            "bh": B_INIT,
            "v": V_INIT,
            "bo": BO_INIT[0],
            "theta": [float(v) for v in theta_init],
            "loss": round(bce(prob_init, y), ROUND),
            "pred": [round(float(v), ROUND) for v in prob_init.ravel()],
        },
        "history": res["history"],
        "frames": rounded_frames(res["frames"]),
        "final": {
            "loss": round(res["loss"], ROUND),
            "theta": [round(float(v), ROUND) for v in theta],
            "pred": [round(float(v), ROUND) for v in pred],
            "hidden": [[round(float(v), ROUND) for v in row] for row in res["hidden"]],
            "correct": correct,
            "min_margin": round(min(margins), ROUND),
        },
        "config": {
            "samples": 4,
            "hidden_units": 2,
            "parameters": 9,
            "epochs": EPOCHS,
            "frames": len(res["frames"]),
            "history_points": len(res["history"]),
        },
    }


def main(argv) -> int:
    root = Path(__file__).resolve().parents[1]
    out = Path(argv[1]).resolve() if len(argv) > 1 else root / "data" / "metrics.json"

    metrics = build_metrics()
    frames = metrics["frames"]
    assert len(frames) == EPOCHS + 1, f"帧数应为 {EPOCHS + 1}，实际 {len(frames)}"
    assert all(len(f) == 32 for f in frames), "每帧必须是 32 个数"
    assert metrics["final"]["correct"] == 4

    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(metrics, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8")

    print(f"写入 {out}")
    print(f"帧数 {len(frames)} · 每帧 32 个数 · 参数 9 · 梯度 9")
    print(f"loss {frames[0][1]:.4f} -> {metrics['final']['loss']:.4f}")
    print("最终参数 " + "  ".join(f"{k}={v:.4f}" for k, v in zip(PARAM_ORDER, metrics["final"]["theta"])))
    print("最终概率 " + "  ".join(f"{v:.4f}" for v in metrics["final"]["pred"]))
    print("判对 4/4 · 最小判对边界 {:.4f}".format(metrics["final"]["min_margin"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
