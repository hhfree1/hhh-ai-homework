"""在官方 xor_pytorch.py 的基础上加日志：逐轮帧 + 每 10 轮快照。

与官方脚本的差别只在记录，不在计算：
  - 训练前记录一次初始状态（epoch = 0）
  - 每轮更新完参数后，用新参数重新前向一次，记录 9 个参数与对应的 loss
    （早期版本记的是更新前的 loss，会和参数错开一轮，已修正）
  - 每 10 轮额外记录一份完整快照（含隐藏层输出与概率）
  - 结束后写出 data/metrics.json 与 data/final_predictions.csv

初值、损失、优化器、轮数与官方脚本完全一致。
"""

import json
import os

import torch
import torch.nn.functional as F

torch.manual_seed(7)
torch.set_default_dtype(torch.float64)

x = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=torch.float64,
)
target = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float64)

W = torch.nn.Parameter(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))
b_hidden = torch.nn.Parameter(torch.tensor([-1.0, -1.0]))
v = torch.nn.Parameter(torch.tensor([[3.0], [3.0]]))
b_output = torch.nn.Parameter(torch.tensor([-2.5]))

parameters = [W, b_hidden, v, b_output]
optimizer = torch.optim.Adam(parameters, lr=0.03)


def snapshot(epoch, loss_value=None):
    with torch.no_grad():
        hidden = torch.sigmoid(x @ W + b_hidden)
        logits = hidden @ v + b_output
        loss = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
    return {
        "epoch": epoch,
        "loss": round(float(loss.detach()), 6) if loss_value is None else round(float(loss_value), 6),
        "hidden": [[round(float(a), 6) for a in row] for row in hidden.tolist()],
        "prob": [round(float(p), 6) for p in probs.flatten().tolist()],
        "pred": [int(p) for p in (probs >= 0.5).int().flatten().tolist()],
        "W": [[round(float(a), 6) for a in row] for row in W.tolist()],
        "b_hidden": [round(float(a), 6) for a in b_hidden.tolist()],
        "v": [round(float(a), 6) for a in v.flatten().tolist()],
        "b_output": round(float(b_output), 6),
    }


def compact(epoch, loss_value):
    """给前端动画用的极简帧：九个参数 + 与这批参数对应的损失。"""
    return {
        "e": epoch,
        "l": round(float(loss_value), 5),
        "p": [
            round(float(W[0][0]), 4), round(float(W[0][1]), 4),
            round(float(W[1][0]), 4), round(float(W[1][1]), 4),
            round(float(b_hidden[0]), 4), round(float(b_hidden[1]), 4),
            round(float(v[0][0]), 4), round(float(v[1][0]), 4),
            round(float(b_output), 4),
        ],
    }


history = [snapshot(0)]
frames = [compact(0, history[0]["loss"])]

for epoch in range(1, 301):
    optimizer.zero_grad(set_to_none=True)
    hidden = torch.sigmoid(x @ W + b_hidden)
    logits = hidden @ v + b_output
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    optimizer.step()

    # 参数已经变了，帧里要配"更新之后"的损失，否则读数和参数差一轮
    with torch.no_grad():
        hidden_after = torch.sigmoid(x @ W + b_hidden)
        logits_after = hidden_after @ v + b_output
        loss_after = F.binary_cross_entropy_with_logits(logits_after, target)
    frames.append(compact(epoch, loss_after))

    if epoch % 10 == 0 or epoch == 1:
        history.append(snapshot(epoch))

final = history[-1]
samples = [[int(a), int(b)] for a, b in x.tolist()]
labels = [int(t) for t in target.flatten().tolist()]

os.makedirs("data", exist_ok=True)
with open("data/metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "samples": samples,
            "labels": labels,
            "history": history,
            "frames": frames,
            "config": {
                "structure": "2-2-1",
                "optimizer": "Adam",
                "lr": 0.03,
                "epochs": 300,
                "loss": "binary_cross_entropy_with_logits",
                "dtype": "float64",
                "seed": 7,
                "threshold": 0.5,
                "framework": "pytorch " + torch.__version__,
            },
        },
        f,
        ensure_ascii=False,
        indent=1,
    )

with open("data/final_predictions.csv", "w", encoding="utf-8", newline="") as f:
    f.write("x1,x2,target,h1,h2,probability,prediction,correct\n")
    for (a, b), y, (h1, h2), p, pred in zip(
        samples, labels, final["hidden"], final["prob"], final["pred"]
    ):
        f.write(f"{a},{b},{y},{h1:.4f},{h2:.4f},{p:.4f},{pred},{int(pred == y)}\n")

print(f"epoch={300}, loss={final['loss']:.4f}")
print("x1 x2 | target | hidden (h1, h2) | probability | predicted")
for (a, b), y, (h1, h2), p, pred in zip(
    samples, labels, final["hidden"], final["prob"], final["pred"]
):
    print(f"{a}  {b}  |   {y}    |    ({h1:.3f}, {h2:.3f})    |    {p:.4f}    |     {pred}")

if final["pred"] == labels:
    print("All four XOR samples classified correctly.")
else:
    raise SystemExit("XOR was not fully classified; try more epochs.")
