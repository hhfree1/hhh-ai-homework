# MLP 学习参数的实现与可视化 · 2-2-1 网络学 XOR

**在线页面**：https://hhfree1.github.io/hhh-ai-homework/assignment-01-xor-mlp/

返回作业目录：[../](../)

页面里包含一段**动态演示**：把 300 轮训练的每一轮参数记录下来重放，可以拖动进度条停在任意一轮。
演示下方还有两块跟着进度条同步变化的面板：左侧 2-2-1 结构图（线宽=权重绝对值、颜色=正负、节点亮度=激活），
右侧是样本 `01` 的逐节点算式（h1、h2、logit、概率四组数字）。

课程《人工智能通识》第二篇 15 节「人工神经元网络」，对应讲义第 66 页「多层网络怎样学习参数？Python 实现与反向传播」。
把该页给出的 XOR 示例**真实跑一遍**，记录每一轮的参数，把学习过程做成可回放的图表，部署到 GitHub Pages。

讲义该节未附作业提交格式与评分标准，本仓库按「真实运行 + 逐轮可追溯 + 数字可复算」三条组织。

## 结论（全部来自实际运行）

| 输入 | 正确答案 | h1 | h2 | 概率 | 预测 |
|---|---|---|---|---|---|
| 0 0 | 0 | 0.030 | 0.030 | 0.0287 | 0 |
| 0 1 | 1 | 0.000 | 0.935 | 0.9745 | 1 |
| 1 0 | 1 | 0.935 | 0.000 | 0.9745 | 1 |
| 1 1 | 0 | 0.034 | 0.034 | 0.0305 | 0 |

- 损失：第 0 轮 `0.5620` → 第 300 轮 `0.0279`
- 四个样本全部分对，与课程脚本输出的 `All four XOR samples classified correctly.` 一致
- 补充实验：把初值换成随机数重复 20 次，**9 次**完全分对（45%）

## 复现

```bash
git clone https://github.com/hhfree1/hhh-ai-homework.git
cd hhh-ai-homework/assignment-01-xor-mlp
pip install torch numpy matplotlib

python code/xor_pytorch.py    # 官方脚本原样运行（对照用）
python code/run_and_log.py    # 同上逻辑 + 逐轮记录参数 → data/metrics.json
python code/random_init.py    # 随机初值实验 → data/random_init.json
python code/make_figures.py   # 由 json 生成 figures/ 下的静态图
python code/build_page.py     # 由全部 json 生成 index.html（最后运行）
```

环境：Python 3.13.14 + PyTorch 2.15.0+cpu，Windows 11，CPU 训练约 7 秒（详见 `data/env.txt`）。
TensorFlow 版本未运行（本机未安装），`code/xor_tensorflow.py` 仅作对照保留。

## 目录

```
code/    官方脚本 2 个 + 记录/绘图/建页脚本 4 个
data/    metrics.json（301 帧逐轮参数 + 32 条每 10 轮快照）、final_predictions.csv、random_init.json、env.txt
figures/ 损失曲线、参数轨迹、隐藏层激活、隐藏层空间、概率条形图
index.html  GitHub Pages 入口
```

## 固定超参（与第 66 页一致，未修改）

2 输入 → 2 个 sigmoid 隐藏单元 → 1 输出；四个样本整批；二元交叉熵（吃 logits）；
Adam，学习率 0.03；300 轮；float64；阈值 0.5。

## 这个实验不能证明什么

1. **不能证明泛化**：只有 4 个样本、没有测试集，拟合的就是这 4 个点。
2. **不能证明随机初值一定收敛**：官方初值已接近答案；换成随机初值后 20 次里只有 9 次成功，其余卡在局部极小（损失约 0.35～0.49）。
3. **不能证明多层网络万能**：通用逼近定理只保证「存在这样一组参数」，不保证梯度下降能找到。

## 代码来源

课程页 `assets/xor-code/` 下的 `xor_pytorch.py` 与 `xor_tensorflow.py`，原样保留、未改逻辑。
`run_and_log.py` 在其基础上增加日志与导出，计算部分完全一致。
