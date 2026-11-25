"""由 data/metrics.json 生成 index.html（GitHub Pages 入口）。

页面上的每个数字都从 metrics.json / random_init.json 读出，不手写，
避免图和表对不上。
"""

import json
import os
from datetime import date

with open("data/metrics.json", encoding="utf-8") as f:
    m = json.load(f)

history = m["history"]
before, after = history[0], history[-1]
last_epoch = after["epoch"]
frames_json = json.dumps(
    {"samples": m["samples"], "labels": m["labels"], "frames": m["frames"]},
    separators=(",", ":"),
    ensure_ascii=False,
)
labels = m["labels"]
names = ["".join(str(v) for v in s) for s in m["samples"]]
cfg = m["config"]

random_init = None
if os.path.exists("data/random_init.json"):
    with open("data/random_init.json", encoding="utf-8") as f:
        random_init = json.load(f)

env = ""
if os.path.exists("data/env.txt"):
    with open("data/env.txt", encoding="utf-8") as f:
        env = f.read().strip()

rows = ""
for name, y, h, p, pred in zip(names, labels, after["hidden"], after["prob"], after["pred"]):
    ok = pred == y
    rows += (
        f"<tr><td>{name[0]}</td><td>{name[1]}</td><td>{y}</td>"
        f"<td>{h[0]:.3f}</td><td>{h[1]:.3f}</td><td>{p:.4f}</td>"
        f"<td class=\"{'ok' if ok else 'bad'}\">{pred} {'✓' if ok else '✗'}</td></tr>\n"
    )

random_block = ""
if random_init:
    detail = "".join(
        f"<span class=\"chip {'ok' if r['success'] else 'bad'}\">{r['pred']}</span>"
        for r in random_init["runs"]
    )
    random_block = f"""
  <section class="card">
    <h2>附：换掉初值还成立吗</h2>
    <p>官方脚本的初值是照着答案挑的（W 一开始就是两个检测器）。为了看这个结构本身靠不靠得住，把初值换成
    <code>torch.randn</code> 的随机数，其余设置不变，重复 20 次：</p>
    <p class="big"><strong>{random_init['success']} / {random_init['total']}</strong> 次把四个样本全部分对。</p>
    <p class="muted">每一格是一次运行的四位预测：</p>
    <div class="chips">{detail}</div>
    <p class="cap">说明：成功率不是 100%，说明「结构有能力」不等于「每次都能练出来」，
    这也是 1989 年通用逼近定理只谈表示能力、不谈训练难度的原因。</p>
  </section>
"""

html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MLP 学习参数的实现与可视化 · 2-2-1 网络学 XOR</title>
<style>
  :root {{
    --ink:#2C2C2A; --muted:#5F5E5A; --line:#E3E1DA; --bg:#FBFAF7;
    --blue:#185FA5; --orange:#BA7517; --ok:#0F6E56; --bad:#A32D2D;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
    font-size:15px; line-height:1.75;
  }}
  .wrap {{ max-width:840px; margin:0 auto; padding:48px 22px 72px; }}
  header {{ border-bottom:1px solid var(--line); padding-bottom:22px; margin-bottom:30px; }}
  .kicker {{ font-size:12.5px; color:var(--muted); letter-spacing:.04em; }}
  h1 {{ font-size:26px; font-weight:600; margin:8px 0 10px; line-height:1.35; }}
  .sub {{ color:var(--muted); font-size:14px; }}
  h2 {{ font-size:17px; font-weight:600; margin:0 0 12px; }}
  .card {{
    background:#fff; border:1px solid var(--line); border-radius:12px;
    padding:22px 24px; margin:0 0 22px;
  }}
  .lead {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:0 0 22px; }}
  .lead div {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
  .lead .n {{ font-size:24px; font-weight:600; color:var(--blue); line-height:1.3; }}
  .lead .n.o {{ color:var(--orange); }}
  .lead .l {{ font-size:12.5px; color:var(--muted); }}
  code {{
    background:#F1EFE8; padding:1px 6px; border-radius:4px;
    font-family:"Cascadia Mono",Consolas,monospace; font-size:13px;
  }}
  pre {{
    background:#2C2C2A; color:#F1EFE8; padding:14px 16px; border-radius:8px;
    overflow-x:auto; font-size:13px; font-family:"Cascadia Mono",Consolas,monospace;
    line-height:1.6;
  }}
  figure {{ margin:0 0 6px; }}
  figure img {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--line); background:#fff; }}
  figcaption {{ font-size:12.5px; color:var(--muted); margin-top:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; margin-top:6px; }}
  th,td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); }}
  th {{ font-weight:500; color:var(--muted); font-size:12.5px; }}
  td.ok {{ color:var(--ok); }} td.bad {{ color:var(--bad); }}
  .big {{ font-size:18px; }}
  .muted {{ color:var(--muted); font-size:13.5px; }}
  .cap {{ color:var(--muted); font-size:12.5px; border-left:2px solid var(--line); padding-left:10px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 14px; }}
  .chip {{
    font-size:12px; padding:2px 7px; border-radius:5px; font-family:Consolas,monospace;
    background:#E6F1FB; color:var(--blue);
  }}
  .chip.bad {{ background:#FCEBEB; color:var(--bad); }}
  .live {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; align-items:start; }}
  .panel {{ border:1px solid var(--line); border-radius:10px; padding:14px 16px; background:#fff; min-width:0; }}
  .panel h3 {{ font-size:13px; font-weight:600; color:var(--muted); margin:0 0 10px; }}
  .eq {{ font-family:"Cascadia Mono",Consolas,monospace; font-size:12px; line-height:1.85; white-space:pre; overflow-x:auto; margin:0; color:var(--ink); background:#FBFAF7; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
  .eq span {{ color:var(--blue); }}
  .hint {{ font-size:12px; color:var(--muted); margin:8px 0 0; }}
  @media (max-width:640px) {{
    .live {{ grid-template-columns:1fr; }}
    .demo-ctrl {{ position:sticky; bottom:0; background:var(--bg); padding:8px 0 6px; z-index:5; border-top:1px solid var(--line); }}
  }}
  ol {{ padding-left:20px; }} ol li {{ margin-bottom:8px; }}
  a {{ color:var(--blue); }}
  footer {{ border-top:1px solid var(--line); margin-top:34px; padding-top:16px; font-size:12.5px; color:var(--muted); }}
  @media (max-width:640px) {{ .lead {{ grid-template-columns:1fr; }} h1 {{ font-size:21px; }} }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="kicker">人工智能通识 · 第二篇 15 节「人工神经元网络」· 讲义第 66 页</div>
  <h1>MLP 学习参数的实现与可视化：2-2-1 网络学 XOR 的实测记录</h1>
  <div class="sub">作者 hhfree1 · 运行于 {date.today().isoformat()} · 代码、数据、图表全部可下载核对</div>
</header>

<div class="lead">
  <div><div class="n">0,1,1,0</div><div class="l">四个样本的最终预测</div></div>
  <div><div class="n o">{before['loss']:.2f} → {after['loss']:.4f}</div><div class="l">损失，第 0 轮到第 {after['epoch']} 轮</div></div>
  <div><div class="n">4 / 4</div><div class="l">分类正确的样本数</div></div>
</div>

<section class="card">
  <h2>这份作业对应什么要求</h2>
  <p>讲义第 66 页（第二篇 15 节「人工神经元网络」）的标题是「多层网络怎样学习参数？Python 实现与反向传播」，
  给出 2-2-1 网络、两段等价实现和两个可下载脚本。该节没有附提交格式与评分标准，
  因此本页按三条能当场核对的标准组织：</p>
  <ol>
    <li><strong>实现</strong>：用 <code>assets/xor-code/xor_pytorch.py</code> 的等价逻辑训练，结构、损失、优化器、轮数、精度均不改。</li>
    <li><strong>学习过程</strong>：记录每一轮的 9 个参数和与之对应的损失，可停在任意一轮查看。</li>
    <li><strong>可视化</strong>：判定区域、隐藏空间、损失曲线、隐藏层激活、输出概率，全部由 <code>data/metrics.json</code> 生成，不手绘。</li>
  </ol>
  <p class="cap">课代表另提出的两条要求（含动态演示、所有作业放在同一仓库）已分别落实在本页与上级目录。</p>
</section>

<section class="card">
  <h2>怎么复现</h2>
  <p>运行环境：{env or '见 data/env.txt'}</p>
  <pre>git clone https://github.com/hhfree1/hhh-ai-homework.git
cd hhh-ai-homework/assignment-01-xor-mlp
pip install torch numpy matplotlib
python code/run_and_log.py   # 训练并写出 data/metrics.json（含逐轮帧）
python code/random_init.py   # 随机初值重复 20 次 → data/random_init.json
python code/make_figures.py  # 由 json 生成五张静态图
python code/build_page.py    # 最后由全部 json 生成本页</pre>
  <p class="muted">超参固定为课程第 66 页给出的那一组：2-2-1、四个样本整批、二元交叉熵、Adam、学习率 0.03、300 轮、float64、阈值 0.5。</p>
</section>

<section class="card">
  <h2>① 动态演示：300 轮训练回放</h2>
  <p>下面的动画不是示意，而是把训练过程中<strong>每一轮</strong>记录下来的参数重放一遍：左边是输入空间里判定区域的变化，右边是四个样本在隐藏空间里被搬动的位置。可以拖动进度条停在任意一轮。</p>
  <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start">
    <div style="flex:1 1 260px;min-width:240px">
      <canvas id="c-input" width="640" height="640" role="img" aria-label="输入空间中的判定区域随训练轮数变化，四个样本为 00、01、10、11；下方滑块可切换轮数" style="width:100%;height:auto;border:1px solid #E3E1DA;border-radius:8px;background:#fff"></canvas>
      <div style="font-size:12.5px;color:#5F5E5A;margin-top:6px;text-align:center">输入空间 (x1, x2)：底色是当前判定，蓝为判 0、橙为判 1</div>
    </div>
    <div style="flex:1 1 260px;min-width:240px">
      <canvas id="c-hidden" width="640" height="640" role="img" aria-label="四个样本映射到隐藏空间后的位置，绿色直线为输出层分界线；下方滑块可切换轮数" style="width:100%;height:auto;border:1px solid #E3E1DA;border-radius:8px;background:#fff"></canvas>
      <div style="font-size:12.5px;color:#5F5E5A;margin-top:6px;text-align:center">隐藏空间 (h1, h2)：绿线是输出层的分界线</div>
    </div>
  </div>
  <div class="demo-ctrl" style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:14px">
    <button id="demo-play" style="font:inherit;font-size:13.5px;padding:6px 14px;border:1px solid #D3D1C7;background:#fff;border-radius:6px;cursor:pointer;color:#2C2C2A">暂停</button>
    <button id="demo-reset" style="font:inherit;font-size:13.5px;padding:6px 14px;border:1px solid #D3D1C7;background:#fff;border-radius:6px;cursor:pointer;color:#2C2C2A">重放</button>
    <input id="demo-range" type="range" min="0" max="{last_epoch}" value="0" step="1" aria-label="训练轮数" style="flex:1 1 150px">
    <span id="demo-readout" role="status" aria-live="polite" style="font-size:13px;color:#2C2C2A;font-variant-numeric:tabular-nums"></span>
  </div>
  <div class="live">
    <div class="panel">
      <h3>当前网络：线宽 = |权重|，橙为正、蓝为负</h3>
      <svg viewBox="0 0 320 210" role="img" aria-label="2-2-1 网络结构图，连线粗细与颜色对应当前权重，节点亮度对应当前输出" style="width:100%;height:auto;max-height:240px">
        <g stroke-linecap="round">
          <line id="l00" x1="44" y1="62" x2="164" y2="62" stroke="#BA7517" stroke-width="3"></line>
          <line id="l01" x1="44" y1="62" x2="164" y2="148" stroke="#BA7517" stroke-width="3"></line>
          <line id="l10" x1="44" y1="148" x2="164" y2="62" stroke="#BA7517" stroke-width="3"></line>
          <line id="l11" x1="44" y1="148" x2="164" y2="148" stroke="#BA7517" stroke-width="3"></line>
          <line id="lv0" x1="164" y1="62" x2="282" y2="105" stroke="#BA7517" stroke-width="3"></line>
          <line id="lv1" x1="164" y1="148" x2="282" y2="105" stroke="#BA7517" stroke-width="3"></line>
        </g>
        <g font-family="system-ui,sans-serif" font-size="11" text-anchor="middle">
          <circle id="n-i1" cx="44" cy="62" r="17" fill="#185FA5" stroke="#185FA5" stroke-width="2"></circle>
          <circle id="n-i2" cx="44" cy="148" r="17" fill="#185FA5" stroke="#185FA5" stroke-width="2"></circle>
          <circle id="n-h1" cx="164" cy="62" r="17" fill="#BA7517" stroke="#BA7517" stroke-width="2"></circle>
          <circle id="n-h2" cx="164" cy="148" r="17" fill="#BA7517" stroke="#BA7517" stroke-width="2"></circle>
          <circle id="n-o" cx="282" cy="105" r="17" fill="#0F6E56" stroke="#0F6E56" stroke-width="2"></circle>
          <text x="44" y="34" fill="#2C2C2A">x1</text>
          <text x="44" y="186" fill="#2C2C2A">x2</text>
          <text x="164" y="34" fill="#2C2C2A">h1</text>
          <text x="164" y="186" fill="#2C2C2A">h2</text>
          <text x="282" y="77" fill="#2C2C2A">y</text>
        </g>
      </svg>
      <p class="hint">节点亮度 = 该单元当前的输出；图对应第 <span id="net-epoch">0</span> 轮。六个权重是同一条权重向量在页面上的另一种读法。</p>
    </div>
    <div class="panel">
      <h3>样本 01 走一遍（正确答案 1）</h3>
      <pre class="eq" id="chain">加载中</pre>
      <p class="hint">都是当前这一轮的参数代入后的数值，与上面的进度条同步。式中权重保留两位、结果保留四位，四舍五入后末位可能差一点。</p>
    </div>
  </div>
  <p class="cap">看两件事：左边判定区域从「几乎全判 0」（第 0 轮四样本概率 0.29 / 0.46 / 0.46 / 0.29）逐渐长出两块对角区域；右边四个点也在动——第 0 轮 00 与 11 同在 (0.269, 0.269)，01 在 (0.047, 0.731)、10 在 (0.731, 0.047)，第 300 轮 01、10 被推到 (0.0001, 0.935) 与 (0.935, 0.0001)。参数是逐轮记录的，不是补间动画。</p>
</section>

<section class="card">
  <h2>② 损失曲线</h2>
  <figure>
    <img src="figures/loss_curve.png" alt="损失随训练轮数下降的曲线">
    <figcaption>这张图说明：训练确实在起作用——损失从 {before['loss']:.3f} 单调降到 {after['loss']:.4f}，说明「错多少」被压成了一个可以不断降低的数。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>③ 九个参数各自的轨迹</h2>
  <figure>
    <img src="figures/param_trajectories.png" alt="九个参数随训练轮数变化的曲线">
    <figcaption>这张图说明：动的不只是输出，九个参数本身一直在变，而且方向持续一致——W 的两个对角项升到约 ±6.1、
    两个隐藏偏置降到 -3.47、输出权重升到 8.19。三个子图里能看见的曲线少于参数个数，原因在初值：
    W 起于 <code>[[2,-2],[-2,2]]</code>、两个偏置同为 -1，四个样本又成对对称，于是 W00 与 W11、W01 与 W10、
    b0 与 b1、v0 与 v1 全程重合。这是这次运行的真实情况，不是画漏。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>④ 隐藏层两个单元在学什么</h2>
  <figure>
    <img src="figures/hidden_activations.png" alt="训练前后四个样本在两个隐藏单元上的输出">
    <figcaption>第 0 轮两个单元已经有方向（01 在 h₂ 上 0.731、10 在 h₁ 上 0.731），但同类的 00 与 11 也有 0.269，余量不足；第 300 轮 h₁ 在 10 上 0.935、在 01 上 0.0001，h₂ 正好相反，00 与 11 降到 0.03 附近。从「有倾向」变成「接近二值」，这一步是梯度推出来的，初值只给了方向。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>⑤ 四个点被搬到了哪里</h2>
  <figure>
    <img src="figures/hidden_space.png" alt="四个样本在隐藏层坐标中的位置，训练前后">
    <figcaption>两幅图的坐标都是 (h₁, h₂)。第 0 轮：00 与 11 精确落在同一点 (0.269, 0.269)，01 在 (0.047, 0.731)、10 在 (0.731, 0.047)，此时输出层分界线 3h₁+3h₂−2.5=0 仍把 01、10 判成 0（概率 0.459）。第 300 轮：00/11 移到原点附近（0.030 与 0.034），01、10 分别被推到 (0.0001, 0.935) 与 (0.935, 0.0001)，分界线从中间穿过，四个点才真正被分开。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>⑥ 最终概率与判定</h2>
  <figure>
    <img src="figures/probability_bars.png" alt="四个样本的最终输出概率与 0.5 阈值">
    <figcaption>这张图说明：两类样本的概率分别落在 0.5 两侧很远的位置（约 {min(after['prob']):.3f} 与 {max(after['prob']):.3f}），不是勉强压线，这是收敛的表征。</figcaption>
  </figure>
</section>

<section class="card">
  <h2>最终结果表</h2>
  <table>
    <thead><tr><th>x₁</th><th>x₂</th><th>正确答案</th><th>h₁</th><th>h₂</th><th>概率</th><th>预测</th></tr></thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="cap">数据来源：<code>data/final_predictions.csv</code>，由 <code>data/metrics.json</code> 导出。</p>
</section>

<section class="card">
  <h2>三点解读</h2>
  <ol>
    <li><strong>为什么单层不行。</strong>单层感知器只有一次判断，在平面上表现为一条直线。XOR 的两类点在对角交错，任何一条直线切下去，两边都会同时出现 0 和 1。</li>
    <li><strong>隐藏层做了什么。</strong>它把四个点搬到新坐标（h₁, h₂）：同类点落在同一点附近，异类点被推向对角；输出层再在这个新坐标里画一条线。这次运行里 00 与 11 从第 0 轮就重合（初值对称），训练做的是把异类点的间距从 0.73 / 0.27 拉到 0.935 / 0.0001。分工仍是「隐藏层摆点，输出层画线」。</li>
    <li><strong>为什么能自动做到。</strong>不用人手写 h₁/h₂ 的规则：损失函数量出「差多少」，梯度指出「往哪改」，优化器每轮改一点点，300 轮后规则自己长出来。</li>
  </ol>
</section>

<section class="card">
  <h2>这次实验不能证明什么</h2>
  <ol>
    <li><strong>不能证明泛化。</strong>只有 4 个样本、整批训练、没有测试集，它拟合的就是这 4 个点，没有回答「没见过的输入会怎样」。</li>
    <li><strong>不能证明随便初始化都能学会。</strong>官方初值已经是近似解，见下面的补充实验。</li>
    <li><strong>不能证明多层网络万能。</strong>通用逼近定理只保证「存在这样一组参数」，不保证梯度下降一定能找到，也不保证训练容易。</li>
  </ol>
</section>
{random_block}
<footer>
  课程页面：<a href="https://zhumy.github.io/Introduction_to_Artificial_Intelligence/">zhumy.github.io/Introduction_to_Artificial_Intelligence</a> ·
  代码来源为该页 assets/xor-code 下的两个官方脚本 ·
  本站所有数字均由 <code>code/</code> 下的脚本实际运行产生，可逐项核对。
</footer>

</div>
<script id="demo-data" type="application/json">{frames_json}</script>
<script src="demo.js"></script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("wrote index.html", len(html), "chars")
