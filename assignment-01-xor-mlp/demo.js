/* MLP 训练回放：用 data/metrics.json 里逐轮记录的参数重放 300 轮训练。
   左侧画输入空间里的判定区域，右侧画四个样本在隐藏空间中的位置。 */

(function () {
  const dataEl = document.getElementById("demo-data");
  if (!dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const frames = data.frames;
  const samples = data.samples;
  const labels = data.labels;

  const C0 = "#185FA5";
  const C1 = "#BA7517";
  const INK = "#2C2C2A";
  const MUTED = "#5F5E5A";
  const SIZE = 640;
  const PAD = 70;

  const cvIn = document.getElementById("c-input");
  const cvHi = document.getElementById("c-hidden");
  const playBtn = document.getElementById("demo-play");
  const resetBtn = document.getElementById("demo-reset");
  const range = document.getElementById("demo-range");
  const readout = document.getElementById("demo-readout");

  const ctxIn = cvIn.getContext("2d");
  const ctxHi = cvHi.getContext("2d");

  const N = 64;
  const off = document.createElement("canvas");
  off.width = N;
  off.height = N;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(N, N);

  const sig = (z) => 1 / (1 + Math.exp(-z));

  function params(f) {
    const p = f.p;
    return {
      W00: p[0], W01: p[1], W10: p[2], W11: p[3],
      b0: p[4], b1: p[5], v0: p[6], v1: p[7], bo: p[8],
    };
  }

  function forward(x1, x2, q) {
    const h1 = sig(q.W00 * x1 + q.W10 * x2 + q.b0);
    const h2 = sig(q.W01 * x1 + q.W11 * x2 + q.b1);
    const z = q.v0 * h1 + q.v1 * h2 + q.bo;
    return { h1, h2, p: sig(z) };
  }

  function px(v) {
    return PAD + v * (SIZE - PAD * 2);
  }
  function py(v) {
    return SIZE - PAD - v * (SIZE - PAD * 2);
  }

  function axes(ctx, title) {
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.strokeStyle = "#E3E1DA";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, SIZE - PAD * 2, SIZE - PAD * 2);
    ctx.fillStyle = MUTED;
    ctx.font = "19px system-ui, sans-serif";
    ctx.textAlign = "center";
    for (let k = 0; k <= 1; k += 0.5) {
      ctx.fillText(k.toFixed(1), px(k), SIZE - PAD + 28);
      ctx.textAlign = "right";
      ctx.fillText(k.toFixed(1), PAD - 12, py(k) + 6);
      ctx.textAlign = "center";
    }
    ctx.fillStyle = INK;
    ctx.font = "22px system-ui, sans-serif";
    ctx.fillText(title, SIZE / 2, 34);
  }

  function drawField(q) {
    const d = img.data;
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const x1 = i / (N - 1);
        const x2 = 1 - j / (N - 1);
        const p = forward(x1, x2, q).p;
        const a = 0.12 + Math.min(1, Math.abs(2 * p - 1)) * 0.55;
        const base = p >= 0.5 ? [186, 117, 23] : [24, 95, 165];
        const k = (j * N + i) * 4;
        d[k] = Math.round(255 + (base[0] - 255) * a);
        d[k + 1] = Math.round(255 + (base[1] - 255) * a);
        d[k + 2] = Math.round(255 + (base[2] - 255) * a);
        d[k + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);
    ctxIn.imageSmoothingEnabled = true;
    ctxIn.drawImage(off, PAD, PAD, SIZE - PAD * 2, SIZE - PAD * 2);
  }

  function drawGrouped(ctx, pts) {
    const groups = new Map();
    pts.forEach((pt) => {
      const key = pt.x.toFixed(1) + "," + pt.y.toFixed(1);
      if (!groups.has(key)) {
        groups.set(key, { x: pt.x, y: pt.y, y0: pt.y0, names: [] });
      }
      groups.get(key).names.push(pt.name);
    });
    groups.forEach((g) => {
      ctx.beginPath();
      ctx.arc(px(g.x), py(g.y), 11, 0, Math.PI * 2);
      ctx.fillStyle = g.y0 === 0 ? C0 : C1;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.fillStyle = g.y0 === 0 ? C0 : C1;
      ctx.font = "21px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(g.names.join("/"), px(g.x) + 14, py(g.y) - 14);
    });
  }

  function render(idx) {
    const f = frames[idx];
    const q = params(f);

    axes(ctxIn, "输入空间：判定区域随训练变化");
    drawField(q);
    drawGrouped(
      ctxIn,
      samples.map(([a, b], i) => ({ x: a, y: b, y0: labels[i], name: String(a) + String(b) }))
    );

    axes(ctxHi, "隐藏空间：四个点被搬到哪里");
    if (Math.abs(q.v1) > 1e-4) {
      const h1a = -0.2;
      const h1b = 1.2;
      const h2a = -(q.v0 * h1a + q.bo) / q.v1;
      const h2b = -(q.v0 * h1b + q.bo) / q.v1;
      ctxHi.beginPath();
      ctxHi.moveTo(px(h1a), py(h2a));
      ctxHi.lineTo(px(h1b), py(h2b));
      ctxHi.strokeStyle = "#0F6E56";
      ctxHi.lineWidth = 2.5;
      ctxHi.stroke();
    }
    drawGrouped(
      ctxHi,
      samples.map(([a, b], i) => {
        const r = forward(a, b, q);
        return { x: r.h1, y: r.h2, y0: labels[i], name: String(a) + String(b) };
      })
    );

    const probs = samples.map(([a, b]) => forward(a, b, q).p);
    const preds = probs.map((p) => (p >= 0.5 ? 1 : 0));
    readout.textContent =
      "第 " + f.e + " 轮 · 损失 " + f.l.toFixed(4) +
      " · 预测 " + preds.join(",") +
      " · 概率 " + probs.map((p) => p.toFixed(2)).join(" / ");
    if (Number(range.value) !== idx) range.value = idx;
  }

  let cur = 0;
  let playing = true;
  let last = 0;
  const MS_PER_FRAME = 26;

  const jump = new URLSearchParams(location.search).get("frame");
  if (jump !== null && !isNaN(Number(jump))) {
    cur = Math.max(0, Math.min(Number(jump), frames.length - 1));
    playing = false;
    playBtn.textContent = "继续";
  }

  function tick(ts) {
    if (playing) {
      if (!last) last = ts;
      const dt = ts - last;
      if (dt >= MS_PER_FRAME) {
        const step = Math.floor(dt / MS_PER_FRAME);
        last = ts;
        cur = Math.min(cur + step, frames.length - 1);
        render(cur);
        if (cur >= frames.length - 1) {
          playing = false;
          playBtn.textContent = "重放";
        }
      }
    }
    requestAnimationFrame(tick);
  }

  playBtn.addEventListener("click", () => {
    if (cur >= frames.length - 1) {
      cur = 0;
      playing = true;
      playBtn.textContent = "暂停";
      render(cur);
      return;
    }
    playing = !playing;
    last = 0;
    playBtn.textContent = playing ? "暂停" : "继续";
  });

  resetBtn.addEventListener("click", () => {
    cur = 0;
    last = 0;
    playing = true;
    playBtn.textContent = "暂停";
    render(0);
  });

  range.addEventListener("input", () => {
    cur = Number(range.value);
    playing = false;
    playBtn.textContent = "继续";
    render(cur);
  });

  render(cur);
  requestAnimationFrame(tick);
})();
