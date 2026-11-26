/* XOR 回放台
   数据来自页面里内联的 #xor-data（由 code/build_page.py 从 data/metrics.json 注入），
   不发起任何网络请求，双击本地文件也能跑。
   帧布局 = [e, loss, q0..q3, h0..h7, theta0..8, g0..8]（与 metrics.json 的 meta.frame_layout 一致）。 */
(function () {
  "use strict";

  var node = document.getElementById("xor-data");
  if (!node) return;

  var DATA;
  try {
    DATA = JSON.parse(node.textContent);
  } catch (err) {
    return;
  }

  var F = DATA.frames || [];
  var M = DATA.meta || {};
  var SAMPLES = DATA.samples || [];
  var LABELS = DATA.labels || [];
  var N = F.length;
  if (!N) return;

  var K = 2; // 画布分辨率倍数：CSS 缩到 1/2，视网膜屏也不糊
  var COLOR = {
    bg: "#0B0D10",
    panel: "#12151A",
    text: "#E8E6E1",
    muted: "#9AA0A6",
    line: "rgba(232,230,225,0.10)",
    signal: "#3DDC97"
  };

  var $ = function (id) { return document.getElementById(id); };
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  var idx = 0;
  var timer = null;

  function frame(i) {
    var f = F[Math.max(0, Math.min(N - 1, i))];
    return {
      e: f[0],
      loss: f[1],
      q: [f[2], f[3], f[4], f[5]],
      h: [f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13]],
      t: f.slice(14, 23),
      g: f.slice(23, 32)
    };
  }

  // 线宽的参照取整段训练的最大绝对值，早期和后期的粗细才能互相比较
  var maxW = 1e-6, maxG = 1e-6;
  for (var a = 0; a < N; a++) {
    for (var b = 14; b <= 22; b++) maxW = Math.max(maxW, Math.abs(F[a][b]));
    for (var c = 23; c <= 31; c++) maxG = Math.max(maxG, Math.abs(F[a][c]));
  }

  function sig(z) { return 1 / (1 + Math.exp(-z)); }
  function f4(v) { return v.toFixed(4); }
  function f3(v) { return v.toFixed(3); }
  function signed(v, d) { return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d === undefined ? 2 : d); }
  function clamp01(v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }
  function mix(p, q, t) {
    return "rgb(" + Math.round(p[0] + (q[0] - p[0]) * t) + "," +
      Math.round(p[1] + (q[1] - p[1]) * t) + "," +
      Math.round(p[2] + (q[2] - p[2]) * t) + ")";
  }

  // ---------------- 画布尺寸：逻辑坐标 + 分辨率倍数 ----------------
  function prep(cv, w, h) {
    cv.width = w * K;
    cv.height = h * K;
    var ctx = cv.getContext("2d");
    ctx.setTransform(K, 0, 0, K, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  // ---------------- 网络图 ----------------
  function drawNet(st) {
    var cv = $("cvNet");
    if (!cv) return;
    var W = 520, H = 460;
    var ctx = prep(cv, W, H);
    var inX = 92, hidX = 262, outX = 432;
    var inY = [120, 340], hidY = [120, 340], outY = 230, R = 25;

    var Wm = [[st.t[0], st.t[1]], [st.t[2], st.t[3]]];
    var bh = [st.t[4], st.t[5]];
    var v = [st.t[6], st.t[7]];
    var bo = st.t[8];

    function edge(x1, y1, x2, y2, w, scale) {
      var t = Math.min(1, Math.abs(w) / scale);
      var dx = x2 - x1, dy = y2 - y1;
      var len = Math.sqrt(dx * dx + dy * dy) || 1;
      var ux = dx / len, uy = dy / len;
      var sx = x1 + ux * (R + 2), sy = y1 + uy * (R + 2);
      var ex = x2 - ux * (R + 2), ey = y2 - uy * (R + 2);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(ex, ey);
      ctx.lineWidth = 0.7 + 4.2 * t;
      ctx.strokeStyle = w >= 0 ? COLOR.signal : COLOR.muted;
      ctx.globalAlpha = 0.32 + 0.66 * t;
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.globalAlpha = 1;
      return { sx: sx, sy: sy, ex: ex, ey: ey, ux: ux, uy: uy };
    }

    // 标签靠输入端、离边线让开，并用底色描一圈当晕，压在交叉点上也不糊
    function tag(g, frac, off, str) {
      var mx = g.sx + (g.ex - g.sx) * frac;
      var my = g.sy + (g.ey - g.sy) * frac;
      var nx = -g.uy * off, ny = g.ux * off;
      ctx.font = "10px Consolas, ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      var w = ctx.measureText(str).width;
      ctx.fillStyle = COLOR.panel;
      ctx.fillRect(mx + nx - w / 2 - 2, my + ny - 6, w + 4, 12);
      ctx.fillStyle = COLOR.muted;
      ctx.fillText(str, mx + nx, my + ny + 0.5);
    }

    var i, j, g;
    for (i = 0; i < 2; i++) {
      for (j = 0; j < 2; j++) {
        g = edge(inX, inY[i], hidX, hidY[j], Wm[j][i], maxW);
        // 对角线（i≠j）标签往输入一侧收，横线（i=j）放中间
        tag(g, i === j ? 0.5 : 0.3, i === 0 ? -11 : 13, signed(Wm[j][i], 2));
      }
    }
    for (j = 0; j < 2; j++) {
      g = edge(hidX, hidY[j], outX, outY, v[j], maxW);
      tag(g, 0.44, j === 0 ? -12 : 14, signed(v[j], 2));
    }

    function node(x, y, label, accent) {
      ctx.beginPath();
      ctx.arc(x, y, R, 0, Math.PI * 2);
      ctx.fillStyle = COLOR.bg;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = accent ? COLOR.signal : COLOR.line;
      ctx.stroke();
      ctx.font = "11px ui-monospace,SFMono-Regular,Consolas,monospace";
      ctx.fillStyle = COLOR.text;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x, y + 0.5);
    }

    node(inX, inY[0], "x1");
    node(inX, inY[1], "x2");
    node(hidX, hidY[0], "h1");
    node(hidX, hidY[1], "h2");
    node(outX, outY, "p", true);

    ctx.font = "10px ui-monospace,SFMono-Regular,Consolas,monospace";
    ctx.fillStyle = COLOR.muted;
    ctx.textAlign = "center";
    ctx.fillText("b1 " + signed(bh[0], 2), hidX, hidY[0] + R + 15);
    ctx.fillText("b2 " + signed(bh[1], 2), hidX, hidY[1] + R + 15);
    ctx.fillText("b0 " + signed(bo, 2), outX, outY + R + 15);

    ctx.textAlign = "left";
    ctx.fillStyle = COLOR.muted;
    ctx.fillText("输入 2", 12, 22);
    ctx.fillText("隐藏 2", hidX - 20, 22);
    ctx.textAlign = "right";
    ctx.fillText("输出 1", W - 12, 22);
  }

  // ---------------- 决策边界 ----------------
  var GRID = 108;
  var off = document.createElement("canvas");
  off.width = GRID;
  off.height = GRID;

  function drawField(st) {
    var cv = $("cvField");
    if (!cv) return;
    var S = 420, PAD = 12, INNER = S - PAD * 2;
    var ctx = prep(cv, S, S);

    var t = st.t;
    var W00 = t[0], W01 = t[1], W10 = t[2], W11 = t[3], b1 = t[4], b2 = t[5], v0 = t[6], v1 = t[7], bo = t[8];
    var ictx = off.getContext("2d");
    var img = ictx.createImageData(GRID, GRID);
    var px = img.data;
    var bg = [11, 13, 16], pos = [61, 220, 151], neg = [58, 66, 78], white = [232, 230, 225];

    for (var j = 0; j < GRID; j++) {
      var x2 = 1 - j / (GRID - 1);
      for (var i = 0; i < GRID; i++) {
        var x1 = i / (GRID - 1);
        var h1 = sig(W00 * x1 + W01 * x2 + b1);
        var h2 = sig(W10 * x1 + W11 * x2 + b2);
        var p = sig(v0 * h1 + v1 * h2 + bo);
        var conf = Math.pow(Math.abs(p - 0.5) * 2, 0.6);
        var col;
        if (Math.abs(p - 0.5) < 0.008) {
          col = white;
        } else if (p >= 0.5) {
          col = [0, 0, 0];
          for (var k = 0; k < 3; k++) col[k] = Math.round(bg[k] + (pos[k] - bg[k]) * (0.09 + 0.56 * conf));
        } else {
          col = [0, 0, 0];
          for (k = 0; k < 3; k++) col[k] = Math.round(bg[k] + (neg[k] - bg[k]) * (0.09 + 0.62 * conf));
        }
        var o = (j * GRID + i) * 4;
        px[o] = col[0];
        px[o + 1] = col[1];
        px[o + 2] = col[2];
        px[o + 3] = 255;
      }
    }
    ictx.putImageData(img, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, PAD, PAD, INNER, INNER);

    // 细网格，帮助对齐
    ctx.strokeStyle = "rgba(232,230,225,0.06)";
    ctx.lineWidth = 1;
    for (k = 1; k <= 3; k++) {
      var g = PAD + INNER * k / 4;
      ctx.beginPath();
      ctx.moveTo(g, PAD);
      ctx.lineTo(g, PAD + INNER);
      ctx.moveTo(PAD, g);
      ctx.lineTo(PAD + INNER, g);
      ctx.stroke();
    }
    ctx.strokeStyle = COLOR.line;
    ctx.strokeRect(PAD + 0.5, PAD + 0.5, INNER - 1, INNER - 1);

    // 四个样本点：空心 = 标 0，实心 = 标 1
    for (var s = 0; s < SAMPLES.length; s++) {
      var sx = PAD + clamp01(SAMPLES[s][0]) * INNER;
      var sy = PAD + (1 - clamp01(SAMPLES[s][1])) * INNER;
      var isOne = LABELS[s] === 1;
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      if (isOne) {
        ctx.fillStyle = COLOR.signal;
        ctx.fill();
      } else {
        ctx.fillStyle = COLOR.bg;
        ctx.fill();
        ctx.strokeStyle = COLOR.text;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      ctx.font = "10px ui-monospace,SFMono-Regular,Consolas,monospace";
      ctx.fillStyle = isOne ? COLOR.signal : COLOR.text;
      ctx.textAlign = SAMPLES[s][0] >= 1 ? "right" : "left";
      ctx.textBaseline = "middle";
      var tx = sx + (SAMPLES[s][0] >= 1 ? -9 : 9);
      ctx.fillText(String(LABELS[s]), tx, sy - 9);
    }

    ctx.font = "10px ui-monospace,SFMono-Regular,Consolas,monospace";
    ctx.fillStyle = COLOR.muted;
    ctx.textAlign = "left";
    ctx.fillText("x2", 4, 10);
    ctx.textAlign = "right";
    ctx.fillText("x1", S - 4, S - 3);
  }

  // ---------------- 隐藏层散点 ----------------
  function drawHidden(st) {
    var cv = $("cvHidden");
    if (!cv) return;
    var S = 380, PAD = 34, INNER = S - PAD * 2;
    var ctx = prep(cv, S, S);

    ctx.strokeStyle = "rgba(232,230,225,0.08)";
    ctx.lineWidth = 1;
    for (var k = 0; k <= 4; k++) {
      var g = PAD + INNER * k / 4;
      ctx.beginPath();
      ctx.moveTo(g, PAD);
      ctx.lineTo(g, PAD + INNER);
      ctx.moveTo(PAD, g);
      ctx.lineTo(PAD + INNER, g);
      ctx.stroke();
    }

    // 输出层在隐藏空间里的分界线：v0·h1 + v1·h2 + b0 = 0
    var v0 = st.t[6], v1 = st.t[7], bo = st.t[8];
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD, PAD, INNER, INNER);
    ctx.clip();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = "rgba(232,230,225,0.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    if (Math.abs(v1) > Math.abs(v0)) {
      for (var i = 0; i <= 40; i++) {
        var h1 = i / 40;
        var h2 = -(v0 * h1 + bo) / v1;
        var X = PAD + h1 * INNER, Y = PAD + (1 - h2) * INNER;
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      }
    } else {
      for (i = 0; i <= 40; i++) {
        var hh2 = i / 40;
        var hh1 = -(v1 * hh2 + bo) / v0;
        var X2 = PAD + hh1 * INNER, Y2 = PAD + (1 - hh2) * INNER;
        if (i === 0) ctx.moveTo(X2, Y2); else ctx.lineTo(X2, Y2);
      }
    }
    ctx.stroke();
    ctx.restore();

    var drawn = [];
    for (var s = 0; s < SAMPLES.length; s++) {
      var h1v = st.h[s * 2], h2v = st.h[s * 2 + 1];
      var cx = PAD + clamp01(h1v) * INNER;
      var cy = PAD + (1 - clamp01(h2v)) * INNER;
      var isOne = LABELS[s] === 1;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      if (isOne) {
        ctx.fillStyle = COLOR.signal;
        ctx.fill();
      } else {
        ctx.fillStyle = COLOR.bg;
        ctx.fill();
        ctx.strokeStyle = COLOR.text;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      // 两个点几乎重合时把标签错开，保证两条标注都能读
      var ly = cy - 8;
      for (var d = 0; d < drawn.length; d++) {
        if (Math.abs(drawn[d][0] - cx) < 14 && Math.abs(drawn[d][1] - ly) < 13) ly -= 14;
      }
      drawn.push([cx, ly]);
      ctx.font = "10px ui-monospace,SFMono-Regular,Consolas,monospace";
      ctx.fillStyle = isOne ? COLOR.signal : COLOR.text;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("(" + SAMPLES[s][0] + "," + SAMPLES[s][1] + ")", cx + 9, ly);
    }

    ctx.font = "10px ui-monospace,SFMono-Regular,Consolas,monospace";
    ctx.fillStyle = COLOR.muted;
    ctx.textAlign = "left";
    ctx.fillText("h1", S - 16, S - 16);
    ctx.fillText("h2", 6, 22);
  }

  // ---------------- 数值列表 ----------------
  function buildProbs() {
    var box = $("probList");
    if (!box) return;
    var html = "";
    for (var s = 0; s < SAMPLES.length; s++) {
      html += '<div class="prob">' +
        '<span class="prob__k">(' + SAMPLES[s][0] + ',' + SAMPLES[s][1] + ')→' + LABELS[s] + "</span>" +
        '<span class="prob__bar"><i></i></span>' +
        '<span class="prob__v">—</span></div>';
    }
    box.innerHTML = html;
  }

  function buildGrad() {
    var box = $("gradRows");
    if (!box) return;
    var order = M.param_order || [];
    var html = "";
    for (var i = 0; i < order.length; i++) {
      html += '<div class="prow">' +
        '<span class="prow__k">' + order[i] + "</span>" +
        '<span class="prow__bar"><i></i></span>' +
        '<span class="prow__v">—</span></div>';
    }
    box.innerHTML = html;
  }

  function updateProbs(st) {
    var box = $("probList");
    if (!box) return;
    var rows = box.querySelectorAll(".prob");
    for (var s = 0; s < rows.length; s++) {
      var p = st.q[s];
      var bar = rows[s].querySelector("i");
      bar.style.width = (p * 100).toFixed(1) + "%";
      bar.className = p < 0.5 ? "neg" : "";
      rows[s].querySelector(".prob__v").textContent = f4(p);
    }
  }

  function updateGrad(st) {
    var box = $("gradRows");
    if (!box) return;
    var rows = box.querySelectorAll(".prow");
    for (var i = 0; i < rows.length; i++) {
      var g = st.g[i];
      var bar = rows[i].querySelector("i");
      bar.style.width = (Math.abs(g) / maxG * 100).toFixed(1) + "%";
      bar.className = g < 0 ? "neg" : "";
      rows[i].querySelector(".prow__v").textContent = signed(g, 4);
    }
  }

  function updateText(st) {
    var e = Math.round(st.e);
    var clock = $("clockEpoch");
    if (clock) clock.textContent = String(e);
    var clockLoss = $("clockLoss");
    if (clockLoss) clockLoss.textContent = f4(st.loss);
    var ge = $("gradEpoch");
    if (ge) ge.textContent = String(e);

    var acc = 0, margin = 1, maxH = 0;
    for (var s = 0; s < st.q.length; s++) {
      if ((st.q[s] >= 0.5) === (LABELS[s] === 1)) acc++;
      margin = Math.min(margin, Math.abs(st.q[s] - 0.5));
    }
    for (var k = 0; k < st.h.length; k++) maxH = Math.max(maxH, st.h[k]);

    var el;
    if ((el = $("rLoss"))) el.textContent = f4(st.loss);
    if ((el = $("rAcc"))) el.textContent = acc + "/4";
    if ((el = $("rMargin"))) el.textContent = f4(margin);
    if ((el = $("rHid"))) el.textContent = f3(maxH);

    var hint = $("fieldHint");
    if (hint) hint.textContent = "第 " + e + " 轮 · p ≥ 0.5 判 1";

    var scrub = $("scrub");
    if (scrub && document.activeElement !== scrub) scrub.value = String(Math.round(st.e));
  }

  function render(i) {
    idx = Math.max(0, Math.min(N - 1, i));
    var st = frame(idx);
    drawNet(st);
    drawField(st);
    drawHidden(st);
    updateProbs(st);
    updateGrad(st);
    updateText(st);
  }

  // ---------------- 播放控制 ----------------
  function speed() {
    var sel = $("selSpeed");
    var v = sel ? parseInt(sel.value, 10) : 20;
    return Math.max(10, Math.min(200, isNaN(v) ? 20 : v));
  }

  function setPlaying(on) {
    playing = on;
    var btn = $("btnPlay");
    if (btn) {
      btn.textContent = on ? "暂停" : "播放";
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    }
  }

  function stop() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    setPlaying(false);
  }

  function play() {
    if (timer) return;
    if (idx >= N - 1) render(0);
    setPlaying(true);
    timer = setInterval(function () {
      if (idx >= N - 1) {
        stop();
        return;
      }
      render(idx + 1);
    }, speed());
  }

  function toggle() {
    if (timer) stop(); else play();
  }

  function bind() {
    var btnPlay = $("btnPlay");
    if (btnPlay) btnPlay.addEventListener("click", toggle);

    var prev = $("btnPrev");
    if (prev) prev.addEventListener("click", function () { stop(); render(idx - 1); });

    var next = $("btnNext");
    if (next) next.addEventListener("click", function () { stop(); render(idx + 1); });

    var first = $("btnFirst");
    if (first) first.addEventListener("click", function () { stop(); render(0); });

    var last = $("btnLast");
    if (last) last.addEventListener("click", function () { stop(); render(N - 1); });

    var scrub = $("scrub");
    if (scrub) {
      scrub.addEventListener("input", function () {
        stop();
        render(parseInt(scrub.value, 10) || 0);
      });
    }

    var sel = $("selSpeed");
    if (sel) {
      sel.addEventListener("change", function () {
        if (timer) {
          clearInterval(timer);
          timer = null;
          play();
        }
      });
    }

    var replay = document.querySelector('[data-act="replay"]');
    if (replay) {
      replay.addEventListener("click", function () {
        var stage = document.getElementById("play");
        if (stage && stage.scrollIntoView) {
          stage.scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
        }
        stop();
        render(0);
        play();
      });
    }

    // 空格只在回放台的按钮上生效，避免抢走页面滚动
    document.addEventListener("keydown", function (ev) {
      if (ev.key !== " " && ev.key !== "Spacebar") return;
      var t = ev.target;
      if (!t || !t.closest || !t.closest(".stage")) return;
      if (t.tagName === "BUTTON" || t.tagName === "SELECT") return;
      ev.preventDefault();
      toggle();
    });
  }

  buildProbs();
  buildGrad();
  bind();
  render(0);
})();
