/* site.js —— 三站共用的渐进增强层：入场、阅读进度、代码复制、图片灯箱。
   任何一项失败都不影响阅读：没有 JS 时页面本来就是完整可见的。 */

(function () {
  'use strict';

  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- 入场 ---------- */

  function initReveal() {
    var items = document.querySelectorAll('.reveal');
    if (!items.length) return;

    if (reduce || !('IntersectionObserver' in window)) {
      items.forEach(function (el) { el.classList.add('is-in'); });
      return;
    }

    // 同组递次：最多 6 个，超过就整组同现
    document.querySelectorAll('[data-stagger]').forEach(function (group) {
      var kids = Array.prototype.slice.call(group.children);
      if (kids.length > 6) return;
      kids.forEach(function (kid, i) {
        if (kid.classList.contains('reveal')) {
          kid.style.transitionDelay = (i * 60) + 'ms';
        }
      });
    });

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-in');
        io.unobserve(entry.target);
      });
    }, { threshold: 0.15, rootMargin: '0px 0px -8% 0px' });

    items.forEach(function (el) { io.observe(el); });
  }

  /* ---------- 阅读进度（只改 transform） ---------- */

  function initProgress() {
    var bar = document.querySelector('.progress');
    if (!bar || reduce) return;
    var ticking = false;

    function paint() {
      var doc = document.documentElement;
      var max = doc.scrollHeight - window.innerHeight;
      var ratio = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      bar.style.transform = 'scaleX(' + ratio.toFixed(4) + ')';
      ticking = false;
    }

    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(paint);
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    paint();
  }

  /* ---------- 代码复制 ---------- */

  function initCopy() {
    if (!navigator.clipboard) return;
    document.querySelectorAll('.code').forEach(function (box) {
      var pre = box.querySelector('pre');
      if (!pre) return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'code__copy';
      btn.textContent = '复制';
      btn.setAttribute('aria-label', '复制这段代码');
      btn.addEventListener('click', function () {
        var text = pre.innerText;
        navigator.clipboard.writeText(text).then(function () {
          btn.textContent = '已复制';
          setTimeout(function () { btn.textContent = '复制'; }, 1600);
        }).catch(function () {
          btn.textContent = '复制失败';
          setTimeout(function () { btn.textContent = '复制'; }, 1600);
        });
      });
      box.appendChild(btn);
    });
  }

  /* ---------- 灯箱 ---------- */

  function initLightbox() {
    var figures = Array.prototype.slice.call(document.querySelectorAll('[data-lightbox]'));
    if (!figures.length) return;

    var overlay = document.createElement('div');
    overlay.className = 'lbx';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', '图片查看');
    overlay.hidden = true;
    overlay.innerHTML =
      '<div class="lbx__bar">' +
        '<span class="lbx__count mono"></span>' +
        '<button type="button" class="btn lbx__close">关闭 Esc</button>' +
      '</div>' +
      '<div class="lbx__stage">' +
        '<button type="button" class="btn lbx__prev" aria-label="上一张">←</button>' +
        '<img class="lbx__img" alt="">' +
        '<button type="button" class="btn lbx__next" aria-label="下一张">→</button>' +
      '</div>' +
      '<p class="lbx__cap tiny"></p>';
    document.body.appendChild(overlay);

    var img = overlay.querySelector('.lbx__img');
    var cap = overlay.querySelector('.lbx__cap');
    var count = overlay.querySelector('.lbx__count');
    var index = 0;
    var lastFocus = null;

    function open(i) {
      index = (i + figures.length) % figures.length;
      var fig = figures[index];
      var source = fig.querySelector('img') || fig;
      var caption = fig.getAttribute('data-caption') || '';
      img.src = source.getAttribute('src') || source.dataset.src || '';
      img.alt = source.getAttribute('alt') || caption;
      cap.textContent = caption;
      count.textContent = (index + 1) + ' / ' + figures.length;
      overlay.hidden = false;
      lastFocus = document.activeElement;
      overlay.querySelector('.lbx__close').focus();
    }

    function close() {
      overlay.hidden = true;
      img.removeAttribute('src');
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }

    figures.forEach(function (fig, i) {
      fig.setAttribute('tabindex', '0');
      fig.setAttribute('role', 'button');
      fig.addEventListener('click', function () { open(i); });
      fig.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(i); }
      });
    });

    overlay.querySelector('.lbx__close').addEventListener('click', close);
    overlay.querySelector('.lbx__prev').addEventListener('click', function () { open(index - 1); });
    overlay.querySelector('.lbx__next').addEventListener('click', function () { open(index + 1); });
    overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });

    document.addEventListener('keydown', function (e) {
      if (overlay.hidden) return;
      if (e.key === 'Escape') { e.preventDefault(); close(); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); open(index + 1); return; }
      if (e.key === 'ArrowLeft') { e.preventDefault(); open(index - 1); return; }
      if (e.key !== 'Tab') return;
      // 焦点陷阱：只有 bar 里的按钮和两张箭头可聚焦
      var focusable = overlay.querySelectorAll('button:not([disabled])');
      if (!focusable.length) return;
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });
  }

  function boot() {
    initReveal();
    initProgress();
    initCopy();
    initLightbox();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
