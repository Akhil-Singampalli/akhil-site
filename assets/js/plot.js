/* plot.js — small canvas helpers for the explorer pages.
   Colours are read from the site's CSS custom properties at paint time, so
   every chart follows the theme toggle instead of hard-coding a palette. */
window.Plot = (function () {
  'use strict';

  function tokens() {
    var cs = getComputedStyle(document.documentElement);
    function v(name, fallback) {
      var out = cs.getPropertyValue(name).trim();
      return out || fallback;
    }
    return {
      surface: v('--surface', '#fff'),
      sunken: v('--bg-sunken', '#f5f8f8'),
      ink: v('--ink', '#0f1419'),
      ink2: v('--ink-2', '#384049'),
      ink3: v('--ink-3', '#5f6b75'),
      line: v('--line', '#e3e9ea'),
      lineStrong: v('--line-strong', '#c9d4d5'),
      brand: v('--brand', '#0d5c63'),
      t1: v('--t1', '#0d5c63'),
      t2: v('--t2', '#6d28d9'),
      t3: v('--t3', '#b45309'),
      t4: v('--t4', '#475467'),
      dark: document.documentElement.getAttribute('data-theme') === 'dark' ||
            (!document.documentElement.getAttribute('data-theme') &&
             window.matchMedia('(prefers-color-scheme: dark)').matches)
    };
  }

  /* Size the backing store to device pixels; draw in CSS pixels. */
  function setup(canvas) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width));
    var h = Math.max(1, Math.round(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function hexToRgb(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    var n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function mix(a, b, t) {
    return [
      Math.round(a[0] + (b[0] - a[0]) * t),
      Math.round(a[1] + (b[1] - a[1]) * t),
      Math.round(a[2] + (b[2] - a[2]) * t)
    ];
  }

  /* Loss ramp: low loss in the brand teal, high loss in amber/red.
     Deliberately not a rainbow — the ordering has to read at a glance. */
  function lossRamp(t, dark) {
    t = Math.max(0, Math.min(1, t));
    var stops = dark
      ? [[10, 38, 42], [45, 145, 145], [222, 231, 231], [240, 180, 41], [176, 42, 30]]
      : [[8, 62, 68], [58, 150, 150], [238, 244, 244], [226, 160, 46], [158, 36, 26]];
    var seg = t * (stops.length - 1);
    var i = Math.min(stops.length - 2, Math.floor(seg));
    return mix(stops[i], stops[i + 1], seg - i);
  }

  function classColors(tk) {
    return [tk.t1, tk.t2, tk.t3, tk.t4];
  }

  /* ---------- decision field ----------
     classify(x, y) -> {cls, conf}. Painted at `res` px per cell, then the
     data points are drawn over it by the caller. */
  /* Raster helpers render into an offscreen canvas sized to the sample grid and
     are then drawn with drawImage. putImageData bypasses the context transform,
     so writing CSS-pixel ImageData straight to a HiDPI canvas would cover only
     the top-left 1/dpr of it. drawImage respects the transform and interpolates. */
  function blit(ctx, off, w, h) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, w, h);
  }

  function offscreen(cw, ch) {
    var c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    return c;
  }

  function decisionField(ctx, w, h, domain, classify, tk, res) {
    res = res || 4;
    var cols = classColors(tk).map(hexToRgb);
    var bg = hexToRgb(tk.surface);
    var gw = Math.max(2, Math.ceil(w / res)), gh = Math.max(2, Math.ceil(h / res));
    var off = offscreen(gw, gh), octx = off.getContext('2d');
    var img = octx.createImageData(gw, gh), d = img.data;

    for (var py = 0; py < gh; py++) {
      var wy = domain.y1 + (domain.y0 - domain.y1) * (py / (gh - 1));
      for (var px = 0; px < gw; px++) {
        var wx = domain.x0 + (domain.x1 - domain.x0) * (px / (gw - 1));
        var r = classify(wx, wy);
        var c = mix(bg, cols[r.cls % cols.length], 0.18 + 0.42 * r.conf);
        var o = (py * gw + px) * 4;
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    blit(ctx, off, w, h);
  }

  function scatter(ctx, w, h, X, y, idx, domain, tk, opts) {
    opts = opts || {};
    var cols = classColors(tk);
    var rad = opts.r || 3.2;
    /* A thin, low-contrast halo separates overlapping dots without bleaching
       them; a full-weight surface-coloured stroke swallows a small marker. */
    ctx.lineWidth = 0.9;
    ctx.strokeStyle = tk.dark ? 'rgba(0,0,0,.55)' : 'rgba(255,255,255,.85)';
    for (var k = 0; k < idx.length; k++) {
      var i = idx[k];
      var px = (X[i * 2] - domain.x0) / (domain.x1 - domain.x0) * w;
      var py = (domain.y1 - X[i * 2 + 1]) / (domain.y1 - domain.y0) * h;
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fillStyle = cols[y[i] % cols.length];
      ctx.globalAlpha = opts.alpha == null ? 1 : opts.alpha;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
  }

  /* ---------- heatmap from a value grid ---------- */
  function heatmap(ctx, w, h, grid, n, lo, hi, tk) {
    var off = offscreen(n, n), octx = off.getContext('2d');
    var img = octx.createImageData(n, n), d = img.data;
    var span = (hi - lo) || 1;
    for (var i = 0; i < n * n; i++) {
      var val = grid[i];
      var c = lossRamp(isFinite(val) ? (val - lo) / span : 1, tk.dark);
      var o = i * 4;
      d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2];
      d[o + 3] = isFinite(val) ? 255 : 0;
    }
    octx.putImageData(img, 0, 0);
    blit(ctx, off, w, h);
  }

  /* Iso-loss contour rings, drawn with marching-squares on the same grid. */
  function contours(ctx, w, h, grid, n, levels, tk) {
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = tk.dark ? 'rgba(255,255,255,.30)' : 'rgba(0,0,0,.26)';
    var sx = w / (n - 1), sy = h / (n - 1);

    for (var li = 0; li < levels.length; li++) {
      var lv = levels[li];
      ctx.beginPath();
      for (var j = 0; j < n - 1; j++) {
        for (var i = 0; i < n - 1; i++) {
          var a = grid[j * n + i], b = grid[j * n + i + 1];
          var c = grid[(j + 1) * n + i + 1], e = grid[(j + 1) * n + i];
          if (!isFinite(a) || !isFinite(b) || !isFinite(c) || !isFinite(e)) continue;
          var pts = [];
          if ((a < lv) !== (b < lv)) pts.push([i + (lv - a) / (b - a), j]);
          if ((b < lv) !== (c < lv)) pts.push([i + 1, j + (lv - b) / (c - b)]);
          if ((e < lv) !== (c < lv)) pts.push([i + (lv - e) / (c - e), j + 1]);
          if ((a < lv) !== (e < lv)) pts.push([i, j + (lv - a) / (e - a)]);
          if (pts.length >= 2) {
            ctx.moveTo(pts[0][0] * sx, pts[0][1] * sy);
            ctx.lineTo(pts[1][0] * sx, pts[1][1] * sy);
          }
        }
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ---------- line chart ---------- */
  function lineChart(ctx, w, h, series, opts) {
    opts = opts || {};
    var tk = opts.tokens || tokens();
    var pad = opts.pad || { l: 42, r: 12, t: 14, b: 26 };
    var iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
    var xMax = opts.xMax || 1, yMin = opts.yMin == null ? 0 : opts.yMin;
    var yMax = opts.yMax == null ? 1 : opts.yMax;

    function X(v) { return pad.l + (v / xMax) * iw; }
    function Y(v) { return pad.t + ih - ((v - yMin) / (yMax - yMin)) * ih; }

    /* grid + axis labels */
    ctx.strokeStyle = tk.line;
    ctx.fillStyle = tk.ink3;
    ctx.lineWidth = 1;
    ctx.font = '11px ' + (opts.mono || 'monospace');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var ticks = opts.yTicks || [0, 0.25, 0.5, 0.75, 1];
    for (var t = 0; t < ticks.length; t++) {
      var yy = Y(ticks[t]);
      ctx.beginPath();
      ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy);
      ctx.stroke();
      ctx.fillText(opts.yFmt ? opts.yFmt(ticks[t]) : ticks[t], pad.l - 8, yy);
    }

    /* optional shaded task bands */
    if (opts.bands) {
      opts.bands.forEach(function (band) {
        ctx.fillStyle = band.color;
        ctx.globalAlpha = 0.1;
        ctx.fillRect(X(band.from), pad.t, X(band.to) - X(band.from), ih);
        ctx.globalAlpha = 1;
        ctx.fillStyle = tk.ink3;
        ctx.textAlign = 'left';
        ctx.font = '10px ' + (opts.mono || 'monospace');
        ctx.fillText(band.label, X(band.from) + 6, pad.t + 10);
      });
    }

    series.forEach(function (s) {
      if (!s.data.length) return;
      ctx.beginPath();
      for (var i = 0; i < s.data.length; i++) {
        var px = X(s.data[i][0]), py = Y(s.data[i][1]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 2;
      if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.strokeStyle = tk.lineStrong;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ih); ctx.lineTo(w - pad.r, pad.t + ih);
    ctx.stroke();
  }

  return {
    tokens: tokens, setup: setup, lossRamp: lossRamp, classColors: classColors,
    hexToRgb: hexToRgb, mix: mix,
    decisionField: decisionField, scatter: scatter,
    heatmap: heatmap, contours: contours, lineChart: lineChart
  };
})();
