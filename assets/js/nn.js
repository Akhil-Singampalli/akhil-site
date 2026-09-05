/* nn.js — a small dense network with hand-written forward/backward passes.
   Everything on the explorer pages is computed with this in the browser:
   no pre-baked curves, no canned numbers.

   Layout: tanh hidden layers, linear output, softmax + cross-entropy.
   Weights are Float64Array, one flat row-major block per layer. */
window.NN = (function () {
  'use strict';

  /* Seeded RNG so a given configuration always reproduces the same run. */
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = s + 0x6D2B79F5 | 0;
      var t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function gauss(r) {
    var u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ---------------- construction ---------------- */

  function create(sizes, r) {
    var W = [], b = [], l;
    for (l = 0; l < sizes.length - 1; l++) {
      var nin = sizes[l], nout = sizes[l + 1];
      var w = new Float64Array(nout * nin);
      var scale = Math.sqrt(2 / nin);
      for (var i = 0; i < w.length; i++) w[i] = gauss(r) * scale;
      W.push(w);
      b.push(new Float64Array(nout));
    }
    return { sizes: sizes.slice(), W: W, b: b, L: sizes.length - 1 };
  }

  function clone(net) {
    return {
      sizes: net.sizes.slice(),
      W: net.W.map(function (w) { return new Float64Array(w); }),
      b: net.b.map(function (v) { return new Float64Array(v); }),
      L: net.L
    };
  }

  /* Scratch buffers, allocated once and reused for every sample. */
  function scratch(net) {
    var a = [], d = [], l;
    for (l = 0; l <= net.L; l++) a.push(new Float64Array(net.sizes[l]));
    for (l = 0; l < net.L; l++) d.push(new Float64Array(net.sizes[l + 1]));
    return { a: a, d: d, p: new Float64Array(net.sizes[net.L]) };
  }

  function grads(net) {
    return {
      W: net.W.map(function (w) { return new Float64Array(w.length); }),
      b: net.b.map(function (v) { return new Float64Array(v.length); })
    };
  }

  function zero(g) {
    for (var l = 0; l < g.W.length; l++) { g.W[l].fill(0); g.b[l].fill(0); }
  }

  /* ---------------- forward ---------------- */

  function forward(net, x, sc) {
    var a = sc.a, l, i, j;
    a[0].set(x);
    for (l = 0; l < net.L; l++) {
      var nin = net.sizes[l], nout = net.sizes[l + 1];
      var W = net.W[l], bb = net.b[l], src = a[l], dst = a[l + 1];
      var last = (l === net.L - 1);
      for (i = 0; i < nout; i++) {
        var s = bb[i], off = i * nin;
        for (j = 0; j < nin; j++) s += W[off + j] * src[j];
        dst[i] = last ? s : Math.tanh(s);
      }
    }
    return a[net.L];
  }

  function softmax(logits, out) {
    var n = logits.length, m = -Infinity, i, sum = 0;
    for (i = 0; i < n; i++) if (logits[i] > m) m = logits[i];
    for (i = 0; i < n; i++) { out[i] = Math.exp(logits[i] - m); sum += out[i]; }
    for (i = 0; i < n; i++) out[i] /= sum;
    return out;
  }

  /* ---------------- evaluation ----------------
     `pick` optionally restricts which sample indices are used, so we can
     score a model on one task's data while it trains on another. */
  function evaluate(net, X, y, sc, pick) {
    var C = net.sizes[net.L];
    var n = pick ? pick.length : y.length;
    if (!n) return { loss: 0, acc: 0 };
    var loss = 0, correct = 0, xb = new Float64Array(net.sizes[0]);

    for (var k = 0; k < n; k++) {
      var idx = pick ? pick[k] : k;
      xb[0] = X[idx * 2]; xb[1] = X[idx * 2 + 1];
      var p = softmax(forward(net, xb, sc), sc.p);
      loss += -Math.log(Math.max(p[y[idx]], 1e-12));
      var best = 0;
      for (var c = 1; c < C; c++) if (p[c] > p[best]) best = c;
      if (best === y[idx]) correct++;
    }
    return { loss: loss / n, acc: correct / n };
  }

  /* ---------------- backward ---------------- */

  /* Accumulates dLoss/dparams for one sample into g. Returns the sample loss. */
  function accumulate(net, x, label, sc, g) {
    var a = sc.a, d = sc.d, L = net.L, l, i, j;
    var p = softmax(forward(net, x, sc), sc.p);
    var loss = -Math.log(Math.max(p[label], 1e-12));

    /* Softmax + cross-entropy collapse to (p - onehot) at the logits. */
    var dl = d[L - 1];
    for (i = 0; i < dl.length; i++) dl[i] = p[i] - (i === label ? 1 : 0);

    for (l = L - 1; l >= 0; l--) {
      var nin = net.sizes[l], nout = net.sizes[l + 1];
      var delta = d[l], src = a[l], gW = g.W[l], gb = g.b[l];

      for (i = 0; i < nout; i++) {
        var di = delta[i], off = i * nin;
        gb[i] += di;
        for (j = 0; j < nin; j++) gW[off + j] += di * src[j];
      }

      if (l > 0) {
        var prev = d[l - 1], W = net.W[l], act = a[l];
        for (j = 0; j < nin; j++) {
          var s = 0;
          for (i = 0; i < nout; i++) s += W[i * nin + j] * delta[i];
          prev[j] = s * (1 - act[j] * act[j]);   /* tanh' */
        }
      }
    }
    return loss;
  }

  /* ---------------- optimisation ---------------- */

  function momentumState(net) {
    return {
      W: net.W.map(function (w) { return new Float64Array(w.length); }),
      b: net.b.map(function (v) { return new Float64Array(v.length); })
    };
  }

  function step(net, g, mom, lr, beta, scale) {
    for (var l = 0; l < net.L; l++) {
      var W = net.W[l], gW = g.W[l], mW = mom.W[l], i;
      for (i = 0; i < W.length; i++) {
        mW[i] = beta * mW[i] + gW[i] * scale;
        W[i] -= lr * mW[i];
      }
      var b = net.b[l], gb = g.b[l], mb = mom.b[l];
      for (i = 0; i < b.length; i++) {
        mb[i] = beta * mb[i] + gb[i] * scale;
        b[i] -= lr * mb[i];
      }
    }
  }

  /* Elastic Weight Consolidation: pull each weight back toward its post-task-1
     value, weighted by how much task 1 cared about it (the Fisher diagonal). */
  function ewcPenaltyGrad(net, anchor, fisher, lambda, g) {
    if (!lambda) return 0;
    var total = 0;
    for (var l = 0; l < net.L; l++) {
      var W = net.W[l], aW = anchor.W[l], fW = fisher.W[l], gW = g.W[l], i, diff;
      for (i = 0; i < W.length; i++) {
        diff = W[i] - aW[i];
        gW[i] += lambda * fW[i] * diff;
        total += 0.5 * lambda * fW[i] * diff * diff;
      }
      var b = net.b[l], ab = anchor.b[l], fb = fisher.b[l], gb = g.b[l];
      for (i = 0; i < b.length; i++) {
        diff = b[i] - ab[i];
        gb[i] += lambda * fb[i] * diff;
        total += 0.5 * lambda * fb[i] * diff * diff;
      }
    }
    return total;
  }

  /* Diagonal Fisher, estimated as the mean squared per-sample gradient. */
  function fisherDiagonal(net, X, y, sc, pick) {
    var f = grads(net), one = grads(net);
    var n = pick ? pick.length : y.length;
    var xb = new Float64Array(net.sizes[0]);

    for (var k = 0; k < n; k++) {
      var idx = pick ? pick[k] : k;
      zero(one);
      xb[0] = X[idx * 2]; xb[1] = X[idx * 2 + 1];
      accumulate(net, xb, y[idx], sc, one);
      for (var l = 0; l < net.L; l++) {
        var a = one.W[l], t = f.W[l], i;
        for (i = 0; i < a.length; i++) t[i] += a[i] * a[i];
        var ab = one.b[l], tb = f.b[l];
        for (i = 0; i < ab.length; i++) tb[i] += ab[i] * ab[i];
      }
    }
    /* Normalise to a [0,1] importance map. The raw empirical Fisher collapses
       toward zero once the task has converged (near-zero per-sample gradients),
       which would leave lambda with no usable scale. */
    var max = 0, l2, j;
    for (l2 = 0; l2 < net.L; l2++) {
      for (j = 0; j < f.W[l2].length; j++) if (f.W[l2][j] > max) max = f.W[l2][j];
      for (j = 0; j < f.b[l2].length; j++) if (f.b[l2][j] > max) max = f.b[l2][j];
    }
    var inv = max > 0 ? 1 / max : 0;
    for (l2 = 0; l2 < net.L; l2++) {
      var w = f.W[l2];
      for (j = 0; j < w.length; j++) w[j] *= inv;
      var bb = f.b[l2];
      for (j = 0; j < bb.length; j++) bb[j] *= inv;
    }
    return f;
  }

  /* ---------------- parameter-space helpers (loss landscapes) ---------------- */

  function paramCount(net) {
    var n = 0;
    for (var l = 0; l < net.L; l++) n += net.W[l].length + net.b[l].length;
    return n;
  }

  function getParams(net) {
    var p = new Float64Array(paramCount(net)), o = 0;
    for (var l = 0; l < net.L; l++) {
      p.set(net.W[l], o); o += net.W[l].length;
      p.set(net.b[l], o); o += net.b[l].length;
    }
    return p;
  }

  function setParams(net, p) {
    var o = 0;
    for (var l = 0; l < net.L; l++) {
      net.W[l].set(p.subarray(o, o + net.W[l].length)); o += net.W[l].length;
      net.b[l].set(p.subarray(o, o + net.b[l].length)); o += net.b[l].length;
    }
  }

  /* Filter-normalised random direction, per Li et al. 2018. Each neuron's
     incoming weight vector gets a direction scaled to that neuron's own norm,
     which is what makes two different models' landscapes comparable. */
  function direction(net, r) {
    var d = new Float64Array(paramCount(net)), o = 0, l, i, j;
    for (l = 0; l < net.L; l++) {
      var nin = net.sizes[l], nout = net.sizes[l + 1], W = net.W[l];
      for (i = 0; i < nout; i++) {
        var dn = 0, wn = 0, base = o + i * nin, v;
        for (j = 0; j < nin; j++) {
          v = gauss(r);
          d[base + j] = v;
          dn += v * v;
          wn += W[i * nin + j] * W[i * nin + j];
        }
        var s = Math.sqrt(wn) / (Math.sqrt(dn) + 1e-10);
        for (j = 0; j < nin; j++) d[base + j] *= s;
      }
      o += W.length;
      /* Biases are left at zero: perturbing them is not part of the method. */
      o += net.b[l].length;
    }
    return d;
  }

  return {
    rng: rng, gauss: gauss,
    create: create, clone: clone, scratch: scratch, grads: grads, zero: zero,
    forward: forward, softmax: softmax, evaluate: evaluate, accumulate: accumulate,
    momentumState: momentumState, step: step,
    ewcPenaltyGrad: ewcPenaltyGrad, fisherDiagonal: fisherDiagonal,
    paramCount: paramCount, getParams: getParams, setParams: setParams,
    direction: direction
  };
})();
