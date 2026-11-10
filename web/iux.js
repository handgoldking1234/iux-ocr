/* IUX 엔진 (웹) — 손글씨·흐린 글씨용 딥러닝 OCR을 브라우저 안에서 실행 (ONNX Runtime Web, 서버 전송 없음).
   app/iux_engine.py 와 같은 방식: 글줄 찾기 → 인쇄체/손글씨 두 모델로 읽고 더 확신하는 쪽 선택 → 띄어쓰기 복원 → 읽는 순서 정리 */
(function () {
  'use strict';
  const ORT = 'web/vendor/ort/', MODELS = 'models/';
  const S = { ready: null, det: null, rec: null, hw: null, chars: null };
  const mk = (w, h) => { const c = document.createElement('canvas'); c.width = Math.max(1, w); c.height = Math.max(1, h); return c; };
  function loadScript(src) { return new Promise((ok, no) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => no(new Error(src + ' 로드 실패')); document.head.appendChild(s); }); }
  async function fetchBuf(url, onp) {
    const r = await fetch(url); if (!r.ok) throw new Error(url + ' 내려받기 실패 (' + r.status + ')');
    const total = +r.headers.get('content-length') || 0;
    if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
    const rd = r.body.getReader(), parts = []; let n = 0;
    for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); n += value.length; onp && onp(value.length); }
    const out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out;
  }
  function load(onProgress) {
    if (S.ready) return S.ready;
    S.ready = (async () => {
      if (!window.ort) await loadScript(ORT + 'ort.wasm.min.js');
      ort.env.wasm.wasmPaths = new URL(ORT, location.href).href;
      ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      ort.env.logLevel = 'error';
      const TOTAL = 33e6; let got = 0;
      const onp = n => { got += n; onProgress && onProgress(Math.min(0.99, got / TOTAL)); };
      const [det, rec, hw, dict] = await Promise.all([
        fetchBuf(MODELS + 'iux_det.onnx', onp), fetchBuf(MODELS + 'iux_rec_ko.onnx', onp), fetchBuf(MODELS + 'iux_rec_hw.onnx', onp),
        fetch(MODELS + 'iux_rec_ko_dict.txt').then(r => r.text()),
      ]);
      const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      S.det = await ort.InferenceSession.create(det, opt);
      S.rec = await ort.InferenceSession.create(rec, opt);
      S.hw = await ort.InferenceSession.create(hw, opt);
      const ch = dict.split('\n'); if (ch[ch.length - 1] === '') ch.pop();
      S.chars = ['', ...ch, ' '];
      onProgress && onProgress(1);
    })().catch(e => { S.ready = null; throw e; });
    return S.ready;
  }

  /* ---------- 글줄 찾기 (DB) ---------- */
  async function detect(src, side = 1280) {
    const w = src.width, h = src.height;
    let s = Math.min(1, side / Math.max(w, h)); if (Math.max(w, h) < 960) s = Math.min(2, 960 / Math.max(w, h));
    const nw = Math.max(32, Math.round(w * s / 32) * 32), nh = Math.max(32, Math.round(h * s / 32) * 32);
    const c = mk(nw, nh), x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high'; x.drawImage(src, 0, 0, nw, nh);
    const d = x.getImageData(0, 0, nw, nh).data, N = nw * nh, f = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { // 모델 입력은 B,G,R 순서
      f[i] = (d[i * 4 + 2] / 255 - 0.485) / 0.229; f[N + i] = (d[i * 4 + 1] / 255 - 0.456) / 0.224; f[2 * N + i] = (d[i * 4] / 255 - 0.406) / 0.225;
    }
    const out = await S.det.run({ [S.det.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, nh, nw]) });
    const prob = out[S.det.outputNames[0]].data;
    const lab = new Int32Array(N), stack = new Int32Array(N), boxes = [], sx = w / nw, sy = h / nh;
    let id = 0;
    for (let p0 = 0; p0 < N; p0++) {
      if (prob[p0] <= 0.3 || lab[p0]) continue;
      id++; let sp = 0; stack[sp++] = p0; lab[p0] = id;
      let n = 0, sumx = 0, sumy = 0, sxx = 0, syy = 0, sxy = 0, sprob = 0; const px = [];
      while (sp) {
        const p = stack[--sp], yy = (p / nw) | 0, xx = p - yy * nw;
        n++; sumx += xx; sumy += yy; sxx += xx * xx; syy += yy * yy; sxy += xx * yy; sprob += prob[p]; px.push(p);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const X = xx + dx, Y = yy + dy; if (X < 0 || Y < 0 || X >= nw || Y >= nh) continue;
          const q = Y * nw + X; if (!lab[q] && prob[q] > 0.3) { lab[q] = id; stack[sp++] = q; }
        }
      }
      if (n < 12 || sprob / n < 0.5) continue;
      const mx = sumx / n, my = sumy / n, cxx = sxx / n - mx * mx, cyy = syy / n - my * my, cxy = sxy / n - mx * my;
      let th = 0.5 * Math.atan2(2 * cxy, cxx - cyy), ux = Math.cos(th), uy = Math.sin(th);
      if (Math.abs(ux) >= Math.abs(uy)) { if (ux < 0) { ux = -ux; uy = -uy; } } else if (uy < 0) { ux = -ux; uy = -uy; } // 가로면 오른쪽, 세로면 아래쪽을 향하게
      let a0 = 1e9, a1 = -1e9, b0 = 1e9, b1 = -1e9;
      for (const p of px) { const yy = (p / nw) | 0, xx = p - yy * nw, a = (xx - mx) * ux + (yy - my) * uy, b = -(xx - mx) * uy + (yy - my) * ux; if (a < a0) a0 = a; if (a > a1) a1 = a; if (b < b0) b0 = b; if (b > b1) b1 = b; }
      let L = a1 - a0 + 1, T = b1 - b0 + 1; if (Math.min(L, T) < 3) continue;
      const ca = (a0 + a1) / 2, cb = (b0 + b1) / 2, cx = mx + ca * ux - cb * uy, cy = my + ca * uy + cb * ux;
      const dd = L * T * 1.7 / (2 * (L + T)); L += 2 * dd; T += 2 * dd;
      // 원본 좌표로 (가로·세로 배율이 거의 같다고 봄)
      const k = (sx + sy) / 2;
      boxes.push({ cx: cx * sx, cy: cy * sy, L: L * k, S: T * k, ux, uy, vertical: Math.abs(uy) > Math.abs(ux) && L > T * 1.5 });
    }
    return boxes;
  }
  function cropBox(src, b) {
    const L = Math.max(2, Math.round(b.L)), T = Math.max(2, Math.round(b.S)), c = mk(L, T), x = c.getContext('2d', { willReadFrequently: true });
    x.fillStyle = '#fff'; x.fillRect(0, 0, L, T); x.imageSmoothingQuality = 'high';
    x.translate(L / 2, T / 2); x.rotate(-Math.atan2(b.uy, b.ux)); x.drawImage(src, -b.cx, -b.cy);
    return c;
  }

  /* ---------- 글자 읽기 (CTC) ---------- */
  async function rec(sess, crops) {
    const out = new Array(crops.length).fill(null), H = 48, B = 6;
    const order = crops.map((c, i) => i).sort((a, b) => crops[a].width / crops[a].height - crops[b].width / crops[b].height);
    for (let s = 0; s < order.length; s += B) {
      const idx = order.slice(s, s + B), ratio = Math.max(...idx.map(i => crops[i].width / crops[i].height));
      const W = Math.min(3200, Math.max(160, Math.ceil(H * ratio / 32) * 32)), N = idx.length, f = new Float32Array(N * 3 * H * W), ws = [];
      idx.forEach((i, k) => {
        const c = crops[i], w = Math.min(W, Math.max(8, Math.round(H * c.width / c.height))); ws.push(w);
        const t = mk(w, H), x = t.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high'; x.drawImage(c, 0, 0, w, H);
        const d = x.getImageData(0, 0, w, H).data, base = k * 3 * H * W;
        for (let y = 0; y < H; y++) for (let xx = 0; xx < w; xx++) {
          const p = (y * w + xx) * 4, q = y * W + xx;
          f[base + q] = d[p + 2] / 127.5 - 1; f[base + H * W + q] = d[p + 1] / 127.5 - 1; f[base + 2 * H * W + q] = d[p] / 127.5 - 1;
        }
      });
      const o = (await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', f, [N, 3, H, W]) }))[sess.outputNames[0]];
      const [, T, C] = o.dims, data = o.data;
      idx.forEach((i, k) => {
        const chars = [], xs = []; let sum = 0, prev = 0;
        for (let t = 0; t < T; t++) {
          const off = (k * T + t) * C; let best = 0, bv = data[off];
          for (let c = 1; c < C; c++) { const v = data[off + c]; if (v > bv) { bv = v; best = c; } }
          if (best !== 0 && best !== prev && best < S.chars.length) { chars.push(S.chars[best]); sum += bv; xs.push((t + 0.5) / T * W / ws[k]); }
          prev = best;
        }
        const conf = chars.length ? sum / chars.length : 0;
        out[i] = { text: respace(crops[i], chars, xs).join('').trim(), conf };
      });
    }
    return out;
  }
  async function recBest(crops) {
    const b = await rec(S.hw, crops), a = await rec(S.rec, crops);
    // 손글씨 모델은 인쇄체에서도 확신이 높게 나오는 편이라, 뚜렷이(0.1 이상) 더 확신하는 줄만 손글씨 결과로 바꾼다
    return a.map((x, i) => (x.text && x.conf + 0.1 >= b[i].conf) || !b[i].text ? x : b[i]);
  }
  /* 모델이 띄어쓰기를 자주 빠뜨려서, 글줄 이미지의 실제 빈칸을 보고 공백을 되살린다 */
  function respace(c, chars, xs) {
    if (chars.length < 2) return chars;
    const w = c.width, h = c.height, d = c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data, g = new Uint8Array(w * h), hist = new Int32Array(256);
    for (let i = 0; i < w * h; i++) { const v = (d[i * 4] * 299 + d[i * 4 + 1] * 587 + d[i * 4 + 2] * 114) / 1000 | 0; g[i] = v; hist[v]++; }
    let sumAll = 0; for (let i = 0; i < 256; i++) sumAll += i * hist[i];
    let wB = 0, sB = 0, mxv = -1, thr = 127; const tot = w * h;
    for (let t = 0; t < 256; t++) { wB += hist[t]; if (!wB) continue; const wF = tot - wB; if (!wF) break; sB += t * hist[t]; const mB = sB / wB, mF = (sumAll - sB) / wF, v = wB * wF * (mB - mF) * (mB - mF); if (v > mxv) { mxv = v; thr = t; } }
    const col = new Int32Array(w); let r0 = -1, r1 = -1;
    for (let y = 0; y < h; y++) {
      let n = 0; for (let x = 0; x < w; x++) if (g[y * w + x] <= thr) n++;
      if (!n || n > w * 0.6) continue; // 공책 줄·밑줄처럼 가로로 길게 이어진 선은 글자가 아님
      for (let x = 0; x < w; x++) if (g[y * w + x] <= thr) col[x]++;
      if (r0 < 0) r0 = y; r1 = y;
    }
    const th = r0 < 0 ? h : r1 - r0 + 1, lim = Math.max(1, th * 0.06), ink = x => col[x] > lim;
    let first = 0, last = w; while (first < w && !ink(first)) first++; while (last > 0 && !ink(last - 1)) last--;
    const inner = []; let st = -1;
    for (let x = 0; x < w; x++) { if (!ink(x) && st < 0) st = x; else if (ink(x) && st >= 0) { if (st > first && x < last) inner.push([st, x]); st = -1; } }
    let gl = th * 0.28; const nCh = chars.filter(c => c !== ' ').length;
    if (inner.length >= Math.max(3, nCh * 0.5)) { // 글자가 하나하나 떨어진 글씨: 틈 너비가 '글자 사이'와 '낱말 사이'로 갈리는 지점을 기준으로
      const ws = inner.map(([a, b]) => b - a).sort((a, b) => a - b); let best = 1, cut = null;
      for (let k = Math.max(1, Math.floor(ws.length / 3)); k < ws.length; k++) if (ws[k] >= th * 0.18 && ws[k - 1] > 0 && ws[k] / ws[k - 1] > best) { best = ws[k] / ws[k - 1]; cut = (ws[k] + ws[k - 1]) / 2; }
      gl = cut != null && best >= 1.45 ? cut : Math.max(gl, ws[ws.length >> 1] * 1.7);
    }
    const gaps = inner.filter(([a, b]) => b - a >= gl).map(([a, b]) => (a + b) / 2);
    if (!gaps.length) return chars;
    const out = [chars[0]];
    for (let j = 1; j < chars.length; j++) {
      if (chars[j] !== ' ' && out[out.length - 1] !== ' ') { const x0 = xs[j - 1] * w, x1 = xs[j] * w; if (gaps.some(m => x0 < m && m < x1)) out.push(' '); }
      out.push(chars[j]);
    }
    return out;
  }

  /* ---------- 전체: 찾기 → 읽기 → 순서 정리 ---------- */
  async function readLines(src, onProgress) {
    await load(onProgress);
    const w = src.width, h = src.height, boxes = await detect(src);
    const crops = [], keep = [];
    for (const b of boxes) { const c = cropBox(src, b); if (c.width >= 6 && c.height >= 6) { crops.push(c); keep.push(b); } }
    const res = crops.length ? await recBest(crops) : [], items = [];
    res.forEach((r, i) => {
      const t = r.text; if (!t || !/[가-힣A-Za-z0-9]/.test(t)) return;
      const core = t.replace(/\s+/g, '');
      if ((core.length <= 2 && r.conf < 0.6) || (!/[가-힣]/.test(core) && r.conf < 0.55)) return;
      const b = keep[i]; items.push({ b, text: t, conf: r.conf, ang: b.vertical ? null : Math.atan2(b.uy, b.ux) * 180 / Math.PI });
    });
    if (!items.length) return [];
    const corners = b => { const hx = b.ux * b.L / 2, hy = b.uy * b.L / 2, vx = -b.uy * b.S / 2, vy = b.ux * b.S / 2; return [[b.cx - hx - vx, b.cy - hy - vy], [b.cx + hx - vx, b.cy + hy - vy], [b.cx + hx + vx, b.cy + hy + vy], [b.cx - hx + vx, b.cy - hy + vy]]; };
    const out = (grp, block) => {
      let x1 = 1e9, y1 = 1e9, x2 = -1e9, y2 = -1e9;
      grp.forEach(g => corners(g.b).forEach(([x, y]) => { x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x); y2 = Math.max(y2, y); }));
      x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(w, x2); y2 = Math.min(h, y2);
      return { text: grp.map(g => g.text).join('  '), conf: Math.round(Math.min(...grp.map(g => g.conf)) * 1000) / 10, block, box: [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h] };
    };
    if (items.filter(i => i.ang == null).length >= Math.max(2, items.length * 0.6)) { // 세로쓰기: 오른쪽 줄부터
      items.sort((a, b) => b.b.cx - a.b.cx); return items.map(i => out([i], 0));
    }
    const angs = items.filter(i => i.ang != null && i.b.L > i.b.S * 2).map(i => i.ang).sort((a, b) => a - b);
    let A = angs.length ? angs[angs.length >> 1] : 0; if (Math.abs(A) < 1.5) A = 0;
    const ca = Math.cos(-A * Math.PI / 180), sa = Math.sin(-A * Math.PI / 180);
    items.forEach(i => { i.u = i.b.cx * ca - i.b.cy * sa; i.v = i.b.cx * sa + i.b.cy * ca; });
    items.sort((a, b) => a.v - b.v);
    const rows = [];
    for (const it of items) {
      const last = rows[rows.length - 1];
      if (last) { const lv = last.reduce((s, x) => s + x.v, 0) / last.length, lh = last.reduce((s, x) => s + x.b.S, 0) / last.length; if (Math.abs(it.v - lv) < Math.min(lh, it.b.S) * 0.55) { last.push(it); continue; } }
      rows.push([it]);
    }
    const lines = []; let block = 0, prev = null;
    for (const row of rows) {
      row.sort((a, b) => a.u - b.u);
      const v = row.reduce((s, x) => s + x.v, 0) / row.length, hh = row.reduce((s, x) => s + x.b.S, 0) / row.length;
      if (prev && v - prev[0] > (prev[1] + hh) * 1.25) block++;
      prev = [v, hh]; lines.push(out(row, block));
    }
    return lines;
  }
  window.IUX = { load, readLines, loaded: () => !!S.chars };
})();
