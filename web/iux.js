/* IUX 엔진 (웹) — 손글씨·흐린 글씨용 딥러닝 OCR을 브라우저 안에서 실행 (ONNX Runtime Web, 서버 전송 없음).
   app/iux_engine.py 와 같은 방식: 글줄 찾기 → 인쇄체/손글씨 두 모델로 읽고 더 확신하는 쪽 선택 → 띄어쓰기 복원 → 읽는 순서 정리 */
(function () {
  'use strict';
  const ORT = 'web/vendor/ort/', MODELS = 'models/';
  const S = { ready: null, det: null, rec: null, hw: null, zh: null, chars: null, zchars: null, zmask: null };
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
      const TOTAL = 54e6; let got = 0;
      const onp = n => { got += n; onProgress && onProgress(Math.min(0.99, got / TOTAL)); };
      const [det, rec, hw, zh, dict, zdict, ks] = await Promise.all([
        fetchBuf(MODELS + 'iux_det.onnx', onp), fetchBuf(MODELS + 'iux_rec_ko.onnx', onp), fetchBuf(MODELS + 'iux_rec_hw.onnx', onp), fetchBuf(MODELS + 'iux_rec_zh.onnx', onp),
        fetch(MODELS + 'iux_rec_ko_dict.txt').then(r => r.text()), fetch(MODELS + 'iux_rec_zh_dict.txt').then(r => r.text()), fetch(MODELS + 'iux_rec_zh_ks.txt').then(r => r.text()),
      ]);
      const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      S.det = await ort.InferenceSession.create(det, opt);
      S.rec = await ort.InferenceSession.create(rec, opt);
      S.hw = await ort.InferenceSession.create(hw, opt);
      const ch = dict.split('\n'); if (ch[ch.length - 1] === '') ch.pop();
      S.chars = ['', ...ch, ' '];
      S.zh = await ort.InferenceSession.create(zh, opt);
      const zc = zdict.split('\n'); if (zc[zc.length - 1] === '') zc.pop();
      S.zchars = ['', ...zc, ' '];
      // 한국에서 쓰는 한자(KS X 1001)와 숫자·영문·기호만 → 간체 대신 정자(번체)
      const ksSet = new Set([...ks.trim()]), isHz = c => /[\u3400-\u9fff\uf900-\ufaff]/.test(c);
      S.zmask = new Uint8Array(S.zchars.length); S.zchars.forEach((c, i) => { S.zmask[i] = (c && isHz(c) && !ksSet.has(c)) ? 1 : 0; });
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
        const chars = [], xs = [], ps = []; let sum = 0, prev = 0;
        for (let t = 0; t < T; t++) {
          const off = (k * T + t) * C; let best = 0, bv = data[off];
          for (let c = 1; c < C; c++) { const v = data[off + c]; if (v > bv) { bv = v; best = c; } }
          if (best !== 0 && best !== prev && best < S.chars.length) { chars.push(S.chars[best]); ps.push(bv); sum += bv; xs.push((t + 0.5) / T * W / ws[k]); }
          prev = best;
        }
        const conf = chars.length ? sum / chars.length : 0;
        let [rc, rp] = respace(crops[i], chars, xs, ps);
        while (rc.length && rc[0] === ' ') { rc = rc.slice(1); rp = rp.slice(1); }
        while (rc.length && rc[rc.length - 1] === ' ') { rc = rc.slice(0, -1); rp = rp.slice(0, -1); }
        out[i] = { text: rc.join(''), conf, ps: rp };
      });
    }
    return out;
  }
  const pick = (x, y) => (x.text && x.conf + 0.1 >= y.conf) || !y.text ? x : y; // 손글씨 모델은 뚜렷이(0.1 이상) 더 확신할 때만
  /* 글줄 이미지를 낱말 사이 빈칸에서 나눈 가로 구간들 */
  function wordSpans(c) {
    const w = c.width, h = c.height, d = c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data, g = new Uint8Array(w * h), hist = new Int32Array(256);
    for (let i = 0; i < w * h; i++) { const v = (d[i * 4] * 299 + d[i * 4 + 1] * 587 + d[i * 4 + 2] * 114) / 1000 | 0; g[i] = v; hist[v]++; }
    let sumAll = 0; for (let i = 0; i < 256; i++) sumAll += i * hist[i];
    let wB = 0, sB = 0, mxv = -1, thr = 127; const tot = w * h;
    for (let t = 0; t < 256; t++) { wB += hist[t]; if (!wB) continue; const wF = tot - wB; if (!wF) break; sB += t * hist[t]; const mB = sB / wB, mF = (sumAll - sB) / wF, v = wB * wF * (mB - mF) * (mB - mF); if (v > mxv) { mxv = v; thr = t; } }
    const col = new Int32Array(w); let r0 = -1, r1 = -1;
    for (let y = 0; y < h; y++) { let n = 0; for (let x = 0; x < w; x++) if (g[y * w + x] <= thr) n++; if (!n || n > w * 0.6) continue; for (let x = 0; x < w; x++) if (g[y * w + x] <= thr) col[x]++; if (r0 < 0) r0 = y; r1 = y; }
    const th = r0 < 0 ? h : r1 - r0 + 1, ink = x => col[x] > Math.max(1, th * 0.06);
    let first = 0, last = w; while (first < w && !ink(first)) first++; while (last > 0 && !ink(last - 1)) last--;
    if (first >= last) return [[0, w]];
    const gaps = []; let st = -1;
    for (let x = first; x < last; x++) { if (!ink(x) && st < 0) st = x; else if (ink(x) && st >= 0) { gaps.push([st, x]); st = -1; } }
    if (!gaps.length) return [[0, w]];
    const ws = gaps.map(([a, b]) => b - a).sort((a, b) => a - b); let lim = th * 0.35;
    if (ws.length >= 3) { let best = 1, cut = null; for (let k = Math.max(1, Math.floor(ws.length / 3)); k < ws.length; k++) if (ws[k] >= th * 0.2 && ws[k - 1] > 0 && ws[k] / ws[k - 1] > best) { best = ws[k] / ws[k - 1]; cut = (ws[k] + ws[k - 1]) / 2; } if (cut != null && best >= 1.45) lim = Math.max(th * 0.2, cut); }
    const cuts = gaps.filter(([a, b]) => b - a >= lim).map(([a, b]) => (a + b) >> 1); if (!cuts.length) return [[0, w]];
    const pad = Math.max(2, th * 0.1 | 0), e = [first, ...cuts, last], out = [];
    for (let i = 0; i < e.length - 1; i++) if (e[i + 1] - e[i] > 2) out.push([Math.max(0, e[i] - pad), Math.min(w, e[i + 1] + pad)]);
    return out;
  }
  function slice(c, a, b) { const t = mk(b - a, c.height); t.getContext('2d').drawImage(c, a, 0, b - a, c.height, 0, 0, b - a, c.height); return t; }
  /* 줄마다 인쇄체·손글씨 두 모델로 읽어 나은 쪽을 고르고, 확신이 낮은 긴 줄은 낱말 단위로도 읽어 더 나으면 바꾼다 */
  async function recBest(crops) {
    const b = await rec(S.hw, crops), a = await rec(S.rec, crops), line = a.map((x, i) => pick(x, b[i]));
    const pieces = [], owner = [];
    crops.forEach((c, i) => {
      if (c.width <= c.height * 7 || line[i].conf >= 0.85) return;
      const sp = wordSpans(c); if (sp.length < 2) return;
      sp.forEach(([x0, x1]) => { pieces.push(slice(c, x0, x1)); owner.push(i); });
    });
    if (!pieces.length) return line;
    const pa = await rec(S.rec, pieces), pb = await rec(S.hw, pieces), words = {};
    owner.forEach((i, k) => { const r = pick(pa[k], pb[k]); if (r.text) (words[i] = words[i] || []).push(r); });
    return line.map((l, i) => {
      const ws = words[i]; if (!ws) return l;
      const n = ws.reduce((s, r) => s + r.text.length, 0), cf = n ? ws.reduce((s, r) => s + r.conf * r.text.length, 0) / n : 0;
      if (!(cf > l.conf + 0.02)) return l;
      const ps = []; ws.forEach((r, k) => { if (k) ps.push(1); ps.push(...(r.ps || [])); });
      return { text: ws.map(r => r.text).join(' '), conf: cf, ps };
    });
  }
  /* 모델이 띄어쓰기를 자주 빠뜨려서, 글줄 이미지의 실제 빈칸을 보고 공백을 되살린다 */
  function respace(c, chars, xs, ps) {
    if (chars.length < 2) return [chars, ps];
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
      gl = cut != null && best >= 1.45 ? Math.max(cut, th * 0.3) : Math.max(gl, ws[ws.length >> 1] * 1.7);
    }
    const gaps = inner.filter(([a, b]) => b - a >= gl).map(([a, b]) => (a + b) / 2);
    if (!gaps.length) return [chars, ps];
    const out = [chars[0]], op = [ps[0]];
    for (let j = 1; j < chars.length; j++) {
      const pa = out[out.length - 1], pb = chars[j], num = (/[0-9,]/.test(pa)) && /[0-9,.]/.test(pb); // 1,100 · 2030.10 처럼 숫자 안은 띄우지 않음
      if (pb !== ' ' && pa !== ' ' && !num) { const x0 = xs[j - 1] * w, x1 = xs[j] * w; if (gaps.some(m => x0 < m && m < x1)) { out.push(' '); op.push(1); } }
      out.push(chars[j]); op.push(ps[j]);
    }
    return [out, op];
  }

  /* ---------- 전체: 찾기 → 읽기 → 순서 정리 ---------- */
  /* 띄어쓰기 처리 전의 글자·확신도·가로 위치 (한 줄씩) */
  async function recRaw(sess, chars, c, mask) {
    const H = 48, w0 = Math.max(8, Math.round(H * c.width / c.height)), W = Math.min(3200, Math.max(160, Math.ceil(w0 / 32) * 32)), w = Math.min(w0, W);
    const t = mk(w, H), x = t.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high'; x.drawImage(c, 0, 0, w, H);
    const d = x.getImageData(0, 0, w, H).data, f = new Float32Array(3 * H * W);
    for (let y = 0; y < H; y++) for (let xx = 0; xx < w; xx++) { const p = (y * w + xx) * 4, q = y * W + xx; f[q] = d[p + 2] / 127.5 - 1; f[H * W + q] = d[p + 1] / 127.5 - 1; f[2 * H * W + q] = d[p] / 127.5 - 1; }
    const o = (await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, H, W]) }))[sess.outputNames[0]];
    const [, T, C] = o.dims, data = o.data, ch = [], ps = [], xs = []; let prev = 0;
    for (let tt = 0; tt < T; tt++) {
      const off = tt * C; let best = 0, bv = data[off];
      for (let k = 1; k < C; k++) { if (mask && mask[k]) continue; const v = data[off + k]; if (v > bv) { bv = v; best = k; } }
      if (best !== 0 && best !== prev && best < chars.length) { ch.push(chars[best]); ps.push(bv); xs.push((tt + 0.5) / T * W / w); }
      prev = best;
    }
    return { ch, ps, xs };
  }
  const isHz = c => /^[\u3400-\u9fff\uf900-\ufaff]$/.test(c), isKo = c => /^[가-힣]$/.test(c);
  /* 한글 결과 cur 와 한자 모델 결과를 글자 위치로 맞춰 합친다 (app/iux_engine.py _hanja_line 과 같은 규칙) */
  async function hanjaLine(c, cur, z) {
    const strong = z.ch.map((ch, k) => isHz(ch) && z.ps[k] >= 0.9 ? k : -1).filter(k => k >= 0);
    if (!strong.length) return null;
    const core = z.ch.filter(ch => ch.trim()), nHz = core.filter(isHz).length;
    const zc = z.ps.length ? z.ps.reduce((a, b) => a + b, 0) / z.ps.length : 0;
    const raws = [await recRaw(S.rec, S.chars, c), await recRaw(S.hw, S.chars, c)];
    const mean = r => r.ps.length ? r.ps.reduce((a, b) => a + b, 0) / r.ps.length : 0;
    const k0 = mean(raws[0]) >= mean(raws[1]) ? raws[0] : raws[1];
    const gaps = z.xs.slice(1).map((v, i) => v - z.xs[i]).filter(v => v > 0), half = (gaps.length ? Math.min(...gaps) : 0.5 / Math.max(1, z.ch.length)) * 0.6;
    const nearest = x => { let j = -1, d = 1e9; k0.xs.forEach((v, i) => { const e = Math.abs(v - x); if (e <= half && e < d) { d = e; j = i; } }); return j; };
    if (nHz >= core.length * 0.6 && zc > cur.conf + 0.1 && zc >= 0.7) { // 줄 전체가 한자
      const ch = [...z.ch], ps = [...z.ps];
      ch.forEach((cc, k) => { if (isHz(cc) && ps[k] < 0.97) { const j = nearest(z.xs[k]); if (j >= 0 && isKo(k0.ch[j]) && k0.ps[j] >= 0.9) { ch[k] = k0.ch[j]; ps[k] = k0.ps[j]; } } });
      const [rc, rp] = respace(c, ch, z.xs, ps);
      return { text: rc.join('').trim(), conf: zc, ps: rp };
    }
    const used = new Set(), drop = new Set(), add = [];
    for (const k of strong) {
      const j = nearest(z.xs[k]);
      if (j < 0 || used.has(j)) { add.push([z.xs[k], z.ch[k], z.ps[k]]); continue; }
      used.add(j);
      if ((!isKo(k0.ch[j]) && k0.ps[j] < 0.9) || k0.ps[j] < 0.5) { drop.add(j); add.push([z.xs[k], z.ch[k], z.ps[k]]); }
    }
    if (!add.length) return null;
    const seq = k0.ch.map((ch, j) => [k0.xs[j], ch, k0.ps[j], j]).filter(t => !drop.has(t[3])).map(t => t.slice(0, 3)).concat(add).sort((a, b) => a[0] - b[0]);
    let [rc, rp] = respace(c, seq.map(t => t[1]), seq.map(t => t[0]), seq.map(t => t[2]));
    while (rc.length && rc[0] === ' ') { rc = rc.slice(1); rp = rp.slice(1); }
    while (rc.length && rc[rc.length - 1] === ' ') { rc = rc.slice(0, -1); rp = rp.slice(0, -1); }
    const real = rp.filter((p, i) => rc[i] !== ' ');
    return { text: rc.join(''), conf: real.length ? real.reduce((a, b) => a + b, 0) / real.length : 0, ps: rp };
  }
  /* 빨강·주황·노랑(직인·무궁화 바탕·형광펜)을 밝기로 바꿔 흰색에 가깝게 — 겹친 검은 글씨는 그대로 남음 */
  function decolor(src) {
    const w = src.width, h = src.height, c = mk(w, h), x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(src, 0, 0); const im = x.getImageData(0, 0, w, h), d = im.data; let warm = 0;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b), sat = mx ? (mx - mn) / mx * 255 : 0;
      let hue = 0; if (mx !== mn) { if (mx === r) hue = ((g - b) / (mx - mn) + 6) % 6; else if (mx === g) hue = (b - r) / (mx - mn) + 2; else hue = (r - g) / (mx - mn) + 4; } hue *= 30; // OpenCV 단위(0~180)
      let v; if ((hue <= 35 || hue >= 150) && sat >= 60) { v = mx; warm++; } else v = (r * 299 + g * 587 + b * 114) / 1000;
      d[i] = d[i + 1] = d[i + 2] = v;
    }
    x.putImageData(im, 0, 0); return [c, warm / (w * h)];
  }
  /* 흐린 글씨(옅은 타자기·복사본)면 대비를 늘린 캔버스, 아니면 null */
  function enhanceFaint(src) {
    const w = src.width, h = src.height, c = mk(w, h), x = c.getContext('2d', { willReadFrequently: true });
    x.drawImage(src, 0, 0); const im = x.getImageData(0, 0, w, h), d = im.data, hist = new Int32Array(256), n = w * h;
    for (let i = 0; i < d.length; i += 4) hist[(d[i] * 299 + d[i + 1] * 587 + d[i + 2] * 114) / 1000 | 0]++;
    const pct = q => { let a = 0; for (let k = 0; k < 256; k++) { a += hist[k]; if (a >= n * q) return k; } return 255; };
    const p1 = pct(0.005), p50 = pct(0.5);
    if (p1 < 110 || p50 - p1 < 25) return null;
    for (let i = 0; i < d.length; i += 4) {
      const g = (d[i] * 299 + d[i + 1] * 587 + d[i + 2] * 114) / 1000, t = Math.min(1, Math.max(0, (g - p1) / Math.max(1, p50 - p1)));
      d[i] = d[i + 1] = d[i + 2] = 255 * Math.pow(t, 1.6);
    }
    x.putImageData(im, 0, 0); return c;
  }

  /* ---------------- 표 칸 찾기 (app/table.py 와 같은 규칙) ---------------- */
  function findTables(src) {
    const W0 = src.width, H0 = src.height;
    let s = Math.min(2200 / Math.max(W0, H0), 2400 / W0, 3); if (Math.abs(s - 1) <= 0.05) s = 1;
    const w = Math.round(W0 * s), h = Math.round(H0 * s), cv = mk(w, h), cx = cv.getContext('2d', { willReadFrequently: true });
    cx.imageSmoothingQuality = 'high'; cx.drawImage(src, 0, 0, w, h);
    const d = cx.getImageData(0, 0, w, h).data, N = w * h, g = new Uint8Array(N), hist = new Int32Array(256);
    for (let i = 0; i < N; i++) { const v = (d[i * 4] * 299 + d[i * 4 + 1] * 587 + d[i * 4 + 2] * 114) / 1000 | 0; g[i] = v; hist[v]++; }
    let sumAll = 0; for (let i = 0; i < 256; i++) sumAll += i * hist[i];
    let wB = 0, sB = 0, mxv = -1, thr = 127;
    for (let t = 0; t < 256; t++) { wB += hist[t]; if (!wB) continue; const wF = N - wB; if (!wF) break; sB += t * hist[t]; const mB = sB / wB, mF = (sumAll - sB) / wF, v = wB * wF * (mB - mF) ** 2; if (v > mxv) { mxv = v; thr = t; } }
    // 적분 영상으로 국소 평균 (흐린 선·고르지 않은 조명)
    const I = new Float64Array((w + 1) * (h + 1));
    for (let y = 0; y < h; y++) { let row = 0; for (let x = 0; x < w; x++) { row += g[y * w + x]; I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row; } }
    const blk = Math.max(15, Math.floor(Math.min(w, h) / 25) | 1), r = blk >> 1;
    const b = new Uint8Array(N); let dark = 0;
    for (let i = 0; i < N; i++) if (g[i] <= thr) dark++;
    const useO = dark / N <= 0.25;
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1), cnt = (y1 - y0) * (x1 - x0);
        const sum = I[y1 * (w + 1) + x1] - I[y0 * (w + 1) + x1] - I[y1 * (w + 1) + x0] + I[y0 * (w + 1) + x0];
        const v = g[y * w + x]; if (v < sum / cnt - 8 || (useO && v <= thr)) b[y * w + x] = 1;
      }
    }
    const kh = Math.max(20, Math.floor(w / 50)), kv = Math.max(20, Math.floor(Math.min(h, w * 1.6) / 40));
    const m = new Uint8Array(N);
    const lineRuns = (len, cnt, at, k, gapK) => { // 한 방향: k 이상 이어진 검은 점만 남기고, gapK 이하 틈은 이음
      for (let a = 0; a < cnt; a++) {
        const runs = []; let st = -1;
        for (let t = 0; t <= len; t++) { const on = t < len && b[at(a, t)]; if (on && st < 0) st = t; else if (!on && st >= 0) { if (t - st >= k) runs.push([st, t]); st = -1; } }
        for (let i = 0; i < runs.length; i++) {
          let [p, q] = runs[i]; if (i + 1 < runs.length && runs[i + 1][0] - q < gapK) q = runs[i + 1][0];
          for (let t = p; t < q; t++) m[at(a, t)] = 1;
        }
      }
    };
    lineRuns(w, h, (y, x) => y * w + x, kh, Math.max(9, Math.floor(kh / 3)));
    lineRuns(h, w, (x, y) => y * w + x, kv, Math.max(9, Math.floor(kv / 3)));
    const dl = Math.max(2, Math.round(Math.min(w, h) / 400)), t1 = new Uint8Array(N), mm = new Uint8Array(N); // 모서리 이어 붙이기
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) for (let k = Math.max(0, x - dl); k <= Math.min(w - 1, x + dl); k++) t1[y * w + k] = 1;
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) if (t1[y * w + x]) for (let k = Math.max(0, y - dl); k <= Math.min(h - 1, y + dl); k++) mm[k * w + x] = 1;
    // 선으로 둘러싸인 빈 영역 = 칸 후보
    const lab = new Int32Array(N), st = [], stack = new Int32Array(N); let nl = 0;
    for (let i = 0; i < N; i++) {
      if (mm[i] || lab[i]) continue;
      nl++; let sp = 0, x1 = w, y1 = h, x2 = 0, y2 = 0, area = 0; stack[sp++] = i; lab[i] = nl;
      while (sp) { const p = stack[--sp], py = (p / w) | 0, px = p - py * w; area++; if (px < x1) x1 = px; if (px > x2) x2 = px; if (py < y1) y1 = py; if (py > y2) y2 = py;
        if (px > 0 && !mm[p - 1] && !lab[p - 1]) { lab[p - 1] = nl; stack[sp++] = p - 1; }
        if (px < w - 1 && !mm[p + 1] && !lab[p + 1]) { lab[p + 1] = nl; stack[sp++] = p + 1; }
        if (py > 0 && !mm[p - w] && !lab[p - w]) { lab[p - w] = nl; stack[sp++] = p - w; }
        if (py < h - 1 && !mm[p + w] && !lab[p + w]) { lab[p + w] = nl; stack[sp++] = p + w; } }
      st[nl] = [x1, y1, x2 + 1, y2 + 1, area];
    }
    const minw = Math.max(10, w * 0.008), minh = Math.max(10, h * 0.01), cand = [];
    for (let l = 1; l <= nl; l++) {
      const [x1, y1, x2, y2, area] = st[l], bw = x2 - x1, bh = y2 - y1;
      if (bw < minw || bh < minh || x1 <= 1 || y1 <= 1 || x2 >= w - 1 || y2 >= h - 1) continue;
      if (bw > w * 0.97 && bh > h * 0.97) continue;
      let filled = 0;
      for (let y = y1; y < y2; y++) { let a = -1, z = -1; for (let x = x1; x < x2; x++) if (lab[y * w + x] === l) { if (a < 0) a = x; z = x; } if (a >= 0) filled += z - a + 1; }
      if (filled < bw * bh * 0.8 || area < bw * bh * 0.35) continue;
      cand.push([x1, y1, x2, y2]);
    }
    const ins = (q, p) => q !== p && q[0] >= p[0] - 2 && q[1] >= p[1] - 2 && q[2] <= p[2] + 2 && q[3] <= p[3] + 2;
    const big = cand.filter(p => cand.filter(q => ins(q, p)).length < 2);
    const cells = big.filter(p => !big.some(q => ins(p, q)));
    if (cells.length < 4) return [];
    const gap = dl * 2 + 2 + Math.max(4, Math.round(Math.min(w, h) / 150)), par = cells.map((_, i) => i);
    const f = i => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
    for (let i = 0; i < cells.length; i++) for (let j = i + 1; j < cells.length; j++) {
      const a = cells[i], c = cells[j], ox = Math.min(a[2], c[2]) - Math.max(a[0], c[0]), oy = Math.min(a[3], c[3]) - Math.max(a[1], c[1]);
      if ((ox > 0 && -gap <= oy && oy <= gap * 0.5) || (oy > 0 && -gap <= ox && ox <= gap * 0.5) || (ox > 0 && oy > 0)) par[f(i)] = f(j);
    }
    const groups = {}; cells.forEach((c, i) => (groups[f(i)] = groups[f(i)] || []).push(c));
    const cluster = (vals, tol) => { vals.sort((a, b) => a - b); const gs = []; for (const v of vals) { const last = gs[gs.length - 1]; if (last && v - last[last.length - 1] <= tol) last.push(v); else gs.push([v]); } return gs.map(g => g.reduce((a, b) => a + b, 0) / g.length); };
    const near = (arr, v) => { let k = 0; arr.forEach((a, i) => { if (Math.abs(a - v) < Math.abs(arr[k] - v)) k = i; }); return k; };
    const tables = [];
    for (const cs of Object.values(groups)) {
      if (cs.length < 4) continue;
      const tol = dl * 2 + Math.max(4, Math.round(Math.min(w, h) / 300));
      const xs = cluster(cs.flatMap(c => [c[0], c[2]]), tol), ys = cluster(cs.flatMap(c => [c[1], c[3]]), tol);
      let out = [];
      for (const c of cs) {
        const c0 = near(xs, c[0]), c1 = near(xs, c[2]), r0 = near(ys, c[1]), r1 = near(ys, c[3]);
        if (c1 <= c0 || r1 <= r0) continue;
        out.push({ r: r0, c: c0, rs: r1 - r0, cs: c1 - c0, box: [c[0] / s, c[1] / s, c[2] / s, c[3] / s] });
      }
      if (out.length < 4 || xs.length < 2 || ys.length < 2) continue;
      const ur = [...new Set(out.flatMap(o => [o.r, o.r + o.rs]))].sort((a, b) => a - b), uc = [...new Set(out.flatMap(o => [o.c, o.c + o.cs]))].sort((a, b) => a - b);
      out.forEach(o => { const r0 = ur.indexOf(o.r), r1 = ur.indexOf(o.r + o.rs), c0 = uc.indexOf(o.c), c1 = uc.indexOf(o.c + o.cs); Object.assign(o, { r: r0, rs: r1 - r0, c: c0, cs: c1 - c0 }); });
      const rows = ur.length - 1, cols = uc.length - 1;
      if (!isGrid(out, rows, cols)) continue;
      out.sort((a, b) => a.r - b.r || a.c - b.c);
      tables.push({ box: [Math.min(...out.map(o => o.box[0])), Math.min(...out.map(o => o.box[1])), Math.max(...out.map(o => o.box[2])), Math.max(...out.map(o => o.box[3]))], rows, cols, cells: out });
    }
    return tables.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
  }
  function isGrid(cells, rows, cols) { // 세로 경계가 여러 행에 같은 자리 → 진짜 표 (형광펜 띠·공책 줄·손글씨 세로획 거름)
    if (cols < 2) return false;
    if (rows === 1) return true;
    const inner = []; for (let k = 1; k < cols; k++) if (cells.some(c => c.c === k)) inner.push(k);
    if (!inner.length) return false;
    const shared = inner.filter(k => new Set(cells.filter(c => c.c === k || c.c + c.cs === k).map(c => c.r)).size >= 2).length;
    return shared >= Math.max(1, inner.length * 0.5);
  }
  function splitBoxes(tables, boxes) { // 칸 세로 경계를 넘는 글줄은 칸마다 잘라 읽는다
    const out = [];
    for (const b of boxes) {
      const hx = Math.abs(b.ux * b.L / 2) + Math.abs(b.uy * b.S / 2), hy = Math.abs(b.uy * b.L / 2) + Math.abs(b.ux * b.S / 2);
      const x1 = b.cx - hx, x2 = b.cx + hx, y1 = b.cy - hy, y2 = b.cy + hy, hh = y2 - y1;
      const flat = !b.vertical && Math.abs(Math.atan2(b.uy, b.ux) * 180 / Math.PI) % 180 < 4 || Math.abs(Math.atan2(b.uy, b.ux) * 180 / Math.PI) % 180 > 176;
      const segs = [];
      if (flat && x2 - x1 > hh * 1.5) for (const t of tables) for (const c of t.cells) {
        const q = c.box, oy = Math.min(y2, q[3]) - Math.max(y1, q[1]), ox = Math.min(x2, q[2]) - Math.max(x1, q[0]);
        if (oy >= hh * 0.5 && ox >= Math.max(4, hh * 0.4)) segs.push([Math.max(x1, q[0]), Math.min(x2, q[2])]);
      }
      if (segs.length < 2) { out.push(b); continue; }
      segs.sort((a, c) => a[0] - c[0]).forEach(([a, c]) => out.push({ cx: (a + c) / 2, cy: (y1 + y2) / 2, L: c - a, S: b.S, ux: 1, uy: 0, vertical: false }));
    }
    return out;
  }
  function cellRows(pieces) { // 칸 안 조각: 위→아래 줄, 줄 안은 왼→오
    const ps = [...pieces].sort((a, c) => (a.bx[1] + a.bx[3]) - (c.bx[1] + c.bx[3])), rows = [];
    for (const p of ps) {
      const cy = (p.bx[1] + p.bx[3]) / 2, hh = p.bx[3] - p.bx[1], last = rows[rows.length - 1];
      if (last) { const ly = last.reduce((a, q) => a + (q.bx[1] + q.bx[3]) / 2, 0) / last.length, lh = last.reduce((a, q) => a + q.bx[3] - q.bx[1], 0) / last.length; if (Math.abs(cy - ly) < Math.min(lh, hh) * 0.55) { last.push(p); continue; } }
      rows.push([p]);
    }
    rows.forEach(r => r.sort((a, c) => a.bx[0] - c.bx[0]));
    return rows;
  }
  function joinCell(rows) { // 한두 글자씩 세로로 쌓인 칸(선/결, 접/수)은 붙여 쓴다
    const stacked = rows.length >= 2 && rows.every(r => r.map(p => p.text).join('').replace(/ /g, '').length <= 2);
    let text = ''; const chk = [];
    rows.forEach(r => r.forEach((p, j) => {
      if (stacked) { text += p.text.replace(/ /g, ''); return; }
      if (text) text += j === 0 ? '\n' : ' ';
      p.chk.forEach(k => chk.push(text.length + k)); text += p.text;
    }));
    return [text, chk, Math.min(...rows.flat().map(p => p.conf))];
  }
  function pruneColumns(t) { // 아주 좁은 칸이 잘게 나뉜 열은 글이 한쪽에만 있으면 옆 칸과 합침
    const Wt = (t.box[2] - t.box[0]) || 1;
    for (let guard = 0; guard < 200; guard++) {
      let done = false;
      for (let k = 1; k < t.cols && !done; k++) {
        const L = t.cells.filter(c => c.c + c.cs === k), R = t.cells.filter(c => c.c === k);
        if (!L.length || L.length !== R.length || !L.concat(R).some(c => c.box[2] - c.box[0] < Wt * 0.03)) continue;
        const pairs = []; let ok = true;
        for (const a of L) { const c = R.find(c => c.r === a.r && c.rs === a.rs); if (!c || (a.text && c.text)) { ok = false; break; } pairs.push([a, c]); }
        if (!ok) continue;
        for (const [a, c] of pairs) {
          if (c.text && !a.text) { a.text = c.text; a.conf = c.conf; a.check = c.check; a.pieces = c.pieces; }
          a.box = [Math.min(a.box[0], c.box[0]), Math.min(a.box[1], c.box[1]), Math.max(a.box[2], c.box[2]), Math.max(a.box[3], c.box[3])]; a.cs += c.cs;
          t.cells.splice(t.cells.indexOf(c), 1);
        }
        t.cells.forEach(c => { if (c.c >= k) c.c--; else if (c.c + c.cs > k) c.cs--; }); t.cols--; done = true;
      }
      if (!done) return;
    }
  }

  async function readLines(src, onProgress, extra) {
    await load(onProgress);
    const w = src.width, h = src.height;
    let srcs = [src]; const f = enhanceFaint(src); if (f) srcs = [f, src];
    const [dc, ratio] = decolor(srcs[0]); if (ratio > 0.003) srcs = [dc, ...srcs];
    let boxes = await detect(srcs[0]);
    const tables = findTables(src);
    if (tables.length) boxes = splitBoxes(tables, boxes); // 표 칸 경계를 넘는 글줄은 칸마다 잘라서 읽는다
    const per = srcs.map(() => []), keep = [];
    for (const b of boxes) { const cs = srcs.map(s => cropBox(s, b)); if (cs[0].width >= 6 && cs[0].height >= 6) { cs.forEach((c, k) => per[k].push(c)); keep.push(b); } }
    let res = keep.length ? await recBest(per[0]) : [];
    for (let k = 1; k < srcs.length && keep.length; k++) { const alt = await recBest(per[k]); res = res.map((a, i) => (alt[i].text && alt[i].conf > a.conf + 0.01) || !a.text ? alt[i] : a); }
    if (S.zh) for (let i = 0; i < res.length; i++) { const z = await recRaw(S.zh, S.zchars, per[0][i], S.zmask); const h = await hanjaLine(per[0][i], res[i], z); if (h && h.text) res[i] = h; } // 한자 읽기
    const items = [];
    res.forEach((r, i) => {
      const t = r.text; if (!t || !/[가-힣A-Za-z0-9\u3400-\u9fff]/.test(t)) return;
      const core = t.replace(/\s+/g, '');
      if ((core.length <= 2 && r.conf < 0.6) || (!/[가-힣\u3400-\u9fff]/.test(core) && r.conf < 0.55)) return;
      const chk = []; [...t].forEach((ch, k) => { if (/[0-9]/.test(ch) && r.ps && r.ps[k] < 0.9) chk.push(k); });
      const b = keep[i]; items.push({ b, text: t, conf: r.conf, chk, ang: b.vertical ? null : Math.atan2(b.uy, b.ux) * 180 / Math.PI });
    });
    const corners = b => { const hx = b.ux * b.L / 2, hy = b.uy * b.L / 2, vx = -b.uy * b.S / 2, vy = b.ux * b.S / 2; return [[b.cx - hx - vx, b.cy - hy - vy], [b.cx + hx - vx, b.cy + hy - vy], [b.cx + hx + vx, b.cy + hy + vy], [b.cx - hx + vx, b.cy - hy + vy]]; };
    const out = (grp, block) => {
      let x1 = 1e9, y1 = 1e9, x2 = -1e9, y2 = -1e9;
      grp.forEach(g => corners(g.b).forEach(([x, y]) => { x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x); y2 = Math.max(y2, y); }));
      x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(w, x2); y2 = Math.min(h, y2);
      const check = []; let off = 0; grp.forEach(g => { g.chk.forEach(k => check.push(off + k)); off += g.text.length + 2; });
      const o = { text: grp.map(g => g.text).join('  '), conf: Math.round(Math.min(...grp.map(g => g.conf)) * 1000) / 10, block, box: [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h] };
      if (check.length) o.check = check;
      return o;
    };
    items.forEach(it => { let x1 = 1e9, y1 = 1e9, x2 = -1e9, y2 = -1e9; corners(it.b).forEach(([x, y]) => { x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x); y2 = Math.max(y2, y); }); it.bx = [x1, y1, x2, y2]; });
    let rest = items;
    if (tables.length) {
      for (const it of items) { // 글 조각을 가장 많이 겹친 칸에 넣는다
        const [x1, y1, x2, y2] = it.bx, area = Math.max(1, (x2 - x1) * (y2 - y1)); let best = null, bv = 0, inside = 0;
        tables.forEach(t => { const q = t.box; if (x2 < q[0] || x1 > q[2] || y2 < q[1] || y1 > q[3]) return; inside = Math.max(inside, Math.max(0, Math.min(x2, q[2]) - Math.max(x1, q[0])) * Math.max(0, Math.min(y2, q[3]) - Math.max(y1, q[1]))); t.cells.forEach(c => { const b = c.box, ov = Math.max(0, Math.min(x2, b[2]) - Math.max(x1, b[0])) * Math.max(0, Math.min(y2, b[3]) - Math.max(y1, b[1])); if (ov > bv) { bv = ov; best = c; } }); });
        if (best && (bv >= area * 0.4 || inside >= area * 0.7)) { (best.pieces = best.pieces || []).push(it); it.inCell = true; }
      }
      tables.forEach(t => { t.cells.forEach(c => { if (c.pieces) { const [tx, ch, cf] = joinCell(cellRows(c.pieces)); c.text = tx; c.conf = cf; if (ch.length) c.check = ch; } else c.text = ''; }); pruneColumns(t); });
      rest = items.filter(i => !i.inCell);
    }
    const lines0 = groupLines(rest, w, h, out);
    if (!tables.length) { if (extra) extra.tables = []; return lines0; }
    // 표를 위치(위→아래)에 맞춰 줄 사이에 끼워 넣는다. 칸 하나 = 한 줄
    const units = lines0.map(l => ['l', l.box[1] * h, l]).concat(tables.map(t => ['t', t.box[1], t])).sort((a, b) => a[1] - b[1]);
    const outL = [], outT = []; let blk = -1, prv = null;
    const nb = q => [q[0] / w, q[1] / h, (q[2] - q[0]) / w, (q[3] - q[1]) / h].map(v => Math.round(v * 1e5) / 1e5);
    for (const [kind, , u] of units) {
      if (kind === 'l') { if (prv !== 'l' + u.block) blk++; prv = 'l' + u.block; u.block = blk; outL.push(u); continue; }
      blk++; prv = 't' + outT.length; const ti = outT.length, cells = [];
      const rowsOf = {};
      for (const c of [...u.cells].sort((a, b) => a.r - b.r || a.c - b.c)) {
        const cd = { r: c.r, c: c.c, rs: c.rs, cs: c.cs, box: nb(c.box), text: c.text };
        cells.push(cd);
        if (c.text) (rowsOf[c.r] = rowsOf[c.r] || []).push([c, cd]);
      }
      for (const r of Object.keys(rowsOf).map(Number).sort((a, b) => a - b)) { // 표의 한 행 = 한 줄, 칸 사이는 " | "
        const grp = rowsOf[r]; let x1 = 1e9, y1 = 1e9, x2 = -1e9, y2 = -1e9, txt = ''; const check = [];
        grp.forEach(([c, cd], k) => {
          c.pieces.forEach(p => { x1 = Math.min(x1, p.bx[0]); y1 = Math.min(y1, p.bx[1]); x2 = Math.max(x2, p.bx[2]); y2 = Math.max(y2, p.bx[3]); });
          if (k) txt += ' | ';
          (c.check || []).forEach(q => check.push(txt.length + q));
          txt += c.text.replace(/\n/g, ' '); cd.line = outL.length;
        });
        const ln = { text: txt, conf: Math.round(Math.min(...grp.map(([c]) => c.conf)) * 1000) / 10, block: blk, box: nb([Math.max(0, x1), Math.max(0, y1), Math.min(w, x2), Math.min(h, y2)]), table: ti, row: r };
        if (check.length) ln.check = check;
        outL.push(ln);
      }
      outT.push({ box: nb(u.box), rows: u.rows, cols: u.cols, cells });
    }
    if (extra) extra.tables = outT;
    return outL;
  }
  function groupLines(items, w, h, out) {
    if (!items.length) return [];
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
