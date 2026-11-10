/* IUX OCR 웹버전 — 서버(app/server.py) 대신 브라우저 안에서 같은 API를 흉내 낸다.
   문서·이미지·결과는 이 브라우저의 IndexedDB에만 저장되고 어디로도 전송되지 않는다. */
(function () {
  'use strict';
  const R = window.IUXRules;
  const BASE = new URL('web/vendor/', document.baseURI).href;
  window.WEB = true;

  /* ---------------- IndexedDB ---------------- */
  let dbp = null;
  function db() {
    if (!dbp) dbp = new Promise((res, rej) => {
      const q = indexedDB.open('iux-ocr', 1);
      q.onupgradeneeded = () => { q.result.createObjectStore('docs'); q.result.createObjectStore('blobs'); };
      q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
    });
    return dbp;
  }
  async function tx(store, mode, fn) {
    const d = await db();
    return new Promise((res, rej) => {
      const t = d.transaction(store, mode), s = t.objectStore(store);
      const r = fn(s);
      t.oncomplete = () => res(r && 'result' in r ? r.result : undefined);
      t.onerror = () => rej(t.error);
    });
  }
  const getDoc = id => tx('docs', 'readonly', s => s.get(id));
  const putDoc = doc => tx('docs', 'readwrite', s => s.put(doc, doc.id));
  const allDocs = () => tx('docs', 'readonly', s => s.getAll());
  const getBlob = k => tx('blobs', 'readonly', s => s.get(k));
  const putBlob = (k, b) => tx('blobs', 'readwrite', s => s.put(b, k));
  async function delDoc(id) {
    await tx('docs', 'readwrite', s => s.delete(id));
    await tx('blobs', 'readwrite', s => s.delete(IDBKeyRange.bound(id + '/', id + '/￿')));
    Object.keys(URLS).filter(k => k.startsWith(id + '/')).forEach(k => { URL.revokeObjectURL(URLS[k]); delete URLS[k]; });
  }

  /* ---------------- 이미지 URL ---------------- */
  const URLS = {};
  const KEYS = {};                       // 문서별 쪽 순서 → 저장 키 (쪽이 끼워지거나 빠져도 이미지가 섞이지 않게)
  const pk = p => p.key || String(p.index); // 예전 문서는 쪽 번호가 곧 키
  const noteKeys = d => { if (d && d.pages) KEYS[d.id] = d.pages.map(pk); return d; };
  const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';
  async function loadUrls(doc) {
    noteKeys(doc);
    for (const p of doc.pages) for (const kind of ['image', 'thumb', 'proc']) {
      if (p.pending) continue;
      const k = `${doc.id}/${kind}/${pk(p)}`;
      if (URLS[k]) continue;
      const b = await getBlob(k);
      if (b) URLS[k] = URL.createObjectURL(b);
    }
  }
  window.WEB_FILEURL = function (id, kind, i) {
    const key = (KEYS[id] || [])[i] ?? String(i);
    return URLS[`${id}/${kind}/${key}`] || (kind === 'proc' ? URLS[`${id}/image/${key}`] : null) || BLANK;
  };
  async function setBlob(id, kind, i, blob) {
    const k = `${id}/${kind}/${i}`;
    await putBlob(k, blob);
    if (URLS[k]) URL.revokeObjectURL(URLS[k]);
    URLS[k] = URL.createObjectURL(blob);
  }

  /* ---------------- 캔버스 도우미 ---------------- */
  const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  const toBlob = (c, type = 'image/png', q) => new Promise(r => c.toBlob(r, type, q));
  async function blobToCanvas(blob) {
    let bmp; try { bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' }); } catch (e) { bmp = await createImageBitmap(blob); } /* EXIF 회전 반영 */
    const c = canvas(bmp.width, bmp.height);
    c.getContext('2d').drawImage(bmp, 0, 0);
    bmp.close && bmp.close();
    return c;
  }
  async function thumbOf(c) {
    const s = 220 / c.width, t = canvas(220, Math.max(1, Math.round(c.height * s)));
    t.getContext('2d').drawImage(c, 0, 0, t.width, t.height);
    return toBlob(t, 'image/jpeg', 0.78);
  }

  /* ---------------- 불러오기 (이미지·PDF·TIFF) ---------------- */
  let pdfjs = null;
  async function loadPdfjs() {
    if (!pdfjs) {
      pdfjs = await import(BASE + 'pdf.min.mjs');
      pdfjs.GlobalWorkerOptions.workerSrc = BASE + 'pdf.worker.min.mjs';
    }
    return pdfjs;
  }
  function loadScript(src) {
    return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('불러오기 실패: ' + src)); document.head.appendChild(s); });
  }
  async function tiffPages(buf) {
    if (!window.UTIF) await loadScript(BASE + 'UTIF.js');
    const ifds = UTIF.decode(buf);
    return ifds.map(ifd => {
      UTIF.decodeImage(buf, ifd);
      const rgba = UTIF.toRGBA8(ifd), c = canvas(ifd.width, ifd.height);
      c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer), ifd.width, ifd.height), 0, 0);
      return c;
    });
  }
  const uid = () => Array.from(crypto.getRandomValues(new Uint8Array(6)), b => b.toString(16).padStart(2, '0')).join('');

  async function ingest(file) { return ingestMany([file], file.name); }
  async function ingestMany(files, docName, existing, noSave) {
    const id = existing ? existing.id : uid(), pages = existing ? existing.pages : [];
    const start = pages.length;
    let name = '';
    const addPage = async (c, _i, extra = {}) => {
      const i = pages.length, key = uid();
      await setBlob(id, 'image', key, await toBlob(c, 'image/jpeg', 0.92)); /* PNG보다 훨씬 작게 저장 */
      await setBlob(id, 'thumb', key, await thumbOf(c));
      pages.push({ index: i, key, file: 'image', proc: 'proc', thumb: 'thumb', width: c.width, height: c.height, dpi: extra.dpi || 300, result: extra.result || null, source: name, ...(extra.pdf_text ? { pdf_text: true } : {}) });
    };
    try { for (const file of files) { name = file.name; await addFile(file); } }
    catch (e) { if (existing) pages.splice(start); else await delDoc(id); throw e; }
    if (noSave) return existing;
    const doc = existing || { id, name: docName, created: Date.now() / 1000, pages };
    if (existing) doc.files = (doc.files || []).concat(files.map(f => f.name));
    else if (files.length > 1) doc.files = files.map(f => f.name);
    await putDoc(doc);
    return doc;

    async function addFile(file) {
    const ext = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
    if (ext === '.pdf') {
      const lib = await loadPdfjs();
      const pdf = await lib.getDocument({ data: await file.arrayBuffer() }).promise;
      for (let i = 0; i < pdf.numPages; i++) {
        const page = await pdf.getPage(i + 1);
        let scale = 300 / 72; const vp0 = page.getViewport({ scale: 1 });
        if (vp0.width * vp0.height * scale * scale > 30e6) scale = Math.sqrt(30e6 / (vp0.width * vp0.height));
        const vp = page.getViewport({ scale }), c = canvas(Math.round(vp.width), Math.round(vp.height));
        const ctx = c.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
        await page.render({ canvasContext: ctx, viewport: vp }).promise;
        let extra = { dpi: Math.round(scale * 72) };
        try {
          const tc = await page.getTextContent();
          let text = '', lastY = null;
          for (const it of tc.items) { const y = it.transform[5]; if (lastY !== null && Math.abs(y - lastY) > 2) text += '\n'; text += it.str; lastY = y; if (it.hasEOL) { text += '\n'; lastY = null; } }
          text = text.replace(/\n{2,}/g, '\n').trim();
          if (text.length >= 20) extra = { ...extra, pdf_text: true, result: { engine: 'pdf-text', lines: text.split('\n').map(t => ({ text: t, conf: null, box: null, block: 0 })), text, mean_conf: null, time_ms: 0 } };
        } catch (e) { /* 텍스트 층 없음 */ }
        await addPage(c, i, extra);
      }
    } else if (ext === '.tif' || ext === '.tiff') {
      const cs = await tiffPages(await file.arrayBuffer());
      for (let i = 0; i < cs.length; i++) await addPage(cs[i], i);
    } else if (/^\.(png|jpe?g|bmp|webp|gif)$/.test(ext)) {
      await addPage(await blobToCanvas(file), 0);
    } else throw new Error('지원하지 않는 파일 형식입니다: ' + file.name);
    }
  }

  /* ---------------- 전처리 (app/preprocess.py 와 같은 흐름) ---------------- */
  function grayData(c) {
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, g = new Uint8ClampedArray(c.width * c.height);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) g[j] = (d[i] * 299 + d[i + 1] * 587 + d[i + 2] * 114) / 1000;
    return g;
  }
  function grayCanvas(g, w, h) {
    const c = canvas(w, h), id = c.getContext('2d').createImageData(w, h);
    for (let i = 0, j = 0; j < g.length; i += 4, j++) { id.data[i] = id.data[i + 1] = id.data[i + 2] = g[j]; id.data[i + 3] = 255; }
    c.getContext('2d').putImageData(id, 0, 0);
    return c;
  }
  function otsu(g) {
    const hist = new Array(256).fill(0); for (const v of g) hist[v]++;
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, max = 0, th = 128;
    for (let i = 0; i < 256; i++) { wB += hist[i]; if (!wB) continue; const wF = g.length - wB; if (!wF) break; sumB += i * hist[i]; const mB = sumB / wB, mF = (sum - sumB) / wF, v = wB * wF * (mB - mF) ** 2; if (v > max) { max = v; th = i; } }
    return th;
  }
  function rotated(src, deg, bg) {
    const r = deg * Math.PI / 180, cos = Math.abs(Math.cos(r)), sin = Math.abs(Math.sin(r));
    const w = Math.round(src.width * cos + src.height * sin), h = Math.round(src.width * sin + src.height * cos);
    const c = canvas(w, h), x = c.getContext('2d');
    x.fillStyle = bg; x.fillRect(0, 0, w, h); x.translate(w / 2, h / 2); x.rotate(r); x.imageSmoothingQuality = 'high';
    x.drawImage(src, -src.width / 2, -src.height / 2);
    return c;
  }
  function estimateSkew(gc) {
    const s = Math.min(1, 900 / gc.width), w = Math.round(gc.width * s), h = Math.round(gc.height * s);
    const sm = canvas(w, h); sm.getContext('2d').drawImage(gc, 0, 0, w, h);
    const g = grayData(sm), th = otsu(g), inv = new Uint8ClampedArray(g.length);
    let ink = 0; for (let i = 0; i < g.length; i++) { inv[i] = g[i] < th ? 255 : 0; ink += inv[i]; }
    if (ink / g.length < 1) return 0;
    const bin = grayCanvas(inv, w, h);
    const score = a => { const r = rotated(bin, -a, '#000'), d = r.getContext('2d').getImageData(0, 0, r.width, r.height).data; let mean = 0; const rows = new Float64Array(r.height); for (let y = 0; y < r.height; y++) { let t = 0; for (let x = 0; x < r.width; x++) t += d[(y * r.width + x) * 4]; rows[y] = t; mean += t; } mean /= r.height; let v = 0; for (const t of rows) v += (t - mean) ** 2; return v; };
    let best = 0, bs = score(0);
    for (let a = -10; a <= 10.001; a += 0.5) { const sc = score(a); if (sc > bs) { bs = sc; best = a; } }
    const c0 = best;
    for (let a = c0 - 0.5; a <= c0 + 0.501; a += 0.1) { const sc = score(a); if (sc > bs) { bs = sc; best = a; } }
    return Math.round(best * 100) / 100;
  }
  function preprocess(src, o) {
    const info = { skew: 0, rotate: 0, scale: 1 };
    let c = src;
    if (o.upscale && c.width < 1600) {
      const s = Math.min(2.5, 2000 / c.width), u = canvas(Math.round(c.width * s), Math.round(c.height * s)), x = u.getContext('2d');
      x.imageSmoothingQuality = 'high'; x.drawImage(c, 0, 0, u.width, u.height); c = u; info.scale = Math.round(s * 1000) / 1000;
    }
    if (o.deskew) { const a = estimateSkew(c); if (Math.abs(a) >= 0.15) { c = rotated(c, -a, '#fff'); info.skew = a; } }
    let g = grayData(c);
    if (o.contrast) { // 1~99% 구간 늘이기
      const hist = new Array(256).fill(0); for (const v of g) hist[v]++;
      let lo = 0, hi = 255, acc = 0; const n = g.length;
      for (let i = 0; i < 256; i++) { acc += hist[i]; if (acc > n * 0.01) { lo = i; break; } }
      acc = 0; for (let i = 255; i >= 0; i--) { acc += hist[i]; if (acc > n * 0.01) { hi = i; break; } }
      if (hi - lo > 20) for (let i = 0; i < n; i++) g[i] = (g[i] - lo) * 255 / (hi - lo);
    }
    if (o.binarize) { const th = otsu(g); for (let i = 0; i < g.length; i++) g[i] = g[i] < th ? 0 : 255; }
    return { canvas: grayCanvas(g, c.width, c.height), info };
  }

  /* ---------------- Tesseract.js ---------------- */
  const workers = {};
  async function worker(lang) {
    if (!window.Tesseract) await loadScript(BASE + 'tesseract.min.js');
    if (!workers[lang]) workers[lang] = Tesseract.createWorker(lang.split('+'), 1, {
      workerPath: BASE + 'worker.min.js', corePath: BASE + 'core/', langPath: BASE + 'lang', gzip: true,
      cacheMethod: 'write',
    });
    return workers[lang];
  }
  /* 90/180/270도 돌아간 페이지 감지 (Tesseract OSD) */
  let osdW = null;
  async function osdWorker() {
    if (!window.Tesseract) await loadScript(BASE + 'tesseract.min.js');
    if (!osdW) osdW = Tesseract.createWorker('osd', 0, { workerPath: BASE + 'worker.min.js', corePath: BASE + 'core/', langPath: BASE + 'lang', gzip: true, legacyCore: true, legacyLang: true });
    return osdW;
  }
  // 선택 위/아래 경계에 걸쳐 잘린 옆 줄 글자 조각(ㅇ 등)을 흰색으로 지움
function trimEdgeFragments(x, ox, oy, w, h) {
  if (h < 12) return;
  const img = x.getImageData(ox, oy, w, h), d = img.data, lum = new Uint8Array(w * h);
  let sum = 0;
  for (let i = 0; i < w * h; i++) { const v = (d[i * 4] * 299 + d[i * 4 + 1] * 587 + d[i * 4 + 2] * 114) / 1000; lum[i] = v; sum += v; }
  const th = Math.min(160, (sum / (w * h)) * 0.75), minInk = Math.max(1, w * 0.002), rows = [];
  for (let y = 0; y < h; y++) { let n = 0; for (let k = y * w, e = k + w; k < e; k++) if (lum[k] < th) n++; rows.push(n > minInk); }
  const bands = []; let st = -1;
  rows.forEach((r, i) => { if (r && st < 0) st = i; else if (!r && st >= 0) { bands.push([st, i]); st = -1; } });
  if (st >= 0) bands.push([st, h]);
  if (bands.length < 2) return;
  const tall = Math.max(...bands.map(([a, b]) => b - a));
  x.fillStyle = '#fff';
  [bands[0], bands[bands.length - 1]].forEach(([a, b]) => { if ((a === 0 || b === h) && (b - a) < tall * 0.6) x.fillRect(ox, oy + a, w, b - a); });
}
  async function detectRotation(c) {
    // OSD는 표·도장 많은 한글 공문서에서 자주 틀림 → 힌트로만 쓰고, 실제로 읽어 보아 가장 또렷한 방향을 고른다
    const s = Math.min(1, 1600 / Math.max(c.width, c.height)), sm = canvas(Math.round(c.width * s), Math.round(c.height * s));
    sm.getContext('2d').drawImage(c, 0, 0, sm.width, sm.height);
    let rot = 0, conf = 0;
    try { const { data } = await (await osdWorker()).detect(sm); rot = ((data.orientation_degrees || 0) % 360 + 360) % 360; conf = data.orientation_confidence || 0; }
    catch (e) { console.warn('방향 감지(OSD) 실패', e); }
    if (rot === 0 && conf >= 2.5) return 0;
    let base;
    try { base = await textScore(sm); } catch (e) { return [90, 180, 270].includes(rot) && conf >= 6 ? rot : 0; }
    if (base.good >= 10 && base.good >= base.tot * 0.6) return 0;
    const cands = ([90, 180, 270].includes(rot) ? [rot] : []).concat([90, 270, 180].filter(r => r !== rot));
    let best = 0, bestG = base.good;
    for (const r of cands) {
      let sc; try { sc = await textScore(rotateCw(sm, r)); } catch (e) { continue; }
      if (sc.good >= Math.max(15, base.good * 1.5) && sc.good >= sc.tot * 0.5 && sc.good > bestG) {
        best = r; bestG = sc.good;
        if (sc.good >= sc.tot * 0.75) break;
      }
    }
    return best;
  }
  async function textScore(c) {
    const s = Math.min(1, 1400 / Math.max(c.width, c.height)), sm = canvas(Math.round(c.width * s), Math.round(c.height * s));
    sm.getContext('2d').drawImage(c, 0, 0, sm.width, sm.height);
    const w = await worker('kor+eng');
    await w.setParameters({ tessedit_pageseg_mode: '6' }); // psm 3은 누운 글줄도 읽어 버려 판단이 흐려짐
    const { data } = await w.recognize(sm, {}, { text: false, blocks: true });
    let good = 0, tot = 0;
    (data.blocks || []).forEach(b => (b.paragraphs || []).forEach(p => (p.lines || []).forEach(l => (l.words || []).forEach(wd => {
      const n = (wd.text || '').replace(/[^가-힣A-Za-z0-9]/g, '').length; if (!n) return;
      tot += n; if ((wd.confidence || 0) >= 65) good += n;
    }))));
    return { good, tot };
  }
  function rotateCw(c, deg) {
    deg = ((deg % 360) + 360) % 360; if (!deg) return c;
    const sw = deg % 180 !== 0, r = canvas(sw ? c.height : c.width, sw ? c.width : c.height), x = r.getContext('2d');
    x.translate(r.width / 2, r.height / 2); x.rotate(deg * Math.PI / 180); x.drawImage(c, -c.width / 2, -c.height / 2);
    return r;
  }
  async function rotateStored(doc, idx, deg) {
    const p = doc.pages[idx];
    const key = pk(p);
    const c = rotateCw(await blobToCanvas(await getBlob(`${doc.id}/image/${key}`)), deg);
    await setBlob(doc.id, 'image', key, await toBlob(c, 'image/jpeg', 0.92));
    await setBlob(doc.id, 'thumb', key, await thumbOf(c));
    await tx('blobs', 'readwrite', s => s.delete(`${doc.id}/proc/${key}`));
    const k = `${doc.id}/proc/${key}`; if (URLS[k]) { URL.revokeObjectURL(URLS[k]); delete URLS[k]; }
    p.width = c.width; p.height = c.height; p.result = null; delete p.pdf_text;
    return c;
  }
  async function runOcr(doc, idx, req) {
    const t0 = performance.now(), p = doc.pages[idx];
    if (p.pending) throw new Error('아직 불러오지 않은 파일입니다.');
    let src = await blobToCanvas(await getBlob(`${doc.id}/image/${pk(p)}`));
    let rotated = 0;
    if ((req.pre || {}).orient !== false) {
      const deg = await detectRotation(src);
      if (deg) { src = await rotateStored(doc, idx, deg); rotated = deg; }
    }
    const { canvas: pc, info } = preprocess(src, req.pre || {});
    if (rotated) info.rotate_fixed = rotated;
    await setBlob(doc.id, 'proc', pk(p), await toBlob(pc, 'image/jpeg', 0.9));
    const lang = ['kor+eng', 'kor', 'eng'].includes(req.lang) ? req.lang : 'kor+eng';
    const w = await worker(lang);
    await w.setParameters({ tessedit_pageseg_mode: String(req.psm || 4), preserve_interword_spaces: '1' });
    const { data } = await w.recognize(pc, {}, { text: true, blocks: true });
    const W = pc.width, H = pc.height, lines = [];
    (data.blocks || []).forEach((b, bi) => (b.paragraphs || []).forEach((pa, pi) => (pa.lines || []).forEach(l => {
      let text = (l.text || '').replace(/\s+$/, '');
      if (!text.trim()) return;
      if (req.fix_spacing !== false && lang.includes('kor') && R.looksSplit(text)) text = R.fixSyllableSpacing(text);
      const bb = l.bbox;
      lines.push({ text, conf: Math.round(l.confidence * 10) / 10, box: [bb.x0 / W, bb.y0 / H, (bb.x1 - bb.x0) / W, (bb.y1 - bb.y0) / H], block: bi * 100 + pi });
    })));
    let second = null;
    if ((req.pre || {}).second !== false) second = await secondPass(pc, lines, w, lang, req.fix_spacing !== false);
    await w.setParameters({ tessedit_pageseg_mode: String(req.psm || 4), preserve_interword_spaces: '1' });
    const confs = lines.map(l => l.conf).filter(c => c != null);
    p.result = { engine: 'tesseract', lines, text: R.linesToText(lines), second, mean_conf: confs.length ? Math.round(confs.reduce((a, b) => a + b, 0) / confs.length * 10) / 10 : null,
      time_ms: Math.round(performance.now() - t0), pre: info, lang, at: Date.now() / 1000 };
    delete p.pdf_text;
    return p;
  }

  /* ---------------- 2차 인식 ----------------
     신뢰도 낮은 줄만 잘라 키우고(작은 글씨)·가로로 늘리고(길쭉한 글씨)·획을 굵혀(얇은 글씨)
     여러 버전을 한 장에 모아 한 번에 다시 읽고, 가장 또렷한 버전이 확실히 나을 때만 바꾼다 */
  const SECOND = { conf: 80, max: 60, h: 64, variants: [[1, 0], [1.4, 0], [1.8, 0], [1.6, 1]] };
  const nrm = t => (t || '').replace(/\s+/g, '');
  const cnt = (t, rx) => ((t || '').match(rx) || []).length;
  function thicken(c) { // 2x2 최소값 필터 = 검은 획을 1px 굵게
    const x = c.getContext('2d'), im = x.getImageData(0, 0, c.width, c.height), d = im.data, W = c.width, H = c.height, o = new Uint8ClampedArray(d);
    for (let y = 0; y < H; y++) for (let i = 0; i < W; i++) {
      let m = 255; for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) { const xx = Math.min(W - 1, i + dx), yy = Math.min(H - 1, y + dy); m = Math.min(m, d[(yy * W + xx) * 4]); }
      const k = (y * W + i) * 4; o[k] = o[k + 1] = o[k + 2] = m;
    }
    im.data.set(o); x.putImageData(im, 0, 0); return c;
  }
  async function secondPass(pc, lines, w, lang, fixSp) {
    const W = pc.width, H = pc.height;
    const cand = lines.map((l, i) => i).filter(i => lines[i].box && (lines[i].conf == null || lines[i].conf < SECOND.conf) && nrm(lines[i].text).length)
      .sort((a, b) => (lines[a].conf || 0) - (lines[b].conf || 0)).slice(0, SECOND.max);
    if (!cand.length) return null;
    const tiles = [], slots = [], gap = 28; let y = 0;
    for (const i of cand) {
      const [bx, by, bw, bh] = lines[i].box, padx = Math.round(bh * H * 0.5), pady = Math.round(bh * H * 0.25);
      const X0 = Math.max(0, Math.floor(bx * W) - padx), Y0 = Math.max(0, Math.floor(by * H) - pady);
      const X1 = Math.min(W, Math.round((bx + bw) * W) + padx), Y1 = Math.min(H, Math.round((by + bh) * H) + pady);
      if (X1 - X0 < 6 || Y1 - Y0 < 6) continue;
      const sc = Math.max(1, Math.min(4, SECOND.h / (Y1 - Y0))), cw = Math.round((X1 - X0) * sc), ch = Math.round((Y1 - Y0) * sc);
      const base = canvas(cw, ch), bx2 = base.getContext('2d');
      bx2.fillStyle = '#fff'; bx2.fillRect(0, 0, cw, ch); bx2.drawImage(pc, X0, Y0, X1 - X0, Y1 - Y0, 0, 0, cw, ch);
      trimEdgeFragments(bx2, 0, 0, cw, ch);
      for (let v = 0; v < SECOND.variants.length; v++) {
        const [sx, th] = SECOND.variants[v], g = canvas(Math.round(cw * sx), ch), gx = g.getContext('2d');
        gx.fillStyle = '#fff'; gx.fillRect(0, 0, g.width, ch); gx.imageSmoothingQuality = 'high'; gx.drawImage(base, 0, 0, g.width, ch);
        if (th) thicken(g);
        tiles.push(g); slots.push([i, v, y + gap, y + gap + ch]); y += gap + ch;
      }
    }
    if (!tiles.length) return null;
    const TW = Math.max(...tiles.map(t => t.width)) + 32, sheet = canvas(TW, y + gap), sx_ = sheet.getContext('2d');
    sx_.fillStyle = '#fff'; sx_.fillRect(0, 0, TW, sheet.height);
    tiles.forEach((t, k) => sx_.drawImage(t, 16, slots[k][2]));
    await w.setParameters({ tessedit_pageseg_mode: '6', preserve_interword_spaces: '1' });
    const { data } = await w.recognize(sheet, {}, { text: false, blocks: true });
    const got = new Map();
    (data.blocks || []).forEach(b => (b.paragraphs || []).forEach(pa => (pa.lines || []).forEach(l => {
      const t = (l.text || '').trim(); if (!nrm(t)) return;
      const cy = (l.bbox.y0 + l.bbox.y1) / 2, sl = slots.find(([, , a, b2]) => cy >= a - gap / 2 && cy <= b2 + gap / 2);
      if (!sl) return; const key = sl[0] + ':' + sl[1];
      if (!got.has(key)) got.set(key, { i: sl[0], ls: [] }); got.get(key).ls.push({ t, c: l.confidence, x: l.bbox.x0 });
    })));
    const best = new Map();
    for (const { i, ls } of got.values()) {
      ls.sort((a, b) => a.x - b.x);
      let nt = R.dropStrayLines(ls.map(l => [l.t, l.c])).join(' ').trim();
      if (!nt) continue;
      if (!(lines[i].text || '').includes('  ')) nt = nt.replace(/ {2,}/g, ' ');
      if (fixSp && lang.includes('kor') && R.looksSplit(nt)) nt = R.fixSyllableSpacing(nt);
      const n = ls.reduce((a, l) => a + nrm(l.t).length, 0); if (!n) continue;
      const nc = ls.reduce((a, l) => a + l.c * nrm(l.t).length, 0) / n;
      if (!best.has(i) || nc > best.get(i).nc) best.set(i, { nt, nc });
    }
    let improved = 0;
    for (const [i, { nt, nc }] of best) {
      const o = lines[i], oc = o.conf || 0, on = nrm(o.text).length, nn = nrm(nt).length;
      if (nc < 60 || nc < oc + 5) continue;
      if (on && !(nn >= on * 0.6 && nn <= on * 1.6 + 2)) continue;
      if (cnt(nt, /[가-힣]/g) < cnt(o.text, /[가-힣]/g) - 1 || cnt(nt, /[A-Za-z]/g) > cnt(o.text, /[A-Za-z]/g) + 2) continue;
      o.text_1st = o.text; o.conf_1st = o.conf; o.text = nt; o.conf = Math.round(nc * 10) / 10; o.second = true; improved++;
    }
    return { tried: cand.length, improved };
  }

  /* ---------------- 템플릿·항목 ---------------- */
  const TKEY = 'iuxocr.web.templates';
  function templates() { try { const t = JSON.parse(localStorage.getItem(TKEY)); if (Array.isArray(t)) return t; } catch (e) {} return JSON.parse(JSON.stringify(R.DEFAULT_TEMPLATES)); }
  const AUTO = { id: '__auto__', name: '자동 — 문서에서 항목 찾기', fields: [], auto: true };
  const tplById = id => id === AUTO.id ? AUTO : templates().find(t => t.id === id);
  const valuesFor = (t, text) => t.auto ? R.autoFields(text).map(v => ({ ...v, type: 'auto', note: '' })) : R.runTemplate(text, t.fields);
  function saveTemplates(items) {
    const clean = items.filter(t => !t.auto && t.id !== AUTO.id).map(t => ({ ...t, id: t.id || uid().slice(0, 8), name: (t.name || '새 템플릿').trim(), fields: (t.fields || []).filter(f => (f.name || '').trim()) }));
    try { localStorage.setItem(TKEY, JSON.stringify(clean)); } catch (e) {}
    return clean;
  }
  const fullText = doc => doc.pages.filter(p => p.result).map(p => p.result.text).join('\n');

  /* ---------------- 내보내기 ---------------- */
  function effectiveLines(r) {
    const lines = r.lines || [], text = r.text || '';
    if (text === R.linesToText(lines)) return lines;
    const nw = text.split('\n').filter(t => t.trim());
    if (nw.length === lines.length) return lines.map((l, i) => ({ ...l, text: nw[i] }));
    let block = 0; const out = [];
    for (const t of text.split('\n')) { if (!t.trim()) { block++; continue; } out.push({ text: t, conf: null, box: null, block }); }
    return out;
  }
  const escH = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  async function exportDoc(doc, fmt, pageIdx, tid) {
    const pages = doc.pages.filter(p => p.result && (pageIdx == null || pageIdx.includes(p.index)));
    if (!pages.length) throw new Error('인식된 페이지가 없습니다. 먼저 OCR을 실행하세요.');
    const stem = doc.name.replace(/\.[^.]+$/, ''), multi = pages.length > 1;
    const lab = p => p.label || `${p.index + 1} 페이지`;
    if (fmt === 'txt') return [new Blob(['﻿' + pages.map(p => (multi ? `===== ${lab(p)} =====\n` : '') + p.result.text).join('\n\n')], { type: 'text/plain' }), stem + '.txt'];
    if (fmt === 'md') return [new Blob([`# ${doc.name}\n\n` + pages.map(p => (multi ? `## ${lab(p)}\n\n` : '') + p.result.text).join('\n\n')], { type: 'text/markdown' }), stem + '.md'];
    if (fmt === 'json') return [new Blob([JSON.stringify({ document: doc.name, pages: pages.map(p => ({ page: p.index + 1, ...(p.doc_name ? { document: p.doc_name } : {}), width: p.width, height: p.height, ...p.result, lines: effectiveLines(p.result) })) }, null, 2)], { type: 'application/json' }), stem + '.json'];
    if (fmt === 'docx') { // Word가 여는 HTML 문서
      const html = `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><head><meta charset="utf-8"><style>body{font-family:'맑은 고딕';font-size:10.5pt}</style></head><body><h1>${escH(doc.name)}</h1>` +
        pages.map((p, i) => (multi ? `<h2>${escH(lab(p))}</h2>` : '') + p.result.text.split('\n').map(l => `<p>${escH(l) || '&nbsp;'}</p>`).join('') + (i < pages.length - 1 ? '<br style="page-break-before:always">' : '')).join('') + '</body></html>';
      return [new Blob(['﻿' + html], { type: 'application/msword' }), stem + '.doc'];
    }
    if (fmt === 'xlsx') {
      if (!window.XLSX) await loadScript(BASE + 'xlsx.full.min.js');
      const wb = XLSX.utils.book_new(), full = pages.map(p => p.result.text).join('\n');
      const tpl = tplById(tid);
      if (tpl) XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['항목', '값'], ...valuesFor(tpl, full).map(v => [v.name, v.value])]), '추출 항목');
      const rows = [['페이지', '줄', '텍스트', '신뢰도', '엔진', 'x', 'y', 'w', 'h']];
      pages.forEach(p => effectiveLines(p.result).forEach((l, j) => rows.push([lab(p), j + 1, l.text, l.conf, p.result.engine, ...(l.box || [null, null, null, null]).map(v => v == null ? null : Math.round(v * 1e4) / 1e4)])));
      XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), '줄별 인식결과');
      XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['항목', '값'], ...R.extractFields(full).map(f => [f.label, f.value])]), '공문서 항목');
      XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['페이지', '텍스트', '평균 신뢰도', '엔진'], ...pages.map(p => [lab(p), p.result.text, p.result.mean_conf, p.result.engine])]), '페이지 전체');
      const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
      return [new Blob([out], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), stem + '.xlsx'];
    }
    throw new Error('웹버전에서는 지원하지 않는 형식입니다 (PC 버전 이용)');
  }

  /* ---------------- 라우터 ---------------- */
  const J = (o, status = 200) => o;
  const fail = m => { throw new Error(m); };
  window.WEB_API = async function (path, opt = {}) {
    const method = (opt.method || (opt.json !== undefined || opt.body ? 'POST' : 'GET')).toUpperCase();
    const body = opt.json;
    const url = new URL(path, location.href), p = url.pathname.replace(/^.*?\/api\//, '/api/');
    let m;
    if (p === '/api/engines') return {
      tesseract: { available: true, version: '5 (브라우저)', has_kor: true, path: '이 브라우저 (Tesseract.js, 오프라인 모델 포함)' },
      iux: { available: false, reason: '정확(IUX 엔진)은 PC 설치 버전에서 사용할 수 있습니다.' },
      paddle: { available: false, reason: '정확 모드는 PC 설치 버전에서 사용할 수 있습니다.' },
      ollama: { available: false, reason: 'AI 모드는 PC 설치 버전에서 사용할 수 있습니다.', models: [], vision: [] },
      web: true,
    };
    if (p === '/api/docs' && method === 'GET') {
      const docs = await allDocs();
      return docs.map(d => { const f = d.fields || {}; return { id: d.id, name: d.name, created: d.created, pages: d.pages.length, done: d.pages.filter(x => x.result).length, pending: d.pages.filter(x => x.pending).length, checked: !!f.checked, filled: (f.values || []).filter(v => v.value).length }; }).sort((a, b) => b.created - a.created);
    }
    if (p === '/api/docs/lazy') {
      const id = uid();
      const doc = { id, name: body.name, created: Date.now() / 1000, lazy: true,
        pages: body.files.map((f, i) => ({ index: i, key: uid(), pending: true, path: f, source: f.split('/').pop(), result: null })) };
      await putDoc(doc); noteKeys(doc); return doc;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/fill$/))) {
      const d = await getDoc(m[1]), key = opt.body.get('key'), file = opt.body.get('files');
      const pos = d.pages.findIndex(x => x.key === key);
      if (pos < 0) fail('대기 중인 쪽을 찾을 수 없습니다.');
      const holder = d.pages[pos];
      const tmp = { id: d.id, pages: [] };
      await ingestMany([file], '', tmp, true);
      tmp.pages.forEach(q => { q.path = holder.path; });
      d.pages.splice(pos, 1, ...tmp.pages);
      d.pages.forEach((q, i) => { q.index = i; });
      await putDoc(d); await loadUrls(d);
      return { doc: d, keys: tmp.pages.map(q => q.key) };
    }
    if (p === '/api/upload') {
      const files = opt.body.getAll('files'), out = [];
      if (opt.body.get('append_to') && files.length) { const ex = await getDoc(opt.body.get('append_to')); out.push(await ingestMany(files, '', ex)); await loadUrls(ex); return out; }
      if (opt.body.get('merge') === '1' && files.length) { out.push(await ingestMany(files, opt.body.get('doc_name') || files[0].name)); return out; }
      for (const f of files) out.push(await ingest(f));
      return out;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)$/))) {
      if (method === 'DELETE') { await delDoc(m[1]); return { ok: true }; }
      const d = await getDoc(m[1]) || fail('문서를 찾을 수 없습니다.'); await loadUrls(d); return d;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/ocr$/))) {
      const d = await getDoc(m[1]); const pg = await runOcr(d, body.page, body);
      const fresh = await getDoc(m[1]); fresh.pages[body.page] = pg; await putDoc(fresh); return pg;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/auto-orient$/))) {
      const d = await getDoc(m[1]), fixed = [];
      for (const i of body.pages) {
        if (!d.pages[i]) continue;
        if (d.pages[i].pending) continue;
        const deg = await detectRotation(await blobToCanvas(await getBlob(`${d.id}/image/${pk(d.pages[i])}`)));
        if (deg) { await rotateStored(d, i, deg); fixed.push({ page: i, deg }); }
      }
      await putDoc(d); await loadUrls(d); return { fixed, doc: d };
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/rotate$/))) {
      const d = await getDoc(m[1]);
      for (const i of body.pages) if (d.pages[i]) await rotateStored(d, i, body.deg);
      await putDoc(d); return d;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/ocr-region$/))) {
      const d = await getDoc(m[1]), i = body.page;
      const key = pk(d.pages[i]);
      const k = body.view === 'proc' && await getBlob(`${d.id}/proc/${key}`) ? 'proc' : 'image';
      const src = await blobToCanvas(await getBlob(`${d.id}/${k}/${key}`));
      const [x0, y0, x1, y1] = body.box;
      const X0 = Math.max(0, Math.floor(x0 * src.width)), Y0 = Math.max(0, Math.floor(y0 * src.height));
      const X1 = Math.min(src.width, Math.round(x1 * src.width)), Y1 = Math.min(src.height, Math.round(y1 * src.height));
      if (X1 - X0 < 8 || Y1 - Y0 < 8) fail('선택 영역이 너무 작습니다.');
      const sc = (Y1 - Y0) < 120 ? Math.min(3, 120 / (Y1 - Y0)) : 1, M = 16;
      const cw = Math.round((X1 - X0) * sc), ch = Math.round((Y1 - Y0) * sc);
      const c = canvas(cw + M * 2, ch + M * 2), x = c.getContext('2d');
      x.fillStyle = '#fff'; x.fillRect(0, 0, c.width, c.height); x.imageSmoothingQuality = 'high';
      x.drawImage(src, X0, Y0, X1 - X0, Y1 - Y0, M, M, cw, ch);
      trimEdgeFragments(x, M, M, cw, ch);
      const lang = ['kor+eng', 'kor', 'eng'].includes(body.lang) ? body.lang : 'kor+eng';
      const w = await worker(lang);
      await w.setParameters({ tessedit_pageseg_mode: '6', preserve_interword_spaces: '1' });
      const { data } = await w.recognize(c, {}, { text: true, blocks: true });
      const raw = [];
      (data.blocks || []).forEach(b => (b.paragraphs || []).forEach(pa => (pa.lines || []).forEach(l => { let t = (l.text || '').trim(); if (t) { if (R.looksSplit(t)) t = R.fixSyllableSpacing(t); raw.push([t, l.confidence]); } })));
      return { text: R.dropStrayLines(raw).join('\n'), mean_conf: data.confidence };
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/pages\/(\d+)$/))) {
      const d = await getDoc(m[1]), pg = d.pages[+m[2]];
      const r = pg.result || { engine: 'manual', lines: [], mean_conf: null };
      r.text = body.text; if (body.lines) r.lines = body.lines; r.edited = true; pg.result = r; await putDoc(d); return pg;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/fields$/))) {
      const d = await getDoc(m[1]); d.fields = { ...body, at: Date.now() / 1000 }; await putDoc(d); return d.fields;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/autofill$/))) {
      const d = await getDoc(m[1]), t = tplById(body.template_id) || fail('템플릿을 찾을 수 없습니다.');
      const text = fullText(d); if (!text.trim()) fail('인식된 텍스트가 없습니다.');
      d.fields = { template_id: t.id, values: valuesFor(t, text).map(v => ({ name: v.name, value: v.value, note: v.note || '' })), checked: false, edited: false, at: Date.now() / 1000 };
      await putDoc(d); return d.fields;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/export$/))) {
      const d = await getDoc(m[1]);
      const [blob, name] = await exportDoc(d, body.format, body.pages, body.template_id);
      return new Response(blob, { headers: { 'content-type': blob.type, 'content-disposition': `attachment; filename*=UTF-8''${encodeURIComponent(name)}` } });
    }
    if (p === '/api/export-all') {
      const docs = (await allDocs()).sort((a, b) => a.created - b.created);
      const ids = body.doc_ids || docs.map(d => d.id);
      const pages = []; let skipped = 0;
      for (const id of ids) {
        const d = await getDoc(id); if (!d) continue;
        const multi = d.pages.length > 1;
        for (const pg of d.pages) {
          if (!pg.result) { skipped++; continue; }
          pages.push({ ...pg, label: multi ? `${d.name} · ${pg.index + 1}쪽` : d.name, doc_name: d.name });
        }
      }
      if (!pages.length) fail("인식된 페이지가 없습니다. 먼저 '모두 인식'을 실행하세요.");
      pages.forEach((q, i) => { q.index = i; });
      const name = body.name || `OCR_전체_${ids.length}건_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`;
      const [blob, fname] = await exportDoc({ id: 'all', name, pages }, body.format, null, body.template_id);
      return new Response(blob, { headers: { 'content-type': blob.type, 'content-disposition': `attachment; filename*=UTF-8''${encodeURIComponent(fname)}`, 'x-skipped-pages': String(skipped) } });
    }
    if (p === '/api/refine') {
      if (body.ai) fail('AI 재가공은 PC 설치 버전(Ollama)에서 사용할 수 있습니다.');
      const t = body.text || '';
      switch (body.mode) {
        case 'clean': return { text: R.cleanText(t) };
        case 'mask': return R.maskPii(t);
        case 'outline': return { text: R.toOutline(t) };
        case 'fields': return { fields: R.extractFields(t) };
        case 'body': return { text: R.extractBody(t) };
        case 'normalize': return { text: R.normalize(t) };
      }
      fail('알 수 없는 재가공');
    }
    if (p === '/api/templates') {
      if (method === 'PUT') return { templates: saveTemplates(body) };
      return { templates: [AUTO].concat(templates()), presets: Object.fromEntries(Object.entries(R.PRESETS).map(([k, v]) => [k, v[0]])), gongmun: R.GONGMUN_KEYS, formats: R.FORMATS };
    }
    if (p === '/api/extract') {
      if (body.fields) return { values: R.runTemplate(body.text, body.fields) };
      return { values: valuesFor(tplById(body.template_id) || fail('템플릿을 찾을 수 없습니다.'), body.text) };
    }
    if (p === '/api/storage') {
      let est = {}; try { est = await navigator.storage.estimate(); } catch (e) {}
      let persisted = false; try { persisted = await navigator.storage.persisted() || await navigator.storage.persist(); } catch (e) {}
      return { usage: est.usage || 0, quota: est.quota || 0, persisted };
    }
    if (p === '/api/helper/token') return { token: 'web' };
    if (p.startsWith('/api/helper/')) return { doc: null, fields: [] };
    if (p.startsWith('/api/batch') || p === '/api/pick-folder') fail('폴더 일괄 처리는 PC 설치 버전에서 사용할 수 있습니다.');
    fail('웹버전에서 지원하지 않는 기능입니다: ' + p);
  };
})();
