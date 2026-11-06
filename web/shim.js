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
  const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';
  async function loadUrls(doc) {
    for (const p of doc.pages) for (const kind of ['image', 'thumb', 'proc']) {
      const k = `${doc.id}/${kind}/${p.index}`;
      if (URLS[k]) continue;
      const b = await getBlob(k);
      if (b) URLS[k] = URL.createObjectURL(b);
    }
  }
  window.WEB_FILEURL = function (id, kind, i) {
    return URLS[`${id}/${kind}/${i}`] || (kind === 'proc' ? URLS[`${id}/image/${i}`] : null) || BLANK;
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
    const bmp = await createImageBitmap(blob);
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

  async function ingest(file) {
    const id = uid(), name = file.name, pages = [];
    const ext = (name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
    const addPage = async (c, i, extra = {}) => {
      await setBlob(id, 'image', i, await toBlob(c));
      await setBlob(id, 'thumb', i, await thumbOf(c));
      pages.push({ index: i, file: 'image', proc: 'proc', thumb: 'thumb', width: c.width, height: c.height, dpi: extra.dpi || 300, result: extra.result || null, ...(extra.pdf_text ? { pdf_text: true } : {}) });
    };
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
    } else throw new Error('지원하지 않는 파일 형식입니다: ' + name);
    const doc = { id, name, created: Date.now() / 1000, pages };
    await putDoc(doc);
    return doc;
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
  async function runOcr(doc, idx, req) {
    const t0 = performance.now(), p = doc.pages[idx];
    const src = await blobToCanvas(await getBlob(`${doc.id}/image/${idx}`));
    const { canvas: pc, info } = preprocess(src, req.pre || {});
    await setBlob(doc.id, 'proc', idx, await toBlob(pc));
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
    const confs = lines.map(l => l.conf).filter(c => c != null);
    p.result = { engine: 'tesseract', lines, text: R.linesToText(lines), mean_conf: confs.length ? Math.round(confs.reduce((a, b) => a + b, 0) / confs.length * 10) / 10 : null,
      time_ms: Math.round(performance.now() - t0), pre: info, lang, at: Date.now() / 1000 };
    delete p.pdf_text;
    return p;
  }

  /* ---------------- 템플릿·항목 ---------------- */
  const TKEY = 'iuxocr.web.templates';
  function templates() { try { const t = JSON.parse(localStorage.getItem(TKEY)); if (Array.isArray(t)) return t; } catch (e) {} return JSON.parse(JSON.stringify(R.DEFAULT_TEMPLATES)); }
  function saveTemplates(items) {
    const clean = items.map(t => ({ ...t, id: t.id || uid().slice(0, 8), name: (t.name || '새 템플릿').trim(), fields: (t.fields || []).filter(f => (f.name || '').trim()) }));
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
    if (fmt === 'txt') return [new Blob(['﻿' + pages.map(p => (multi ? `===== ${p.index + 1} 페이지 =====\n` : '') + p.result.text).join('\n\n')], { type: 'text/plain' }), stem + '.txt'];
    if (fmt === 'md') return [new Blob([`# ${doc.name}\n\n` + pages.map(p => (multi ? `## ${p.index + 1} 페이지\n\n` : '') + p.result.text).join('\n\n')], { type: 'text/markdown' }), stem + '.md'];
    if (fmt === 'json') return [new Blob([JSON.stringify({ document: doc.name, pages: pages.map(p => ({ page: p.index + 1, width: p.width, height: p.height, ...p.result, lines: effectiveLines(p.result) })) }, null, 2)], { type: 'application/json' }), stem + '.json'];
    if (fmt === 'docx') { // Word가 여는 HTML 문서
      const html = `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><head><meta charset="utf-8"><style>body{font-family:'맑은 고딕';font-size:10.5pt}</style></head><body><h1>${escH(doc.name)}</h1>` +
        pages.map((p, i) => (multi ? `<h2>${p.index + 1} 페이지</h2>` : '') + p.result.text.split('\n').map(l => `<p>${escH(l) || '&nbsp;'}</p>`).join('') + (i < pages.length - 1 ? '<br style="page-break-before:always">' : '')).join('') + '</body></html>';
      return [new Blob(['﻿' + html], { type: 'application/msword' }), stem + '.doc'];
    }
    if (fmt === 'xlsx') {
      if (!window.XLSX) await loadScript(BASE + 'xlsx.full.min.js');
      const wb = XLSX.utils.book_new(), full = pages.map(p => p.result.text).join('\n');
      const tpl = templates().find(t => t.id === tid);
      if (tpl) XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['항목', '값'], ...R.runTemplate(full, tpl.fields).map(v => [v.name, v.value])]), '추출 항목');
      const rows = [['페이지', '줄', '텍스트', '신뢰도', '엔진', 'x', 'y', 'w', 'h']];
      pages.forEach(p => effectiveLines(p.result).forEach((l, j) => rows.push([p.index + 1, j + 1, l.text, l.conf, p.result.engine, ...(l.box || [null, null, null, null]).map(v => v == null ? null : Math.round(v * 1e4) / 1e4)])));
      XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), '줄별 인식결과');
      XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['항목', '값'], ...R.extractFields(full).map(f => [f.label, f.value])]), '공문서 항목');
      XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['페이지', '텍스트', '평균 신뢰도', '엔진'], ...pages.map(p => [p.index + 1, p.result.text, p.result.mean_conf, p.result.engine])]), '페이지 전체');
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
      paddle: { available: false, reason: '정확 모드는 PC 설치 버전에서 사용할 수 있습니다.' },
      ollama: { available: false, reason: 'AI 모드는 PC 설치 버전에서 사용할 수 있습니다.', models: [], vision: [] },
      web: true,
    };
    if (p === '/api/docs' && method === 'GET') {
      const docs = await allDocs();
      return docs.map(d => { const f = d.fields || {}; return { id: d.id, name: d.name, created: d.created, pages: d.pages.length, done: d.pages.filter(x => x.result).length, checked: !!f.checked, filled: (f.values || []).filter(v => v.value).length }; }).sort((a, b) => b.created - a.created);
    }
    if (p === '/api/upload') {
      const files = opt.body.getAll('files'), out = [];
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
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/pages\/(\d+)$/))) {
      const d = await getDoc(m[1]), pg = d.pages[+m[2]];
      const r = pg.result || { engine: 'manual', lines: [], mean_conf: null };
      r.text = body.text; if (body.lines) r.lines = body.lines; r.edited = true; pg.result = r; await putDoc(d); return pg;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/fields$/))) {
      const d = await getDoc(m[1]); d.fields = { ...body, at: Date.now() / 1000 }; await putDoc(d); return d.fields;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/autofill$/))) {
      const d = await getDoc(m[1]), t = templates().find(x => x.id === body.template_id) || fail('템플릿을 찾을 수 없습니다.');
      const text = fullText(d); if (!text.trim()) fail('인식된 텍스트가 없습니다.');
      d.fields = { template_id: t.id, values: R.runTemplate(text, t.fields).map(v => ({ name: v.name, value: v.value, note: v.note })), checked: false, edited: false, at: Date.now() / 1000 };
      await putDoc(d); return d.fields;
    }
    if ((m = p.match(/^\/api\/docs\/([a-f0-9]+)\/export$/))) {
      const d = await getDoc(m[1]);
      const [blob, name] = await exportDoc(d, body.format, body.pages, body.template_id);
      return new Response(blob, { headers: { 'content-type': blob.type, 'content-disposition': `attachment; filename*=UTF-8''${encodeURIComponent(name)}` } });
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
      return { templates: templates(), presets: Object.fromEntries(Object.entries(R.PRESETS).map(([k, v]) => [k, v[0]])), gongmun: R.GONGMUN_KEYS, formats: R.FORMATS };
    }
    if (p === '/api/extract') {
      const fields = body.fields || (templates().find(t => t.id === body.template_id) || fail('템플릿을 찾을 수 없습니다.')).fields;
      return { values: R.runTemplate(body.text, fields) };
    }
    if (p === '/api/helper/token') return { token: 'web' };
    if (p.startsWith('/api/helper/')) return { doc: null, fields: [] };
    if (p.startsWith('/api/batch') || p === '/api/pick-folder') fail('폴더 일괄 처리는 PC 설치 버전에서 사용할 수 있습니다.');
    fail('웹버전에서 지원하지 않는 기능입니다: ' + p);
  };
})();
