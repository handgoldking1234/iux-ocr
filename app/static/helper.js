/* IUX OCR 시스템 입력 도우미 — 북마클릿으로 업무 시스템 화면 위에 띄워 항목 값을 칸에 채운다.
   주소와 키는 IUX OCR이 설치 링크를 만들 때 채운다. */
(function () {
  var API = '__API__', TOKEN = '__TOKEN__';
  if (window.__iuxHelper) { window.__iuxHelper.toggle(); return; }
  var KEY = 'iuxocr.map.' + location.host + location.pathname;
  var S = { data: null, armed: -1, seq: false, map: {}, offline: false };
  try { S.map = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
  function saveMap() { try { localStorage.setItem(KEY, JSON.stringify(S.map)); } catch (e) {} }

  var host = document.createElement('div');
  host.style.cssText = 'position:fixed;z-index:2147483647;top:16px;right:16px;';
  var root = host.attachShadow ? host.attachShadow({ mode: 'open' }) : host;
  root.innerHTML = '<style>' +
    '*{box-sizing:border-box;margin:0;padding:0;font-family:"Malgun Gothic","Apple SD Gothic Neo",sans-serif;letter-spacing:-.01em}' +
    '.p{width:340px;background:#fff;color:#101828;border-radius:12px;box-shadow:0 12px 40px rgba(0,0,0,.28);border:1px solid #d0d5dd;font-size:12.5px;overflow:hidden}' +
    '.h{display:flex;align-items:center;gap:6px;padding:9px 10px;background:#1f2a44;color:#fff;cursor:move;user-select:none}' +
    '.h b{font-size:12.5px;flex:1}.h button{color:#fff;opacity:.8}.h button:hover{opacity:1}' +
    'button{border:0;background:none;cursor:pointer;font:inherit;color:inherit}' +
    '.doc{padding:8px 10px;border-bottom:1px solid #eaecf0;display:flex;align-items:center;gap:6px;background:#f8f9fb}' +
    '.doc .nm{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:700}' +
    '.nav{width:26px;height:26px;border-radius:6px;border:1px solid #d0d5dd;background:#fff;font-weight:800}' +
    '.tb{display:flex;gap:6px;padding:8px 10px;border-bottom:1px solid #eaecf0}' +
    '.b{flex:1;height:30px;border-radius:7px;border:1px solid #d0d5dd;background:#fff;font-weight:700;font-size:12px}' +
    '.b.pri{background:#2f5bff;border-color:#2f5bff;color:#fff}.b.on{background:#101828;color:#fff;border-color:#101828}' +
    '.l{max-height:46vh;overflow:auto}' +
    '.r{display:flex;align-items:center;gap:6px;padding:7px 10px;border-bottom:1px solid #f2f4f7;cursor:pointer}' +
    '.r:hover{background:#f5f8ff}.r.arm{background:#e8eeff;box-shadow:inset 3px 0 0 #2f5bff}' +
    '.r .k{width:74px;flex:none;font-weight:700;color:#475467;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}' +
    '.r .v{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}' +
    '.r .v.e{color:#98a2b3}.r .m{font-size:10px;font-weight:800;color:#12a150;flex:none}' +
    '.r .k{cursor:pointer}.r .c{width:24px;height:24px;border-radius:6px;flex:none;color:#475467}.r .c:hover{background:#eaecf0}' +
    '.st{padding:8px 10px;font-size:11.5px;color:#475467;background:#f8f9fb;min-height:32px;line-height:1.45}' +
    '.st.w{color:#2f5bff;font-weight:700}.ft{display:flex;gap:6px;padding:8px 10px;border-top:1px solid #eaecf0}' +
    'textarea{width:100%;height:90px;border:1px solid #d0d5dd;border-radius:7px;padding:6px;font-size:11px;font-family:Consolas,monospace}' +
    '.off{padding:10px;display:flex;flex-direction:column;gap:6px}.off .b{flex:none}.min .l,.min .tb,.min .ft,.min .st,.min .doc,.min .off{display:none}' +
    '</style><div class="p" id="p"></div>';
  document.documentElement.appendChild(host);
  var P = root.getElementById ? root.getElementById('p') : root.querySelector('#p');

  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function api(path, body) {
    var u = API + path + (path.indexOf('?') < 0 ? '?' : '&') + 't=' + TOKEN;
    var o = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {};
    return fetch(u, o).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
  }
  function load() {
    status('불러오는 중…');
    return api('/api/helper/current').then(function (d) { S.data = d; S.offline = false; render(); })
      .catch(function () { S.offline = true; render(); });
  }
  function move(dir, check) {
    if (S.offline) return;
    var before = S.data && S.data.doc ? S.data.doc.id : null;
    api('/api/helper/move', { dir: dir, check: !!check }).then(function (d) { S.data = d; S.armed = -1; S.seq = false; render();
      var same = d.doc && d.doc.id === before;
      status(same ? (dir > 0 ? '마지막 문서입니다.' : '첫 문서입니다.') + (check ? ' 확인 완료로 표시했습니다.' : '') : (check ? '확인 완료 처리하고 다음 문서를 불러왔습니다. 자동 채우기를 누르세요.' : ''), same); }).catch(function () { status('IUX OCR에 연결할 수 없습니다.'); });
  }

  /* ---------- 칸 찾기 / 채우기 ---------- */
  function fillable(el) {
    if (!el || el.nodeType !== 1) return null;
    if (el.isContentEditable) return el;
    var t = el.tagName;
    if (t === 'TEXTAREA' || t === 'SELECT') return el;
    if (t === 'INPUT' && !/^(button|submit|reset|checkbox|radio|file|image|hidden)$/i.test(el.type)) return el;
    return null;
  }
  function framePath(win) {
    var path = [];
    try { while (win !== window.top) { var par = win.parent, i; for (i = 0; i < par.frames.length; i++) if (par.frames[i] === win) break; path.unshift(i); win = par; } } catch (e) {}
    return path;
  }
  function winByPath(path) {
    var w = window.top;
    try { for (var i = 0; i < path.length; i++) w = w.frames[path[i]]; return w.document ? w : null; } catch (e) { return null; }
  }
  function cssEsc(s) { return window.CSS && CSS.escape ? CSS.escape(s) : s.replace(/[^\w-]/g, '\\$&'); }
  function selectorOf(el) {
    var doc = el.ownerDocument;
    if (el.id && doc.querySelectorAll('#' + cssEsc(el.id)).length === 1) return '#' + cssEsc(el.id);
    if (el.name) {
      var sel = el.tagName.toLowerCase() + '[name="' + el.name.replace(/"/g, '\\"') + '"]';
      var all = doc.querySelectorAll(sel);
      if (all.length === 1) return sel;
      for (var i = 0; i < all.length; i++) if (all[i] === el) return sel + '@' + i;
    }
    var parts = [], n = el;
    while (n && n.nodeType === 1 && n !== doc.body) {
      var idx = 1, sib = n;
      while ((sib = sib.previousElementSibling)) if (sib.tagName === n.tagName) idx++;
      parts.unshift(n.tagName.toLowerCase() + ':nth-of-type(' + idx + ')');
      n = n.parentElement;
    }
    return 'body > ' + parts.join(' > ');
  }
  function findEl(m) {
    var w = winByPath(m.frame || []);
    if (!w) return null;
    var sel = m.sel, at = -1, k = sel.lastIndexOf('@');
    if (k > 0 && /^\d+$/.test(sel.slice(k + 1))) { at = +sel.slice(k + 1); sel = sel.slice(0, k); }
    try { return at >= 0 ? w.document.querySelectorAll(sel)[at] : w.document.querySelector(sel); } catch (e) { return null; }
  }
  function setVal(el, v) {
    var w = el.ownerDocument.defaultView || window;
    v = v == null ? '' : String(v);
    if (el.isContentEditable) { el.focus(); el.innerText = v; }
    else if (el.tagName === 'SELECT') {
      var opts = el.options, hit = null, i, nv = v.replace(/\s/g, '');
      for (i = 0; i < opts.length; i++) if (opts[i].text.replace(/\s/g, '') === nv || opts[i].value === v) { hit = opts[i]; break; }
      if (!hit && nv) for (i = 0; i < opts.length; i++) if (opts[i].text.replace(/\s/g, '').indexOf(nv) >= 0 || nv.indexOf(opts[i].text.replace(/\s/g, '')) >= 0 && opts[i].text.trim()) { hit = opts[i]; break; }
      if (!hit) return false;
      el.value = hit.value;
    } else {
      if (el.tagName === 'INPUT' && el.maxLength > 0 && v.length > el.maxLength) v = v.slice(0, el.maxLength);
      if (el.tagName === 'INPUT') v = v.replace(/\s*\n\s*/g, ' ');
      var proto = el.tagName === 'TEXTAREA' ? w.HTMLTextAreaElement.prototype : w.HTMLInputElement.prototype;
      var d = Object.getOwnPropertyDescriptor(proto, 'value');
      if (d && d.set) d.set.call(el, v); else el.value = v;   /* React 등 프레임워크 폼 대응 */
    }
    ['input', 'change', 'blur'].forEach(function (t) { el.dispatchEvent(new w.Event(t, { bubbles: true })); });
    var o = el.style.outline; el.style.outline = '3px solid #2f5bff';
    setTimeout(function () { el.style.outline = o; }, 900);
    return true;
  }

  /* ---------- 클릭으로 칸 지정 ---------- */
  function onClick(e) {
    if (S.armed < 0) return;
    var path = e.composedPath ? e.composedPath() : [e.target];
    if (path.indexOf(host) >= 0) return;
    var el = fillable(e.target) || (e.target.closest && fillable(e.target.closest('input,textarea,select,[contenteditable]')));
    if (!el) { status('입력 칸이 아닙니다. 글자를 넣을 칸을 클릭하세요.', true); return; }
    var f = S.data.fields[S.armed];
    var ok = setVal(el, f.value);
    S.map[f.name] = { sel: selectorOf(el), frame: framePath(el.ownerDocument.defaultView) };
    saveMap();
    var name = f.name;
    if (S.seq && S.armed < S.data.fields.length - 1) { S.armed++; render(); status("'" + name + "' 입력 · 다음: '" + S.data.fields[S.armed].name + "' 칸을 클릭하세요", true); }
    else { S.armed = -1; S.seq = false; render(); status(ok ? "'" + name + "' 입력 완료 · 칸 위치를 기억했습니다" : "'" + name + "' 칸은 기억했지만 선택 목록에 맞는 값이 없습니다"); }
  }
  function onKey(e) { if (e.key === 'Escape' && S.armed >= 0) { S.armed = -1; S.seq = false; render(); status('취소했습니다.'); } }
  var hooked = [];
  function hookAll() {
    (function walk(w) {
      try {
        if (hooked.indexOf(w) < 0) { w.document.addEventListener('click', onClick, true); w.document.addEventListener('keydown', onKey, true); hooked.push(w); }
        for (var i = 0; i < w.frames.length; i++) walk(w.frames[i]);
      } catch (e) { /* 다른 출처 프레임은 접근 불가 */ }
    })(window.top);
  }

  function autoFill() {
    if (!S.data || !S.data.fields.length) return;
    hookAll();
    var done = 0, miss = [];
    S.data.fields.forEach(function (f) {
      var m = S.map[f.name];
      if (!m) { miss.push(f.name); return; }
      var el = findEl(m);
      if (el && setVal(el, f.value)) done++; else miss.push(f.name);
    });
    status(done + '개 칸을 채웠습니다.' + (miss.length ? ' 남은 항목: ' + miss.join(', ') + ' (항목을 누르고 칸을 클릭해 지정)' : ''), !!miss.length);
  }
  function copy(v) {
    v = v || '';
    function fb() { var t = document.createElement('textarea'); t.value = v; t.style.cssText = 'position:fixed;top:-99px'; document.body.appendChild(t); t.select(); try { document.execCommand('copy'); } catch (e) {} t.remove(); }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(v).catch(fb); else fb();
  }

  /* ---------- 화면 ---------- */
  var msg = '', warn = false, minimized = false;
  function status(t, w) { msg = t || ''; warn = !!w; var el = P.querySelector('.st'); if (el) { el.textContent = msg; el.className = 'st' + (warn ? ' w' : ''); } }
  function render() {
    var d = S.data, h = '<div class="h" id="drag"><b>IUX OCR 입력 도우미</b><button id="min" title="접기">–</button><button id="x" title="닫기">✕</button></div>';
    if (S.offline && !d) {
      h += '<div class="off"><div>IUX OCR에 연결할 수 없습니다. 프로그램이 켜져 있는지 확인하거나, 항목 탭의 <b>도우미용 복사</b> 내용을 아래에 붙여넣으세요.</div><textarea id="paste" placeholder="여기에 붙여넣기 (Ctrl+V)"></textarea><button class="b pri" id="apply">불러오기</button><button class="b" id="retry">다시 연결</button></div>';
    } else if (!d || !d.doc) {
      h += '<div class="st">IUX OCR에 열린 문서가 없습니다.</div>';
    } else {
      var mapped = d.fields.filter(function (f) { return S.map[f.name]; }).length;
      h += '<div class="doc"><button class="nav" id="prev" title="이전 문서">‹</button><span class="nm" title="' + esc(d.doc.name) + '">' + (d.index ? d.index + '/' + d.total + ' · ' : '') + esc(d.doc.name) + (d.checked ? ' ✓' : '') + '</span><button class="nav" id="next" title="다음 문서">›</button><button class="nav" id="re" title="새로고침">⟳</button></div>';
      h += '<div class="tb"><button class="b pri" id="auto" title="기억한 칸에 한 번에 채우기">자동 채우기 (' + mapped + '/' + d.fields.length + ')</button><button class="b' + (S.seq ? ' on' : '') + '" id="seq" title="첫 항목부터 칸을 차례로 클릭해 지정">순서대로 지정</button></div>';
      h += '<div class="l">' + (d.fields.length ? d.fields.map(function (f, i) {
        return '<div class="r' + (i === S.armed ? ' arm' : '') + '" data-i="' + i + '"><span class="k" title="' + esc(f.name) + '">' + esc(f.name) + '</span><span class="v' + (f.value ? '' : ' e') + '" title="' + esc(f.value) + '">' + (esc(f.value) || '(비어 있음)') + '</span>' + (S.map[f.name] ? '<span class="m" title="칸 위치 기억됨">●</span>' : '') + '<button class="c" data-c="' + i + '" title="복사">⧉</button></div>';
      }).join('') : '<div class="st">이 문서에 채워진 항목이 없습니다. IUX OCR 항목 탭에서 채우세요.</div>') + '</div>';
      h += '<div class="st' + (warn ? ' w' : '') + '">' + esc(msg || (mapped ? '자동 채우기를 누르거나, 항목을 누른 뒤 칸을 클릭하세요.' : '처음 한 번: 항목을 누른 뒤 시스템의 입력 칸을 클릭하면 채워지고 위치를 기억합니다.')) + '</div>';
      h += '<div class="ft"><button class="b" id="reset" title="이 화면의 칸 위치 기억을 지움">위치 초기화</button><button class="b pri" id="done">확인 완료 → 다음 문서</button></div>';
    }
    P.innerHTML = h;
    P.className = 'p' + (minimized ? ' min' : '');
    bind();
  }
  function on(id, fn) { var el = P.querySelector('#' + id); if (el) el.onclick = fn; }
  function bind() {
    on('x', function () { api_close(); });
    on('min', function () { minimized = !minimized; P.className = 'p' + (minimized ? ' min' : ''); });
    on('prev', function () { move(-1); }); on('next', function () { move(1); }); on('re', load);
    on('done', function () { move(1, true); });
    on('auto', autoFill);
    on('seq', function () { S.seq = !S.seq; S.armed = S.seq ? 0 : -1; hookAll(); render(); if (S.seq) status("'" + S.data.fields[0].name + "' 칸을 클릭하세요 (Esc 취소)", true); });
    on('reset', function () { if (confirm('이 화면에서 기억한 칸 위치를 모두 지울까요?')) { S.map = {}; saveMap(); render(); } });
    on('retry', load);
    on('apply', function () {
      try { var j = JSON.parse(P.querySelector('#paste').value); S.data = { doc: j.doc || { name: '붙여넣은 문서' }, fields: j.fields || [], index: 0, total: 0 }; render(); }
      catch (e) { alert('붙여넣은 내용이 올바르지 않습니다.'); }
    });
    P.querySelectorAll('.r').forEach(function (r) {
      r.onclick = function (e) {
        if (e.target.closest('.c')) { var f = S.data.fields[+e.target.closest('.c').dataset.c]; copy(f.value); status("'" + f.name + "' 복사됨 — 원하는 칸에 Ctrl+V"); return; }
        var i = +r.dataset.i; hookAll();
        if (S.seq) { S.armed = i; render(); status("'" + S.data.fields[i].name + "' 칸을 클릭하세요 (앞 항목은 건너뜀)", true); return; }
        S.armed = S.armed === i ? -1 : i; render();
        if (S.armed >= 0) status("'" + S.data.fields[i].name + "'을(를) 넣을 칸을 클릭하세요 (Esc 취소)", true);
      };
    });
    var dg = P.querySelector('#drag');
    if (dg) dg.onmousedown = function (e) {
      if (e.target.tagName === 'BUTTON') return;
      var r = host.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
      function mv(ev) { host.style.left = (ev.clientX - dx) + 'px'; host.style.top = (ev.clientY - dy) + 'px'; host.style.right = 'auto'; }
      function up() { document.removeEventListener('mousemove', mv); document.removeEventListener('mouseup', up); }
      document.addEventListener('mousemove', mv); document.addEventListener('mouseup', up);
    };
  }
  function api_close() {
    hooked.forEach(function (w) { try { w.document.removeEventListener('click', onClick, true); w.document.removeEventListener('keydown', onKey, true); } catch (e) {} });
    host.remove(); delete window.__iuxHelper;
  }
  window.__iuxHelper = { toggle: function () { host.style.display = host.style.display === 'none' ? '' : 'none'; if (host.style.display === '') load(); } };
  hookAll(); render(); load();
})();
