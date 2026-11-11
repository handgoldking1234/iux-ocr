/* IUX OCR 웹버전 — 규칙 기반 재가공·항목 추출 (app/refine.py, app/extract.py 를 자바스크립트로 옮김) */
(function (G) {
  'use strict';
  const R = {};

  /* ---------------- 기본 정리 ---------------- */
  R.normalize = function (text) {
    let t = (text || '').normalize('NFC').replace(/　| |\t/g, ' ');
    t = t.replace(/ {2,}/g, ' ');
    t = t.split('\n').map(l => l.trim()).join('\n');
    t = t.replace(/\n{3,}/g, '\n\n');
    return t.trim();
  };

  const SYL = /^[가-힣]$/;
  const STANDALONE = new Set('및 등 각 약 총 또 더 그 저 수 것 때 안 못 잘 꼭 곧 늘 왜 좀 별 전 후 매 본 동 당 귀 위 아래 우'.split(' '));
  const UNITS = new Set(['년', '월', '일', '매', '부', '개', '명', '원', '건', '권', '회', '차', '시', '분', '호']);
  const PSPLIT = new Set(['을', '를', '의', '과', '와', '는']);
  const PARTICLE_TAIL = /^(?:[가-힣]{1,2})?(?:을|를|이|가|은|는|의|에|에서|으로|로|과|와|도|만|하여|하고|합니다\.?|니다\.?|다\.?)[.,]?$/;
  const ATTACH1 = new Set(['의', '을', '를', '은', '는', '이', '가', '에', '과', '와', '로', '도', '만', '적', '장', '청']);

  function splitParticles(word) {
    const out = []; let cur = '';
    const chars = [...word];
    chars.forEach((ch, k) => {
      cur += ch;
      const rest = chars.length - k - 1;
      if (PSPLIT.has(ch) && [...cur].length >= 3 && rest >= 2) { out.push(cur); cur = ''; }
    });
    if (cur) out.push(cur);
    return out.join(' ');
  }

  R.fixSyllableSpacing = function (text) {
    return (text || '').split('\n').map(line => {
      const toks = line.split(' '), res = [];
      let i = 0;
      while (i < toks.length) {
        const t = toks[i];
        if (SYL.test(t) && !STANDALONE.has(t)) {
          let j = i; const run = [];
          while (j < toks.length && SYL.test(toks[j])) { run.push(toks[j]); j++; }
          if (run.length >= 2) {
            if (UNITS.has(run[0]) && res.length && /\d$/.test(res[res.length - 1])) res[res.length - 1] += run.shift();
            if (run.length < 2) { res.push(...run); i = j; continue; }
            let word = splitParticles(run.join(''));
            if (j < toks.length && PARTICLE_TAIL.test(toks[j])) { word += toks[j]; j++; }
            res.push(word); i = j; continue;
          }
        }
        if (res.length && SYL.test(t) && ATTACH1.has(t) && /[가-힣]{2,}$/.test(res[res.length - 1])) { res[res.length - 1] += t; i++; continue; }
        if (res.length && UNITS.has(t) && /\d$/.test(res[res.length - 1])) { res[res.length - 1] += t; i++; continue; }
        res.push(t); i++;
      }
      return res.join(' ');
    }).join('\n');
  };

  R.dropStrayLines = function (lines) {
    const out = [];
    lines.forEach(([t, conf]) => {
      t = (t || '').trim(); if (!t) return;
      const core = t.replace(/\s+/g, '');
      if (!/[가-힣A-Za-z0-9]/.test(core)) return;
      if (core.length <= 2 && conf != null && conf >= 0 && conf < 55) return;
      out.push(t);
    });
    return out;
  };
  R.looksSplit = function (t) {
    const toks = t.split(/\s+/).filter(Boolean);
    const singles = toks.filter(x => x.length === 1 && x >= '가' && x <= '힣').length;
    return toks.length >= 4 && singles / toks.length >= 0.5;
  };

  const SENT_END = /(?:[.!?。:;)\]」』"'”’…]|다\.?|요\.?|함\.?|음\.?|것\.?|임\.?|끝\.?)$/;
  const LIST_START = /^\s*(?:\d{1,2}\.|[가-하]\.|\d{1,2}\)|[가-하]\)|\(\d{1,2}\)|\([가-하]\)|[①-⑳]|[㉮-㉻]|[□■○●◦◎▶▷►•·\-–※*]|붙\s*임|수\s*신|제\s*목|시\s*행|접\s*수)/;
  const PAGE_NO = /^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$|^\s*\d{1,4}\s*\/\s*\d{1,4}\s*$/;

  R.cleanText = function (text, merge = true, dropPageNo = true, fixSpacing = true) {
    let lines = R.normalize(text).split('\n');
    if (dropPageNo) lines = lines.filter(l => !PAGE_NO.test(l));
    if (fixSpacing) lines = lines.map(l => l
      .replace(/\s+([,.)\]}:;!?])/g, '$1')
      .replace(/([(\[{])\s+/g, '$1')
      .replace(/^([가-하]|\d{1,2})\s+([.)])/, '$1$2')
      .replace(/(\d)\s*\.\s*(\d)(?!\d*\s*\.)/g, '$1.$2')
      .replace(/(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\./g, '$1. $2. $3.'));
    if (!merge) return lines.join('\n').trim();
    const out = [];
    for (const l of lines) {
      if (!l) { if (out.length && out[out.length - 1] !== '') out.push(''); continue; }
      const prev = out[out.length - 1];
      if (out.length && prev && !SENT_END.test(prev) && !LIST_START.test(l)) {
        out[out.length - 1] = (prev.endsWith('-') && /^[a-zA-Z]/.test(l)) ? prev.slice(0, -1) + l : prev + ' ' + l;
      } else out.push(l);
    }
    return out.join('\n').trim();
  };

  /* ---------------- 공문서 항목 ---------------- */
  const DATE = '\\d{4}\\s*[.\\-/년]\\s*\\d{1,2}\\s*[.\\-/월]\\s*\\d{1,2}\\s*[.일]?';
  const DOCNO = '[가-힣A-Za-z0-9·]{1,20}\\s?-\\s?\\d{1,7}';
  R.DATE = DATE; R.DOCNO = DOCNO;
  const FIELD_RULES = [
    ['receiver', '수신', /수\s*신(?:\s*자)?\s*[:：]?\s*(.+)/],
    ['via', '경유', /^\(?\s*경\s*유\s*\)?\s*[:：]?\s*([^\s)].*)$/],
    ['title', '제목', /제\s*목\s*[:：]?\s*(.+)/],
    ['attach', '붙임', /붙\s*임\s*[:：]?\s*(.+)/],
    ['drafter', '기안자', /기\s*안\s*자\s*[:：]?\s*([^\s].{0,20}?)(?=\s{2,}|\s*검\s*토|\s*결\s*재|$)/],
    ['reviewer', '검토자', /검\s*토\s*자\s*[:：]?\s*([^\s].{0,20}?)(?=\s{2,}|\s*결\s*재|\s*협\s*조|$)/],
    ['approver', '결재권자', /결\s*재\s*권\s*자\s*[:：]?\s*([^\s].{0,20}?)(?=\s{2,}|\s*협\s*조|$)/],
    ['tel', '전화', /전\s*화\s*(?:번\s*호)?\s*[:：]?\s*([\d()\-\s]{7,20}\d)/],
    ['fax', '전송', /(?:전\s*송|팩\s*스|FAX)\s*[:：]?\s*([\d()\-\s]{7,20}\d)/],
    ['email', '이메일', /([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})/],
    ['zip', '우편번호', /(?:우\s*)?(\d{5})\s+[가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)/],
    ['address', '주소', /(?:주\s*소\s*[:：]?\s*|\d{5}\s+)([가-힣]+(?:특별시|광역시|특별자치시|특별자치도|도)\s+[^\n/]{4,60})/],
    ['disclosure', '공개구분', /((?:대시민\s*)?(?:비공개|부분\s*공개|공개)(?:\s*\([^)]{0,20}\))?)\s*$/],
  ];
  const SENDER_RX = /^[가-힣A-Za-z0-9·\s]{2,30}(?:장|청장|총장|원장|대표|위원장|이사장|교육감|군수|구청장|시장|도지사)(?:\s*\(?(?:직인|인)\)?)?$/;

  R.extractFields = function (text) {
    const t = R.normalize(text), lines = t.split('\n'), found = {};
    const put = (key, label, value) => {
      value = (value || '').replace(/\s{2,}/g, ' ').replace(/^[\s:：]+|[\s:：]+$/g, '');
      value = value.replace(/\s+([.,)\]])/g, '$1').replace(/([(\[])\s+/g, '$1');
      if (value && !found[key]) found[key] = { key, label, value };
    };
    for (const [kind, label] of [['시\\s*행', '시행'], ['접\\s*수', '접수']]) {
      const m = t.match(new RegExp(kind + '\\s*[:：]?\\s*(' + DOCNO + ')?\\s*\\(?\\s*(' + DATE + ')?'));
      if (m && (m[1] || m[2])) {
        const key = label === '시행' ? 'exec' : 'recv';
        if (m[1]) put(key + '_no', label + ' 문서번호', m[1].replace(/\s/g, ''));
        if (m[2]) put(key + '_date', label + '일자', m[2]);
      }
    }
    for (const [key, label, rx] of FIELD_RULES) {
      for (const l of lines) { const m = l.match(rx); if (m) { put(key, label, m[1]); break; } }
    }
    if (found.attach) {
      const idx = lines.findIndex(l => /붙\s*임/.test(l));
      if (idx >= 0) {
        const more = [];
        for (const l of lines.slice(idx + 1, idx + 8)) {
          if (!l.trim()) continue;
          if (/^\s*\d{1,2}\.\s*\S/.test(l) && !/끝\.?\s*$/.test(found.attach.value)) more.push(l.trim()); else break;
        }
        if (more.length) found.attach.value += ' / ' + more.join(' / ');
      }
      found.attach.value = found.attach.value.replace(/\s+\./g, '.').replace(/\s*끝\s*\.?\s*$/, '');
    }
    for (const l of [...lines].reverse()) {
      const s = l.trim();
      if (SENDER_RX.test(s)) { put('sender', '발신명의', s.replace(/\s*\(?(직인|인)\)?$/, '')); break; }
    }
    const dates = [...new Set((t.match(new RegExp(DATE, 'g')) || []).map(d => d.replace(/\s+/g, ' ').trim()))];
    if (dates.length) put('dates', '등장 날짜', dates.slice(0, 8).join(', '));
    const order = ['exec_no', 'exec_date', 'recv_no', 'recv_date', 'receiver', 'via', 'title', 'attach', 'sender',
      'drafter', 'reviewer', 'approver', 'tel', 'fax', 'email', 'zip', 'address', 'disclosure', 'dates'];
    return order.filter(k => found[k]).map(k => found[k]);
  };

  /* ---------------- 개인정보 마스킹 ---------------- */
  const PII = [
    ['주민/외국인등록번호', /(?<!\d)(\d{6})\s*[-–]\s*([1-8])\d{6}(?!\d)/g, (m, a, b) => `${a}-${b}******`],
    ['카드번호', /(?<!\d)(\d{4})[\s\-](\d{4})[\s\-](\d{4})[\s\-](\d{4})(?!\d)/g, (m, a, b, c, d) => `${a}-****-****-${d}`],
    ['전화번호', /(?<!\d)(01[016789]|0[2-6][1-5]?|070)\s*[-.)]?\s*(\d{3,4})\s*[-.]\s*(\d{4})(?!\d)/g, (m, a, b, c) => `${a}-${'*'.repeat(b.length)}-${c}`],
    ['이메일', /([A-Za-z0-9._%+\-]{1,3})([A-Za-z0-9._%+\-]*)(@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})/g, (m, a, b, c) => a + '*'.repeat(b.length) + c],
    ['계좌번호', /(계좌(?:번호)?\s*[:：]?\s*(?:[가-힣]+은행\s*)?)(\d[\d\-]{7,20}\d)/g, (m, a, b) => a + b.slice(0, -4).replace(/\d/g, '*') + b.slice(-4)],
    ['여권번호', /(?<![A-Za-z0-9])([MSRODG])(\d{3})([A-Z]?\d{4,5})(?![A-Za-z0-9])/g, (m, a, b, c) => a + b + '*'.repeat(c.length)],
  ];
  R.maskPii = function (text) {
    const counts = {};
    for (const [label, rx, fn] of PII) {
      let n = 0;
      text = text.replace(rx, (...args) => { n++; return fn(...args); });
      if (n) counts[label] = n;
    }
    return { text, counts };
  };

  /* ---------------- 개요 ---------------- */
  const LEVELS = [
    [/^(\d{1,2}\.)\s*(.*)/, 0], [/^([가-하]\.)\s*(.*)/, 1], [/^(\d{1,2}\))\s*(.*)/, 2], [/^([가-하]\))\s*(.*)/, 3],
    [/^(\(\d{1,2}\))\s*(.*)/, 4], [/^(\([가-하]\))\s*(.*)/, 5], [/^([①-⑳])\s*(.*)/, 6], [/^([㉮-㉻])\s*(.*)/, 7],
    [/^([□■])\s*(.*)/, 0], [/^([○●◦])\s*(.*)/, 1], [/^([-–])\s+(.*)/, 2], [/^([·•])\s*(.*)/, 3], [/^(※)\s*(.*)/, -1],
  ];
  R.toOutline = function (text) {
    const out = [];
    for (const raw of R.cleanText(text).split('\n')) {
      const l = raw.trim();
      if (!l) { out.push(''); continue; }
      const t = l.match(/^제\s*목\s*[:：]?\s*(.+)/);
      if (t) { out.push('# ' + t[1]); continue; }
      let hit = false;
      for (const [rx, lvl] of LEVELS) {
        const m = l.match(rx);
        if (m) { out.push(lvl < 0 ? `> ※ ${m[2]}` : '  '.repeat(lvl) + `- **${m[1]}** ${m[2]}`); hit = true; break; }
      }
      if (!hit) out.push(l);
    }
    return out.join('\n').replace(/\n{3,}/g, '\n\n').trim();
  };

  /* ---------------- 본문 ---------------- */
  const HEAD = /^\s*(?:제\s*목|수\s*신(?:\s*자)?|\(?\s*경\s*유\s*\)?|참\s*조)(?=\s|$|[:：(])/;
  const TITLE = /^\s*제\s*목\s*[:：]?\s*\S/;
  const END = /^\s*(?:붙\s*임|기\s*안\s*자|검\s*토\s*자|결\s*재\s*권\s*자|협\s*조\s*자|시\s*행\s+\S|접\s*수(?=\s|$)|우\s*\d{5}|전\s*화\s*[\d(]|수\s*신\s*자\s*[:：]|[가-힣·\s]{2,30}(?:장|청장|총장|원장|대표|위원장|이사장|교육감|군수|구청장|시장|도지사)\s*(?:\(?(?:직인|인)\)?)?\s*$)/;
  R.extractBody = function (text) {
    const lines = R.normalize(text).split('\n');
    if (!lines.length || !lines[0] && lines.length === 1) return '';
    let start = 0;
    const ti = lines.findIndex(l => TITLE.test(l));
    if (ti >= 0) start = ti + 1;
    else { const heads = lines.slice(0, 15).map((l, i) => HEAD.test(l) ? i : -1).filter(i => i >= 0); if (heads.length) start = heads[heads.length - 1] + 1; }
    let end = lines.length;
    for (let i = start; i < lines.length; i++) {
      const l = lines[i];
      if (!l.trim()) continue;
      if (END.test(l) && i > start) { end = i; break; }
      if (/끝\s*\.?\s*$/.test(l)) { lines[i] = l.replace(/\s*끝\s*\.?\s*$/, ''); end = i + 1; break; }
    }
    const body = lines.slice(start, end).join('\n').trim();
    return body ? R.cleanText(body) : '';
  };

  /* ---------------- 내 항목 템플릿 ---------------- */
  R.PRESETS = {
    date: ['날짜', DATE + '|\\d{4}-\\d{2}-\\d{2}'],
    period: ['기간(~)', '(?:' + DATE + ')\\s*[~∼～\\-]\\s*(?:' + DATE + ')'],
    money: ['금액', '(?:금\\s*)?\\d[\\d,]*(?:\\.\\d+)?\\s*(?:조|억|천만|백만|만|천)?\\s*원'],
    number: ['수량·숫자+단위', '\\d[\\d,]*(?:\\.\\d+)?\\s*(?:매|건|부|권|명|개소|개|대|회|차|%|㎡|평|km|kg)'],
    phone: ['전화번호', '(?<!\\d)(?:01[016789]|0[2-6][1-5]?|070)\\s*[-.)]?\\s*\\d{3,4}\\s*[-.]\\s*\\d{4}(?!\\d)'],
    email: ['이메일', '[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}'],
    docno: ['문서번호', DOCNO],
    bizno: ['사업자등록번호', '(?<!\\d)\\d{3}-\\d{2}-\\d{5}(?!\\d)'],
    url: ['홈페이지 주소', '(?:https?://|www\\.)[^\\s/]+(?:/[^\\s]*)?'],
  };
  R.GONGMUN_KEYS = {
    title: '제목', exec_no: '시행 문서번호', exec_date: '시행일자', recv_no: '접수 문서번호', recv_date: '접수일자',
    receiver: '수신', via: '경유', attach: '붙임', sender: '발신명의', drafter: '기안자', reviewer: '검토자',
    approver: '결재권자', tel: '전화', fax: '전송', email: '이메일', zip: '우편번호', address: '주소', disclosure: '공개구분',
  };
  R.FORMATS = { '': '그대로', 'ymd-dash': '2026-11-06', 'ymd': '20261106', 'ymd-dot': '2026.11.06', 'digits': '숫자만', 'oneline': '한 줄로' };
  const STOP_LABELS = ['기안자', '검토자', '결재권자', '협조자', '시행', '접수', '전화', '전송', '공개', '담당자', '담당', '연락처', '주소'];
  const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const labelRx = l => [...l.replace(/ /g, '')].map(reEsc).join('\\s*');

  function byLabel(lines, labels, stops, multi) {
    const alt = labels.filter(l => l.trim()).map(labelRx).join('|');
    if (!alt) return [];
    const rx = new RegExp('(?:^|[\\s(\\[○□\\-·|])(?:' + alt + ')\\s*(?:[:：|]|은|는)?\\s*([^|]*)'); // 표 행(칸 | 칸)은 칸 경계까지만
    const stopRx = stops.length ? new RegExp('\\s(?:' + stops.map(labelRx).join('|') + ')\\s*[:：]?\\s') : null;
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(rx);
      if (!m) continue;
      let v = m[1].trim();
      if (!v) v = (lines.slice(i + 1, i + 3).find(x => x.trim()) || '').trim();
      if (stopRx) { const sm = (' ' + v + ' ').match(stopRx); if (sm) v = (' ' + v).slice(0, sm.index).trim(); }
      v = v.replace(/^[\s:：,]+|[\s:：,]+$/g, '');
      if (v) { out.push(v); if (!multi) break; }
    }
    return out;
  }

  function fmtDate(v, kind) {
    return v.replace(/(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*[.일]?/g, (m, y, mo, d) => {
      mo = String(+mo).padStart(2, '0'); d = String(+d).padStart(2, '0');
      return { 'ymd-dash': `${y}-${mo}-${d}`, 'ymd': `${y}${mo}${d}`, 'ymd-dot': `${y}.${mo}.${d}` }[kind];
    });
  }
  R.applyFormat = function (v, fmt) {
    if (!v || !fmt) return v;
    if (fmt === 'ymd-dash' || fmt === 'ymd' || fmt === 'ymd-dot') return fmtDate(v, fmt).trim();
    if (fmt === 'digits') return v.split(' / ').map(x => x.replace(/[^\d.]/g, '')).join(' / ');
    if (fmt === 'oneline') return v.replace(/\s*\n\s*/g, ' ').trim();
    return v;
  };

  R.runTemplate = function (text, fields) {
    const lines = R.normalize(text).split('\n'), joined = lines.join('\n');
    const labelsAll = fields.filter(f => f.type === 'label').flatMap(f => (f.label || '').split(/[|,/]/)).map(s => s.trim()).filter(Boolean);
    let gm = null;
    return fields.map(f => {
      const name = (f.name || '').trim() || '항목', type = f.type || 'label', multi = !!f.multi;
      let value = '', note = '';
      try {
        if (type === 'label') {
          const labs = (f.label || name).split(/[|,/]/).map(s => s.trim()).filter(Boolean);
          const stops = labelsAll.concat(STOP_LABELS).filter(s => !labs.includes(s));
          value = byLabel(lines, labs, stops, multi).join(' / ');
        } else if (type === 'preset') {
          const p = R.PRESETS[f.preset];
          if (p) {
            const found = [...new Set([...joined.matchAll(new RegExp(p[1], 'g'))].map(m => m[0].replace(/\s+/g, ' ').trim()))];
            value = (multi ? found : found.slice(0, 1)).join(' / ');
          }
        } else if (type === 'regex') {
          const rx = new RegExp(f.pattern || '', 'gm'), found = [];
          for (const m of joined.matchAll(rx)) { found.push((m.length > 1 && m[1] != null ? m[1] : m[0]).trim()); if (!multi) break; if (m[0] === '') break; }
          value = [...new Set(found)].join(' / ');
        } else if (type === 'gongmun') {
          if (!gm) gm = Object.fromEntries(R.extractFields(text).map(x => [x.key, x.value]));
          value = gm[f.key] || '';
        } else if (type === 'body') {
          value = R.extractBody(text);
        } else if (type === 'ai') {
          note = 'AI 항목은 PC 버전(Ollama)에서만 채워집니다';
        }
      } catch (e) { note = '정규식 오류: ' + e.message; }
      return { name, value: R.applyFormat(value, f.fmt), type, note };
    });
  };

  R.DEFAULT_TEMPLATES = [
    { id: 'records-meta', name: '기록물 메타데이터 (예시)', fields: [
      { name: '제목', type: 'gongmun', key: 'title' }, { name: '생산기관', type: 'gongmun', key: 'sender' },
      { name: '문서번호', type: 'gongmun', key: 'exec_no' }, { name: '생산일자', type: 'gongmun', key: 'exec_date', fmt: 'ymd-dash' },
      { name: '수신', type: 'gongmun', key: 'receiver' }, { name: '사업기간', type: 'label', label: '사업기간|사업 기간|기간' },
      { name: '금액', type: 'preset', preset: 'money', multi: true }, { name: '연락처', type: 'preset', preset: 'phone', multi: true },
      { name: '붙임', type: 'gongmun', key: 'attach' }, { name: '본문', type: 'body' }] },
    { id: 'contract', name: '계약서 (예시)', fields: [
      { name: '계약명', type: 'label', label: '계약명|용역명|사업명|건명' }, { name: '계약금액', type: 'label', label: '계약금액|총 계약금액|계약 금액' },
      { name: '계약기간', type: 'preset', preset: 'period' }, { name: '발주처', type: 'label', label: '발주처|발주기관|발주자' },
      { name: '계약상대자', type: 'label', label: '계약상대자|수급인|수급자' }, { name: '사업자등록번호', type: 'preset', preset: 'bizno', multi: true }] },
  ];

  /* ---------------- 자동 항목 (문서 속 '이름표 값'을 찾아 항목으로) ---------------- */
  const STRONG = ['수신자', '수신', '경유', '참조', '받음', '발신명의', '발신', '제목', '문서번호', '시행일자', '시행', '접수일자', '접수',
    '보존기간', '공개여부', '공개구분', '기안자', '검토자', '결재권자', '협조자', '담당자', '붙임', '우편번호', '전화번호', '전화', '전송', '팩스',
    '홈페이지', '전자우편', '생산기관', '생산일자', '심사자', '처리과', '등록번호', '분류번호', '건명'];
  const STRONG_ALT = [...STRONG].sort((a, b) => b.length - a.length).map(labelRx).join('|');
  const RX_STRONG = new RegExp('(^|[\\s/|·])[\\[(（]?\\s*(' + STRONG_ALT + ')\\s*[\\])）]?(?=\\s*[:：]|\\s|$)', 'g');
  const LIST_MARK = /^\s*(?:\d{1,2}\.|[가-하]\.|\d{1,2}\)|[가-하]\)|\(\d{1,2}\)|\([가-하]\)|[-·•○□■◦※])\s*/;
  const isSyl = t => t.length === 1 && t >= '가' && t <= '힣';
  function splitLabel(seg) { // 콜론 앞 덩어리에서 이름표(마지막 낱말, '명 칭'처럼 띄운 글자는 합침)와 그 앞 내용을 나눔
    const toks = seg.trim().split(/\s+/).filter(Boolean);
    if (!toks.length) return { label: '', rest: '' };
    let j = toks.length - 1;
    if (isSyl(toks[j])) while (j > 0 && isSyl(toks[j - 1])) j--;
    return { label: toks.slice(j).join(''), rest: toks.slice(0, j).join(' ') };
  }
  function colonPairs(line) {
    const L = line.replace(LIST_MARK, '').replace(/^[\[(（]\s*|\s*[\])）](?=\s*[:：])/g, '');
    const parts = L.split(/(?<!\d)\s*[:：]\s*/);
    if (parts.length < 2) return [];
    let cur;
    if (parts.length === 2) { // 이름표 하나: 콜론 앞 전체가 짧을 때만
      const lab = parts[0].replace(/\s+/g, '');
      if (!/^(?:[가-힣]{2,8}|[A-Za-z]{2,15})$/.test(lab)) return [];
      return [[lab, parts[1]]];
    }
    const pairs = [];
    cur = splitLabel(parts[0]);
    for (let k = 1; k < parts.length; k++) {
      if (k < parts.length - 1) { const nx = splitLabel(parts[k]); pairs.push([cur.label, nx.rest]); cur = nx; }
      else pairs.push([cur.label, parts[k]]);
    }
    return pairs.filter(([l]) => /^(?:[가-힣]{2,10}|[A-Za-z]{2,15})$/.test(l));
  }
  function cleanVal(v) {
    v = v.replace(/^[\s:：]+|[\s/|:：]+$/g, '').replace(/\s+([.,)\]])/g, '$1').replace(/([(\[])\s+/g, '$1');
    if (R.looksSplit(v)) v = R.fixSyllableSpacing(v);
    return v.trim();
  }
  R.autoFields = function (text) {
    const lines = R.normalize(text).split('\n'), out = [], seen = {};
    const add = (name, value) => {
      name = name.replace(/\s+/g, ''); value = cleanVal(value);
      if (!name || !value) return;
      if (name === '수신자') name = '수신';
      seen[name] = (seen[name] || 0) + 1;
      out.push({ name: seen[name] > 1 ? `${name} ${seen[name]}` : name, value });
    };
    const isLabel = c => /^[가-힣A-Za-z·/]{1,12}$/.test(c.replace(/[\s()（）]/g, '')) && !/\d/.test(c);
    let head = null;
    for (const line of lines) {
      if (line.includes(' | ')) { // 표 행: 이름 칸 | 값 칸 짝, 또는 머리 행 아래 값 행
        const cells = line.split(' | ').map(c => c.trim()), labs = cells.map(isLabel);
        if (labs.every(Boolean) && cells.length >= 2) { head = cells; continue; }
        if (head && cells.length === head.length) { head.forEach((h, k) => add(h, cells[k])); continue; }
        head = null;
        for (let k = 0; k < cells.length - 1;) { if (labs[k] && !labs[k + 1]) { add(cells[k], cells[k + 1]); k += 2; } else k++; }
        continue;
      }
      head = null;
      const ms = [...line.matchAll(RX_STRONG)];
      if (ms.length && ms[0].index === 0) {
        const segs = ms.map(m => ({ label: m[2], start: m.index + m[1].length, end: m.index + m[0].length }));
        segs.forEach((g, k) => {
          let v = line.slice(g.end, k + 1 < segs.length ? segs[k + 1].start : line.length);
          v = v.replace(/^\s*[\])）]?\s*[:：]?/, '');
          if (g.label.replace(/\s/g, '') === '붙임') v = v.replace(/\s*끝\s*\.?\s*$/, '');
          add(g.label, v);
        });
        continue;
      }
      for (const [l, v] of colonPairs(line)) add(l, v);
    }
    const names = new Set(out.map(o => o.name.replace(/ \d+$/, '')));
    const gm = Object.fromEntries(R.extractFields(text).map(f => [f.key, f.value]));
    [['문서번호', 'exec_no'], ['시행일자', 'exec_date'], ['발신명의', 'sender']].forEach(([n, k]) => { if (gm[k] && !names.has(n)) out.push({ name: n, value: gm[k] }); });
    const body = R.extractBody(text);
    if (body) out.push({ name: '본문', value: body });
    return out;
  };

  R.linesToText = function (lines) {
    const out = []; let prev = null;
    for (const l of lines) { if (prev !== null && l.block !== prev) out.push(''); out.push(l.text); prev = l.block; }
    return out.join('\n');
  };

  G.IUXRules = R;
})(typeof window !== 'undefined' ? window : globalThis);
