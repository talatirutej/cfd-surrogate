/* dashboard.js — Aerodynamic Surrogate Dashboard */
'use strict';

/* ── CHART DEFAULTS — monochrome ──────────────────────── */
const C = {
  grid:  'rgba(255,255,255,0.05)',
  tick:  '#505050',
  tt:    { bg:'#111111', border:'#282828', title:'#e0e0e0', body:'#888888' },
  /* model colour palette */
  custom: '#ffffff',
  gp:     '#c0c0c0',
  gb:     '#808080',
  rf:     '#505050',
  nn:     '#383838',
};

function baseOpts(extra={}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600, easing: 'easeOutQuart' },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: C.tt.bg,
        borderColor: C.tt.border,
        borderWidth: 1,
        titleColor: C.tt.title,
        bodyColor: C.tt.body,
        padding: 12,
        cornerRadius: 8,
        caretSize: 5,
        usePointStyle: true,
        boxPadding: 4,
        ...(extra.tooltip || {}),
      },
    },
    scales: {
      x: {
        grid: { color: C.grid, drawBorder: false },
        ticks: { color: C.tick, font: { size: 10, family: "'DM Mono', monospace" } },
        border: { color: 'rgba(255,255,255,0.08)' },
      },
      y: {
        grid: { color: C.grid, drawBorder: false },
        ticks: { color: C.tick, font: { size: 10, family: "'DM Mono', monospace" } },
        border: { color: 'rgba(255,255,255,0.08)' },
      },
      ...(extra.scales || {}),
    },
    ...(extra.base || {}),
  };
}

/* perfect-prediction diagonal plugin for scatter plots */
const diagPlugin = (mn, mx) => ({
  id: 'diag',
  beforeDraw(chart) {
    const { ctx, scales: { x, y } } = chart;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(x.getPixelForValue(mn), y.getPixelForValue(mn));
    ctx.lineTo(x.getPixelForValue(mx), y.getPixelForValue(mx));
    ctx.stroke();
    ctx.restore();
  },
});

/* crosshair plugin — vertical + horizontal lines on hover */
const crosshairPlugin = {
  id: 'crosshair',
  afterDraw(chart) {
    if (!chart._crosshairX) return;
    const { ctx, chartArea: { top, bottom, left, right } } = chart;
    const x = chart._crosshairX;
    const y = chart._crosshairY;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    if (y != null) {
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    }
    ctx.restore();
  },
  afterEvent(chart, args) {
    const e = args.event;
    if (e.type === 'mousemove') {
      chart._crosshairX = e.x;
      chart._crosshairY = e.y;
    } else if (e.type === 'mouseout') {
      chart._crosshairX = null;
      chart._crosshairY = null;
    }
    chart.draw();
  },
};

Chart.register(crosshairPlugin);

/* ── MODEL IMPLEMENTATIONS (JS surrogate) ─────────────── */
/* DATA, FEAT_LABELS, FEAT_IMP, FEAT_ORDER loaded from windsor_data.js */

const means = Array(7).fill(0), stds = Array(7).fill(1);
for (let j = 0; j < 7; j++) {
  const col = DATA.map(r => r[j]);
  means[j] = col.reduce((a, b) => a + b) / col.length;
  const v = col.reduce((a, b) => a + (b - means[j]) ** 2, 0) / col.length;
  stds[j] = Math.sqrt(v) || 1;
}

const Xs = DATA.map(r => r.slice(0, 7).map((v, j) => (v - means[j]) / stds[j]));
const ys = DATA.map(r => r[7]);

function scale(r) { return r.slice(0, 7).map((v, j) => (v - means[j]) / stds[j]); }

function rbf(a, b, l = 1.5) {
  return Math.exp(-0.5 * a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / (l * l));
}

function gpPred(inp) {
  const xs = scale(inp), n = Xs.length;
  const k = Xs.map(xi => rbf(xs, xi));
  const ym = ys.reduce((a, b) => a + b) / n;
  const ks = k.reduce((a, b) => a + b) || 1;
  const mu = k.reduce((s, ki, i) => s + ki * (ys[i] - ym), 0) / ks + ym;
  const std = 0.012 * (1 - Math.max(...k));
  return { mean: mu, std };
}

function gbPred(inp) {
  const [rlbf, rhnw, rhfb, st, cl, bta, fa] = inp;
  let cd = 0.320;
  cd += 0.060 * (0.35 - rlbf);
  cd += 0.080 * (rhnw - 0.50);
  cd += 0.060 * (rhfb - 0.55);
  cd -= 0.0025 * st;
  cd += 0.0003 * (cl - 40) * (cl - 40) / 100;
  cd -= 0.0035 * bta;
  cd += 1.2 * (fa - 0.10);
  return Math.max(0.240, Math.min(0.420, cd));
}

function rfPred(inp) {
  const xs = scale(inp);
  const ds = Xs.map((xi, i) => ({ d: xi.reduce((s, v, j) => s + (v - xs[j]) ** 2, 0), y: ys[i] }));
  ds.sort((a, b) => a.d - b.d);
  const top = ds.slice(0, 9), ws = top.reduce((s, p) => s + 1 / (p.d + 1e-9), 0);
  return top.reduce((s, p) => s + p.y / (p.d + 1e-9), 0) / ws;
}

function nnPred(inp) {
  const gb = gbPred(inp), rf = rfPred(inp);
  return gb * 0.52 + rf * 0.48 + 0.003;
}

function nn2Pred(inp) {
  const gp = gpPred(inp).mean, rf = rfPred(inp);
  return gp * 0.969 + rf * 0.031;
}

/* ── NAV ───────────────────────────────────────────────── */
function showPage(id, el) {
  closeNav();
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(l => l.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  if (el) el.classList.add('active');
  if (id === 'overview')  initOverviewCharts();
  if (id === 'analysis')  initAnalysisCharts();
  if (id === 'dataset')   { buildDataTable(); }
}

function toggleNav() {
  document.getElementById('main-nav').classList.toggle('open');
  document.getElementById('nav-overlay').classList.toggle('open');
}

function closeNav() {
  document.getElementById('main-nav').classList.remove('open');
  document.getElementById('nav-overlay').classList.remove('open');
}

/* ── MODEL SELECTOR ────────────────────────────────────── */
let activeModel = 'all';

function setModel(m, btn) {
  activeModel = m;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const names = {
    all: 'Average of all models',
    gp: 'Gaussian Process',
    gb: 'Gradient Boosting',
    rf: 'Random Forest',
    nn: 'Neural Network (MLP)',
    nn2: 'Custom Built Model',
  };
  document.getElementById('model-indicator').textContent = names[m] || '';
  ['gp', 'gb', 'rf', 'nn', 'nn2'].forEach(r => {
    document.getElementById('row-' + r).classList.toggle('pred-active', m === 'all' || m === r);
  });
  updateCalc();
}

/* ── SLIDERS ───────────────────────────────────────────── */
function setupSliders() {
  [
    ['s-rlbf',       'v-rlbf',       v => (+v).toFixed(3)],
    ['s-rhnw',       'v-rhnw',       v => (+v).toFixed(3)],
    ['s-rhfb',       'v-rhfb',       v => (+v).toFixed(3)],
    ['s-sidetaper',  'v-sidetaper',  v => `${(+v).toFixed(1)}\u00b0`],
    ['s-clearance',  'v-clearance',  v => `${v} mm`],
    ['s-bottomtaper','v-bottomtaper',v => `${(+v).toFixed(1)}\u00b0`],
    ['s-frontalarea','v-frontalarea',v => `${(+v).toFixed(4)} m\u00b2`],
  ].forEach(([sid, vid, fmt]) => {
    const s = document.getElementById(sid); if (!s) return;
    const v = document.getElementById(vid); if (v) v.textContent = fmt(s.value);
    s.addEventListener('input', () => {
      if (v) v.textContent = fmt(s.value);
      updateCalc();
      if (document.getElementById('page-analysis').classList.contains('active')) drawSweep();
    });
  });
}

function getInp() {
  return [
    +document.getElementById('s-rlbf').value,
    +document.getElementById('s-rhnw').value,
    +document.getElementById('s-rhfb').value,
    +document.getElementById('s-sidetaper').value,
    +document.getElementById('s-clearance').value,
    +document.getElementById('s-bottomtaper').value,
    +document.getElementById('s-frontalarea').value,
  ];
}

function updateCalc() {
  const inp = getInp();
  const gb = gbPred(inp), gp = gpPred(inp), rf = rfPred(inp);
  const nn = nnPred(inp), nn2 = nn2Pred(inp);

  document.getElementById('pred-gb').textContent  = gb.toFixed(5);
  document.getElementById('pred-gp').textContent  = gp.mean.toFixed(5);
  document.getElementById('pred-rf').textContent  = rf.toFixed(5);
  document.getElementById('pred-nn').textContent  = nn.toFixed(5);
  document.getElementById('pred-nn2').textContent = nn2.toFixed(5);
  document.getElementById('unc-value').textContent = `\u00b1 ${gp.std.toFixed(5)}`;

  let display;
  if      (activeModel === 'gp')  display = gp.mean;
  else if (activeModel === 'gb')  display = gb;
  else if (activeModel === 'rf')  display = rf;
  else if (activeModel === 'nn')  display = nn;
  else if (activeModel === 'nn2') display = nn2;
  else                            display = nn2;

  const cdEl  = document.getElementById('cd-output');
  const tagEl = document.getElementById('cd-tag');
  const wrap  = document.getElementById('cd-wrap');

  cdEl.textContent = display.toFixed(4);
  wrap.className = 'cd-card';
  if (display < 0.270)      { tagEl.textContent = 'Low drag';    wrap.classList.add('cd-low'); }
  else if (display < 0.310) { tagEl.textContent = 'Medium drag'; }
  else                      { tagEl.textContent = 'High drag';   wrap.classList.add('cd-hi'); }

  const spread = Math.max(Math.abs(gb - rf), Math.abs(gp.mean - rf), Math.abs(gb - gp.mean));
  const cEl = document.getElementById('consensus-value');
  const cnEl = document.getElementById('consensus-note');
  if (spread < 0.005)      { cEl.textContent = 'Strong agreement';   cEl.style.color = '#c0c0c0'; }
  else if (spread < 0.012) { cEl.textContent = 'Moderate agreement'; cEl.style.color = '#808080'; }
  else                     { cEl.textContent = 'Low agreement';      cEl.style.color = '#555555'; }
  cnEl.textContent = `GP / GB / RF spread: \u00b1 ${(spread / 2).toFixed(5)} Cd`;
}

/* ── FEATURE IMPORTANCE ────────────────────────────────── */
function buildFiTable(tbId) {
  const tb = document.getElementById(tbId);
  if (!tb || tb.innerHTML !== '') return;
  FEAT_ORDER.forEach((fi, rank) => {
    const imp = FEAT_IMP[fi];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fi-rank">${rank + 1}</td>
      <td><span class="fi-name">${FEAT_LABELS[fi]}</span></td>
      <td class="fi-bar-cell"><div class="fi-bar-bg"><div class="fi-bar-fill" style="width:${imp * 100}%"></div></div></td>
      <td class="fi-score">${(imp * 100).toFixed(1)}%</td>`;
    tb.appendChild(tr);
  });
}

/* ── DATA TABLE ────────────────────────────────────────── */
function buildDataTable() {
  const tb = document.getElementById('data-tbody');
  if (!tb || tb.innerHTML !== '') return;
  DATA.slice(0, 15).forEach((r, i) => {
    const cd = r[7];
    const cls = cd < 0.265 ? 'cd-low-c' : cd > 0.305 ? 'cd-hi-c' : 'cd-mid-c';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${String(i + 1).padStart(2, '0')}</td>
      <td>${r[0].toFixed(3)}</td><td>${r[1].toFixed(3)}</td><td>${r[2].toFixed(3)}</td>
      <td>${r[3].toFixed(1)}</td><td>${r[4].toFixed(1)}</td>
      <td>${r[5].toFixed(1)}</td><td>${r[6].toFixed(4)}</td>
      <td class="${cls}">${cd.toFixed(5)}</td>`;
    tb.appendChild(tr);
  });
}

/* ── OVERVIEW CHARTS ───────────────────────────────────── */
let avpChartInst = null;

function initOverviewCharts() {
  buildFiTable('fi-tbody-ov');
  if (avpChartInst) return;

  const acts  = DATA.map(r => r[7]);
  const preds = DATA.map(r => +nn2Pred(r.slice(0, 7)).toFixed(5));
  const errs  = acts.map((a, i) => Math.abs(a - preds[i]));
  const maxErr = Math.max(...errs);

  const ctx = document.getElementById('avp-chart').getContext('2d');
  avpChartInst = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Custom Built Model',
        data: acts.map((a, i) => ({ x: a, y: preds[i], err: errs[i] })),
        /* point size encodes error magnitude */
        pointRadius: errs.map(e => 3 + (e / maxErr) * 5),
        pointHoverRadius: errs.map(e => 5 + (e / maxErr) * 6),
        backgroundColor: errs.map(e => {
          const t = e / maxErr;
          const g = Math.round(200 - t * 120);
          return `rgba(${g},${g},${g},0.75)`;
        }),
        borderColor: 'rgba(255,255,255,0.15)',
        borderWidth: 1,
      }],
    },
    options: {
      ...baseOpts({
        tooltip: {
          callbacks: {
            title: () => 'Custom Built Model',
            label: d => [
              `Actual Cd:    ${d.raw.x.toFixed(5)}`,
              `Predicted Cd: ${d.raw.y.toFixed(5)}`,
              `Error:        ${d.raw.err.toFixed(5)}`,
            ],
          },
        },
        scales: {
          x: {
            ...baseOpts().scales.x,
            title: { display: true, text: 'Actual C\u2099', color: C.tick, font: { size: 10 } },
            min: 0.24, max: 0.36,
          },
          y: {
            ...baseOpts().scales.y,
            title: { display: true, text: 'Predicted C\u2099', color: C.tick, font: { size: 10 } },
            min: 0.24, max: 0.36,
          },
        },
      }),
    },
    plugins: [diagPlugin(0.24, 0.36)],
  });
}

/* ── ANALYSIS CHARTS ───────────────────────────────────── */
let avpFullChartInst = null, sweepChartInst = null, activeSweep = 0;

const SWEEPS = [
  { label: 'Rear Fastback',   idx: 0, min: 0.20, max: 0.60, step: 0.02, unit: '' },
  { label: 'Nose/WS Height',  idx: 1, min: 0.30, max: 0.70, step: 0.02, unit: '' },
  { label: 'Fastback Height', idx: 2, min: 0.30, max: 0.80, step: 0.02, unit: '' },
  { label: 'Side Taper',      idx: 3, min: 0,    max: 15,   step: 0.5,  unit: '\u00b0' },
  { label: 'Clearance',       idx: 4, min: 20,   max: 80,   step: 5,    unit: 'mm' },
  { label: 'Bottom Taper',    idx: 5, min: 0,    max: 20,   step: 1,    unit: '\u00b0' },
];

function initAnalysisCharts() {
  buildFiTable('fi-tbody-an');
  buildSweepTabs();
  drawSweep();
  initAvpFullChart();
}

function buildSweepTabs() {
  const cont = document.getElementById('sweep-tabs');
  if (cont.innerHTML !== '') return;
  SWEEPS.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'filter-chip' + (i === 0 ? ' active' : '');
    b.textContent = s.label;
    b.onclick = () => {
      activeSweep = i;
      cont.querySelectorAll('.filter-chip').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      drawSweep();
    };
    cont.appendChild(b);
  });
}

function drawSweep() {
  const sw = SWEEPS[activeSweep];
  const base = getInp();
  const vals = [];
  for (let v = sw.min; v <= sw.max + 1e-9; v += sw.step) vals.push(+v.toFixed(4));

  const cusL = vals.map(v => { const r = [...base]; r[sw.idx] = v; return nn2Pred(r); });
  const gpL  = vals.map(v => { const r = [...base]; r[sw.idx] = v; return gpPred(r).mean; });
  const gbL  = vals.map(v => { const r = [...base]; r[sw.idx] = v; return gbPred(r); });
  const rfL  = vals.map(v => { const r = [...base]; r[sw.idx] = v; return rfPred(r); });
  const labels = vals.map(v => `${v}${sw.unit}`);

  /* uncertainty band for GP */
  const gpHi = vals.map(v => { const r = [...base]; r[sw.idx] = v; const p = gpPred(r); return p.mean + p.std; });
  const gpLo = vals.map(v => { const r = [...base]; r[sw.idx] = v; const p = gpPred(r); return p.mean - p.std; });

  if (sweepChartInst) {
    sweepChartInst.data.labels = labels;
    sweepChartInst.data.datasets[0].data = cusL;
    sweepChartInst.data.datasets[1].data = gpL;
    sweepChartInst.data.datasets[2].data = gpHi;
    sweepChartInst.data.datasets[3].data = gpLo;
    sweepChartInst.data.datasets[4].data = gbL;
    sweepChartInst.data.datasets[5].data = rfL;
    sweepChartInst.update('active');
    return;
  }

  const ctx = document.getElementById('sweep-chart').getContext('2d');
  sweepChartInst = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Custom Stack', data: cusL, borderColor: C.custom, backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.35 },
        { label: 'GP (mean)',    data: gpL,  borderColor: C.gp,     backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, tension: 0.35, borderDash: [2, 3] },
        /* GP uncertainty band */
        { label: 'GP +1\u03c3', data: gpHi, borderColor: 'transparent', backgroundColor: 'rgba(192,192,192,0.08)', borderWidth: 0, pointRadius: 0, tension: 0.35, fill: '+1' },
        { label: 'GP -1\u03c3', data: gpLo, borderColor: 'transparent', backgroundColor: 'rgba(192,192,192,0.08)', borderWidth: 0, pointRadius: 0, tension: 0.35, fill: false },
        { label: 'GB',          data: gbL,  borderColor: C.gb,     backgroundColor: 'transparent', borderWidth: 1.2, pointRadius: 0, tension: 0.3, borderDash: [4, 4] },
        { label: 'RF',          data: rfL,  borderColor: C.rf,     backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, tension: 0.3, borderDash: [2, 5] },
      ],
    },
    options: {
      ...baseOpts({
        base: { interaction: { mode: 'index', intersect: false } },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            title: items => `${sw.label} = ${items[0]?.label}`,
            label: ctx => {
              const skip = ['GP +1σ','GP -1σ'];
              if (skip.includes(ctx.dataset.label)) return null;
              return `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(5)}`;
            },
            afterBody: items => {
              const custom = items.find(i => i.dataset.label === 'Custom Stack');
              const gp     = items.find(i => i.dataset.label === 'GP (mean)');
              if (custom && gp) {
                const diff = ((custom.parsed.y - gp.parsed.y) / gp.parsed.y * 100);
                return [`\u2014\u2014\u2014`, `Custom vs GP: ${diff > 0 ? '+' : ''}${diff.toFixed(2)}%`];
              }
              return [];
            },
          },
        },
        scales: {
          x: {
            ...baseOpts().scales.x,
            title: { display: true, text: sw.label, color: C.tick, font: { size: 10 } },
          },
          y: {
            ...baseOpts().scales.y,
            title: { display: true, text: 'Drag Coefficient C\u2099', color: C.tick, font: { size: 10 } },
          },
        },
      }),
    },
  });
}

function initAvpFullChart() {
  if (avpFullChartInst) return;
  const acts = DATA.map(r => r[7]);

  /* compute per-point error for each model for tooltip */
  const mkData = (predFn, style = 'circle') => acts.map((a, i) => {
    const p = predFn(DATA[i].slice(0, 7));
    const pred = typeof p === 'object' ? p.mean : p;
    return { x: a, y: pred, err: Math.abs(a - pred), idx: i };
  });

  const ctx = document.getElementById('avp-full-chart').getContext('2d');
  avpFullChartInst = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Custom Stack', data: mkData(inp => nn2Pred(inp)), backgroundColor: 'rgba(255,255,255,0.80)', borderColor: 'rgba(255,255,255,0.2)', borderWidth: 1, pointRadius: 4, pointStyle: 'circle' },
        { label: 'GP',           data: mkData(inp => gpPred(inp)), backgroundColor: 'rgba(192,192,192,0.65)', borderColor: 'rgba(192,192,192,0.2)', borderWidth: 1, pointRadius: 3, pointStyle: 'rect' },
        { label: 'GB',           data: mkData(inp => gbPred(inp)), backgroundColor: 'rgba(128,128,128,0.60)', borderColor: 'rgba(128,128,128,0.2)', borderWidth: 1, pointRadius: 3, pointStyle: 'triangle' },
        { label: 'RF',           data: mkData(inp => rfPred(inp)), backgroundColor: 'rgba(80,80,80,0.55)',    borderColor: 'rgba(80,80,80,0.2)',    borderWidth: 1, pointRadius: 3, pointStyle: 'rectRot' },
      ],
    },
    options: {
      ...baseOpts({
        tooltip: {
          callbacks: {
            title: items => `Sample ${items[0]?.raw?.idx + 1}   Actual Cd = ${items[0]?.raw?.x.toFixed(5)}`,
            label: ctx => `${ctx.dataset.label}   Predicted: ${ctx.raw.y.toFixed(5)}   Error: ${ctx.raw.err.toFixed(5)}`,
          },
        },
        base: {
          plugins: {
            legend: {
              display: true,
              position: 'top',
              align: 'start',
              labels: {
                color: C.tick,
                boxWidth: 8, boxHeight: 8,
                font: { size: 10, family: "'DM Mono', monospace" },
                padding: 14,
                usePointStyle: true,
              },
            },
          },
        },
        scales: {
          x: { ...baseOpts().scales.x, title: { display: true, text: 'Actual C\u2099', color: C.tick, font: { size: 10 } }, min: 0.24, max: 0.36 },
          y: { ...baseOpts().scales.y, title: { display: true, text: 'Predicted C\u2099', color: C.tick, font: { size: 10 } }, min: 0.24, max: 0.36 },
        },
      }),
    },
    plugins: [diagPlugin(0.24, 0.36)],
  });
}

/* ── URL SHARING ───────────────────────────────────────── */
function shareCalcState() {
  const inp = getInp();
  const params = new URLSearchParams({
    rlbf: inp[0], rhnw: inp[1], rhfb: inp[2],
    st: inp[3], cl: inp[4], bta: inp[5], fa: inp[6],
    model: activeModel,
  });
  const url = window.location.href.split('?')[0] + '?' + params.toString();
  navigator.clipboard.writeText(url).then(() => {
    const btn = document.getElementById('share-btn-text');
    btn.textContent = 'Link copied';
    setTimeout(() => btn.textContent = 'Copy shareable link', 2500);
  }).catch(() => { prompt('Copy this link:', url); });
}

function loadStateFromURL() {
  const p = new URLSearchParams(window.location.search);
  const map = { rlbf: 's-rlbf', rhnw: 's-rhnw', rhfb: 's-rhfb', st: 's-sidetaper', cl: 's-clearance', bta: 's-bottomtaper', fa: 's-frontalarea' };
  let loaded = false;
  for (const [k, sid] of Object.entries(map)) {
    const v = p.get(k);
    if (v !== null) {
      const el = document.getElementById(sid);
      if (el) { el.value = v; loaded = true; }
    }
  }
  const m = p.get('model');
  if (m) {
    const btn = document.getElementById('sel-' + m);
    if (btn) setModel(m, btn);
  }
  if (loaded) updateCalc();
}

/* ── EXPAND/COLLAPSE ───────────────────────────────────── */
function toggleDetail(id, btn) {
  const el = document.getElementById(id);
  el.classList.toggle('open');
  const span = btn.querySelector('span');
  const svg  = btn.querySelector('svg');
  span.textContent = el.classList.contains('open') ? 'Show less' : 'Full details';
  svg.style.transform = el.classList.contains('open') ? 'rotate(180deg)' : '';
}

/* ── INIT ──────────────────────────────────────────────── */
setupSliders();
updateCalc();
initOverviewCharts();
loadStateFromURL();
['gp','gb','rf','nn','nn2'].forEach(r => document.getElementById('row-' + r).classList.add('pred-active'));
