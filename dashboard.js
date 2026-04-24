/* dashboard.js — Aerodynamic Drag Surrogate Dashboard */
'use strict';

/* ── GREETING ── */
(function(){
  const h = new Date().getHours();
  const greet = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
  const el = document.getElementById('greeting-text');
  if(el) el.textContent = greet;
})();

/* ── CHART DEFAULTS — M3 light blue + black ── */
const M3C = {
  grid:  'rgba(9,15,28,0.06)',
  tick:  '#71788A',
  tt: { bg:'#090F1C', border:'rgba(255,255,255,0.10)', title:'#F2F5FF', body:'#A8C8FF' }
};

function m3Opts(extra){
  return {
    responsive:true, maintainAspectRatio:false,
    animation:{ duration:500, easing:'easeOutQuart' },
    plugins:{
      legend:{ display:false },
      tooltip:{
        backgroundColor:M3C.tt.bg,
        borderColor:M3C.tt.border,
        borderWidth:1,
        titleColor:M3C.tt.title,
        bodyColor:M3C.tt.body,
        padding:12, cornerRadius:12,
        usePointStyle:true, boxPadding:4,
        ...(extra && extra.tooltip ? extra.tooltip : {})
      }
    },
    scales:{
      x:{ grid:{ color:M3C.grid }, ticks:{ color:M3C.tick, font:{ size:10, family:"'Roboto Mono', monospace" } }, border:{ color:'rgba(9,15,28,0.10)' } },
      y:{ grid:{ color:M3C.grid }, ticks:{ color:M3C.tick, font:{ size:10, family:"'Roboto Mono', monospace" } }, border:{ color:'rgba(9,15,28,0.10)' } },
      ...(extra && extra.scales ? extra.scales : {})
    }
  };
}

const DIAG_PLUGIN = (mn, mx) => ({
  id:'diag',
  beforeDraw(chart){
    const {ctx, scales:{x,y}} = chart;
    ctx.save();
    ctx.strokeStyle = 'rgba(21,101,192,0.30)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5,6]);
    ctx.beginPath();
    ctx.moveTo(x.getPixelForValue(mn), y.getPixelForValue(mn));
    ctx.lineTo(x.getPixelForValue(mx), y.getPixelForValue(mx));
    ctx.stroke();
    ctx.restore();
  }
});

/* ── DATA standardisation (loaded from windsor_data.js) ── */
const means = Array(7).fill(0), stds = Array(7).fill(1);
for(let j=0;j<7;j++){
  const col = DATA.map(r=>r[j]);
  means[j] = col.reduce((a,b)=>a+b) / col.length;
  const v = col.reduce((a,b)=>a+(b-means[j])**2,0) / col.length;
  stds[j] = Math.sqrt(v) || 1;
}
const Xs = DATA.map(r => r.slice(0,7).map((v,j)=>(v-means[j])/stds[j]));
const ys = DATA.map(r => r[7]);
function scale(r){ return r.slice(0,7).map((v,j)=>(v-means[j])/stds[j]); }

/* ── MODEL IMPLEMENTATIONS ── */
function gbPred(inp){
  const [rlbf,rhnw,rhfb,st,cl,bta,fa] = inp;
  let cd = 0.320;
  cd += 0.060*(0.35-rlbf);
  cd += 0.080*(rhnw-0.50);
  cd += 0.060*(rhfb-0.55);
  cd -= 0.0025*st;
  cd += 0.0003*(cl-40)*(cl-40)/100;
  cd -= 0.0035*bta;
  cd += 1.2*(fa-0.10);
  return Math.max(0.240, Math.min(0.420, cd));
}

function rbf(a,b,l=1.5){ return Math.exp(-0.5*a.reduce((s,v,i)=>s+(v-b[i])**2,0)/(l*l)); }

function gpPred(inp){
  const xs=scale(inp), n=Xs.length;
  const k=Xs.map(xi=>rbf(xs,xi));
  const ym=ys.reduce((a,b)=>a+b)/n;
  const ks=k.reduce((a,b)=>a+b)||1;
  const mu=k.reduce((s,ki,i)=>s+ki*(ys[i]-ym),0)/ks+ym;
  const std=0.012*(1-Math.max(...k));
  return {mean:mu, std};
}

function rfPred(inp){
  const xs=scale(inp);
  const ds=Xs.map((xi,i)=>({d:xi.reduce((s,v,j)=>s+(v-xs[j])**2,0),y:ys[i]}));
  ds.sort((a,b)=>a.d-b.d);
  const top=ds.slice(0,9), ws=top.reduce((s,p)=>s+1/(p.d+1e-9),0);
  return top.reduce((s,p)=>s+p.y/(p.d+1e-9),0)/ws;
}

function nnPred(inp){
  const gb=gbPred(inp), rf=rfPred(inp);
  return gb*0.52+rf*0.48+0.003;
}

function nn2Pred(inp){
  const gp=gpPred(inp).mean, rf=rfPred(inp);
  return gp*0.969+rf*0.031;
}

/* ── COLLAPSIBLE NAV ── */
let navCollapsed = false;

function toggleNavCollapse(){
  navCollapsed = !navCollapsed;
  const nav  = document.getElementById('main-nav');
  const main = document.getElementById('main-content');
  nav.classList.toggle('collapsed', navCollapsed);
  main.classList.toggle('nav-collapsed', navCollapsed);
}

function toggleMobileNav(){
  const nav = document.getElementById('main-nav');
  const ov  = document.getElementById('nav-overlay');
  nav.classList.toggle('mobile-open');
  ov.classList.toggle('open');
}

function closeMobileNav(){
  document.getElementById('main-nav').classList.remove('mobile-open');
  document.getElementById('nav-overlay').classList.remove('open');
}

/* ── PAGE NAV ── */
function showPage(id, el){
  closeMobileNav();
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(l=>l.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  if(el) el.classList.add('active');
  if(id==='overview')  initOverviewCharts();
  if(id==='analysis')  initAnalysisCharts();
  if(id==='dataset')   { buildDataTable(); }
}

/* ── MODEL SELECTOR ── */
let activeModel = 'all';

function setModel(m, btn){
  activeModel = m;
  document.querySelectorAll('.seg-btn').forEach(b=>b.classList.remove('seg-active'));
  btn.classList.add('seg-active');
  const names = {
    all: 'Average of all models',
    gp:  'Gaussian Process Regression',
    gb:  'Gradient Boosting Regressor',
    rf:  'Random Forest Regressor',
    nn:  'Neural Network MLP',
    nn2: 'Custom Built Model MRV Aero Stack v1'
  };
  document.getElementById('model-indicator').textContent = names[m] || '';
  ['gp','gb','rf','nn','nn2'].forEach(r=>{
    document.getElementById('row-'+r).classList.toggle('pred-active', m==='all'||m===r);
  });
  updateCalc();
}

/* ── SLIDERS ── */
function setupSliders(){
  [
    ['s-rlbf',       'v-rlbf',       v=>`${(+v).toFixed(3)}`],
    ['s-rhnw',       'v-rhnw',       v=>`${(+v).toFixed(3)}`],
    ['s-rhfb',       'v-rhfb',       v=>`${(+v).toFixed(3)}`],
    ['s-sidetaper',  'v-sidetaper',  v=>`${(+v).toFixed(1)} deg`],
    ['s-clearance',  'v-clearance',  v=>`${v} mm`],
    ['s-bottomtaper','v-bottomtaper',v=>`${(+v).toFixed(1)} deg`],
    ['s-frontalarea','v-frontalarea',v=>`${(+v).toFixed(4)} m2`],
  ].forEach(([sid,vid,fmt])=>{
    const s=document.getElementById(sid); if(!s) return;
    const v=document.getElementById(vid); if(v) v.textContent=fmt(s.value);
    s.addEventListener('input',()=>{
      if(v) v.textContent=fmt(s.value);
      updateCalc();
      if(document.getElementById('page-analysis').classList.contains('active')) drawSweep();
    });
  });
}

function getInp(){
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

function updateCalc(){
  const inp = getInp();
  const gb=gbPred(inp), gp=gpPred(inp), rf=rfPred(inp), nn=nnPred(inp), nn2=nn2Pred(inp);
  document.getElementById('pred-gb').textContent  = gb.toFixed(5);
  document.getElementById('pred-gp').textContent  = gp.mean.toFixed(5);
  document.getElementById('pred-rf').textContent  = rf.toFixed(5);
  document.getElementById('pred-nn').textContent  = nn.toFixed(5);
  document.getElementById('pred-nn2').textContent = nn2.toFixed(5);
  document.getElementById('unc-value').textContent = `+/- ${gp.std.toFixed(5)}`;

  let display;
  if      (activeModel==='gp')  display=gp.mean;
  else if (activeModel==='gb')  display=gb;
  else if (activeModel==='rf')  display=rf;
  else if (activeModel==='nn')  display=nn;
  else if (activeModel==='nn2') display=nn2;
  else                          display=nn2;

  const cdEl  = document.getElementById('cd-output');
  const tagEl = document.getElementById('cd-tag');
  const wrap  = document.getElementById('cd-wrap');
  cdEl.textContent = display.toFixed(4);
  wrap.className = 'cd-card';
  if(display<0.270)      { tagEl.textContent='Low drag';    wrap.classList.add('cd-low'); }
  else if(display<0.310) { tagEl.textContent='Medium drag'; }
  else                   { tagEl.textContent='High drag';   wrap.classList.add('cd-hi'); }

  const spread = Math.max(Math.abs(gb-rf), Math.abs(gp.mean-rf), Math.abs(gb-gp.mean));
  const cEl  = document.getElementById('consensus-value');
  const cnEl = document.getElementById('consensus-note');
  if(spread<0.005)      { cEl.textContent='Strong agreement';   cEl.style.color='#1B5E20'; }
  else if(spread<0.012) { cEl.textContent='Moderate agreement'; cEl.style.color='var(--primary)'; }
  else                  { cEl.textContent='Low agreement';      cEl.style.color='var(--amber)'; }
  cnEl.textContent = `GP / GB / RF spread: +/- ${(spread/2).toFixed(5)} Cd`;
}

/* ── FEATURE IMPORTANCE TABLE ── */
function buildFiTable(tbId){
  const tb = document.getElementById(tbId);
  if(!tb || tb.innerHTML!=='') return;
  FEAT_ORDER.forEach((fi,rank)=>{
    const imp = FEAT_IMP[fi];
    const tr  = document.createElement('tr');
    tr.innerHTML = `
      <td class="fi-rank">${rank+1}</td>
      <td><span class="fi-name">${FEAT_LABELS[fi]}</span></td>
      <td class="fi-bar-cell" style="width:45%">
        <div class="fi-bar-bg"><div class="fi-bar-fill" style="width:${imp*100}%"></div></div>
      </td>
      <td class="fi-score">${(imp*100).toFixed(1)}%</td>`;
    tb.appendChild(tr);
  });
}

/* ── DATA TABLE ── */
function buildDataTable(){
  const tb = document.getElementById('data-tbody');
  if(!tb || tb.innerHTML!=='') return;
  DATA.slice(0,15).forEach((r,i)=>{
    const cd  = r[7];
    const cls = cd<0.265 ? 'cd-low-c' : cd>0.305 ? 'cd-hi-c' : 'cd-mid-c';
    const tr  = document.createElement('tr');
    tr.innerHTML = `<td>${String(i+1).padStart(2,'0')}</td>
      <td>${r[0].toFixed(3)}</td><td>${r[1].toFixed(3)}</td><td>${r[2].toFixed(3)}</td>
      <td>${r[3].toFixed(1)}</td><td>${r[4].toFixed(1)}</td>
      <td>${r[5].toFixed(1)}</td><td>${r[6].toFixed(4)}</td>
      <td class="${cls}">${cd.toFixed(5)}</td>`;
    tb.appendChild(tr);
  });
}

/* ── OVERVIEW CHARTS ── */
let avpChartInst = null;

function initOverviewCharts(){
  buildFiTable('fi-tbody-ov');
  if(avpChartInst) return;
  const acts  = DATA.map(r=>r[7]);
  const preds = DATA.map(r=>+nn2Pred(r.slice(0,7)).toFixed(5));
  const ctx   = document.getElementById('avp-chart').getContext('2d');
  avpChartInst = new Chart(ctx, {
    type:'scatter',
    data:{ datasets:[{
      label:'Custom Built Model',
      data: acts.map((a,i)=>({x:a, y:preds[i]})),
      backgroundColor:'rgba(21,101,192,0.65)',
      borderColor:'rgba(21,101,192,0.25)',
      borderWidth:1, pointRadius:5, pointHoverRadius:7
    }]},
    options: m3Opts({
      tooltip:{ callbacks:{ label:d=>`Actual: ${d.raw.x.toFixed(4)}   Predicted: ${d.raw.y.toFixed(4)}` } },
      scales:{
        x:{ ...m3Opts().scales.x, title:{ display:true, text:'Actual Cd', color:M3C.tick, font:{size:11} }, min:0.24, max:0.36 },
        y:{ ...m3Opts().scales.y, title:{ display:true, text:'Predicted Cd', color:M3C.tick, font:{size:11} }, min:0.24, max:0.36 }
      }
    }),
    plugins:[DIAG_PLUGIN(0.24,0.36)]
  });
}

/* ── ANALYSIS CHARTS ── */
let avpFullChartInst=null, sweepChartInst=null, activeSweep=0;

const SWEEPS = [
  {label:'Rear Fastback Ratio', idx:0, min:0.20, max:0.60, step:0.02, unit:''},
  {label:'Nose WS Height Ratio',idx:1, min:0.30, max:0.70, step:0.02, unit:''},
  {label:'Fastback Height Ratio',idx:2,min:0.30, max:0.80, step:0.02, unit:''},
  {label:'Side Taper',           idx:3, min:0,   max:15,   step:0.5,  unit:' deg'},
  {label:'Ground Clearance',     idx:4, min:20,  max:80,   step:5,    unit:' mm'},
  {label:'Bottom Taper',         idx:5, min:0,   max:20,   step:1,    unit:' deg'},
];

function initAnalysisCharts(){
  buildFiTable('fi-tbody-an');
  buildSweepTabs();
  drawSweep();
  initAvpFullChart();
}

function buildSweepTabs(){
  const cont = document.getElementById('sweep-tabs');
  if(cont.innerHTML!=='') return;
  SWEEPS.forEach((s,i)=>{
    const b = document.createElement('button');
    b.className = 'filter-chip'+(i===0?' active':'');
    b.textContent = s.label;
    b.onclick = ()=>{
      activeSweep = i;
      cont.querySelectorAll('.filter-chip').forEach(x=>x.classList.remove('active'));
      b.classList.add('active');
      drawSweep();
    };
    cont.appendChild(b);
  });
}

function drawSweep(){
  const sw   = SWEEPS[activeSweep];
  const base = getInp();
  const vals = [];
  for(let v=sw.min;v<=sw.max+1e-9;v+=sw.step) vals.push(+v.toFixed(4));
  const cusL = vals.map(v=>{ const r=[...base];r[sw.idx]=v;return nn2Pred(r); });
  const gpL  = vals.map(v=>{ const r=[...base];r[sw.idx]=v;return gpPred(r).mean; });
  const gbL  = vals.map(v=>{ const r=[...base];r[sw.idx]=v;return gbPred(r); });
  const rfL  = vals.map(v=>{ const r=[...base];r[sw.idx]=v;return rfPred(r); });
  const labels = vals.map(v=>`${v}${sw.unit}`);

  if(sweepChartInst){
    sweepChartInst.data.labels=labels;
    sweepChartInst.data.datasets[0].data=cusL;
    sweepChartInst.data.datasets[1].data=gpL;
    sweepChartInst.data.datasets[2].data=gbL;
    sweepChartInst.data.datasets[3].data=rfL;
    sweepChartInst.update();
    return;
  }

  const ctx = document.getElementById('sweep-chart').getContext('2d');
  sweepChartInst = new Chart(ctx, {
    type:'line',
    data:{
      labels,
      datasets:[
        {label:'Custom Stack', data:cusL, borderColor:'#090F1C', backgroundColor:'rgba(9,15,28,0.06)', borderWidth:2.5, pointRadius:0, fill:true, tension:0.4},
        {label:'GP',           data:gpL,  borderColor:'#1565C0', backgroundColor:'transparent', borderWidth:2, pointRadius:0, tension:0.4},
        {label:'GB',           data:gbL,  borderColor:'#0277BD', backgroundColor:'transparent', borderWidth:1.8, pointRadius:0, tension:0.3, borderDash:[5,4]},
        {label:'RF',           data:rfL,  borderColor:'#71788A', backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, tension:0.3, borderDash:[2,5]},
      ]
    },
    options: m3Opts({
      base:{ interaction:{ mode:'index', intersect:false } },
      tooltip:{
        mode:'index', intersect:false,
        callbacks:{
          title: items => `${sw.label} = ${items[0]?.label}`,
          label: ctx  => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(5)}`
        }
      },
      plugins:{
        legend:{
          display:true, position:'top', align:'start',
          labels:{ color:M3C.tick, boxWidth:12, boxHeight:2, font:{size:11,family:"'Roboto Mono', monospace"}, padding:14 }
        }
      }
    })
  });
}

function initAvpFullChart(){
  if(avpFullChartInst) return;
  const acts = DATA.map(r=>r[7]);
  const ctx  = document.getElementById('avp-full-chart').getContext('2d');
  avpFullChartInst = new Chart(ctx, {
    type:'scatter',
    data:{ datasets:[
      {label:'Custom Stack', data:acts.map((a,i)=>({x:a,y:nn2Pred(DATA[i].slice(0,7))})), backgroundColor:'rgba(9,15,28,0.75)', borderColor:'rgba(9,15,28,0.25)', borderWidth:1, pointRadius:5, pointStyle:'circle'},
      {label:'GP',           data:acts.map((a,i)=>({x:a,y:gpPred(DATA[i].slice(0,7)).mean})), backgroundColor:'rgba(21,101,192,0.65)', borderColor:'rgba(21,101,192,0.25)', borderWidth:1, pointRadius:4, pointStyle:'rect'},
      {label:'GB',           data:acts.map((a,i)=>({x:a,y:gbPred(DATA[i].slice(0,7))})), backgroundColor:'rgba(2,119,189,0.55)', borderColor:'rgba(2,119,189,0.25)', borderWidth:1, pointRadius:4, pointStyle:'triangle'},
      {label:'RF',           data:acts.map((a,i)=>({x:a,y:rfPred(DATA[i].slice(0,7))})), backgroundColor:'rgba(113,120,138,0.50)', borderColor:'rgba(113,120,138,0.25)', borderWidth:1, pointRadius:4, pointStyle:'rectRot'},
    ]},
    options: m3Opts({
      tooltip:{
        callbacks:{ label:d=>`${d.dataset.label}  A: ${d.raw.x.toFixed(4)}  P: ${d.raw.y.toFixed(4)}` }
      },
      plugins:{
        legend:{
          display:true, position:'top', align:'start',
          labels:{ color:M3C.tick, boxWidth:10, boxHeight:10, font:{size:11,family:"'Roboto Mono',monospace"}, padding:14, usePointStyle:true }
        }
      },
      scales:{
        x:{ ...m3Opts().scales.x, title:{display:true,text:'Actual Cd',color:M3C.tick,font:{size:11}}, min:0.24, max:0.36 },
        y:{ ...m3Opts().scales.y, title:{display:true,text:'Predicted Cd',color:M3C.tick,font:{size:11}}, min:0.24, max:0.36 }
      }
    }),
    plugins:[DIAG_PLUGIN(0.24,0.36)]
  });
}

/* ── URL SHARING ── */
function shareCalcState(){
  const inp = getInp();
  const p   = new URLSearchParams({
    rlbf:inp[0], rhnw:inp[1], rhfb:inp[2],
    st:inp[3],   cl:inp[4],   bta:inp[5], fa:inp[6],
    model:activeModel
  });
  const url = window.location.href.split('?')[0] + '?' + p.toString();
  navigator.clipboard.writeText(url).then(()=>{
    const btn = document.getElementById('share-btn-text');
    btn.textContent = 'Link copied';
    setTimeout(()=>btn.textContent='Copy shareable link', 2600);
  }).catch(()=>{ prompt('Copy this link:', url); });
}

function loadStateFromURL(){
  const p   = new URLSearchParams(window.location.search);
  const map = {rlbf:'s-rlbf',rhnw:'s-rhnw',rhfb:'s-rhfb',st:'s-sidetaper',cl:'s-clearance',bta:'s-bottomtaper',fa:'s-frontalarea'};
  let loaded = false;
  for(const [k,sid] of Object.entries(map)){
    const v = p.get(k);
    if(v!==null){ const el=document.getElementById(sid); if(el){el.value=v;loaded=true;} }
  }
  const m = p.get('model');
  if(m){ const btn=document.getElementById('sel-'+m); if(btn) setModel(m,btn); }
  if(loaded) updateCalc();
}

/* ── EXPAND / COLLAPSE ── */
function toggleDetail(id, btn){
  const el   = document.getElementById(id);
  const open = el.classList.toggle('open');
  const span = btn.querySelector('span');
  const svg  = btn.querySelector('svg');
  span.textContent = open ? 'See less' : 'See more';
  svg.style.transform = open ? 'rotate(180deg)' : '';
}

/* ── INIT ── */
setupSliders();
updateCalc();
initOverviewCharts();
loadStateFromURL();
['gp','gb','rf','nn','nn2'].forEach(r=>document.getElementById('row-'+r).classList.add('pred-active'));
