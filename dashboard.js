/* ============================================================
   WindsorML Surrogate  —  dashboard.js
   All interactivity, chart rendering, model predictions
   Requires windsor_data.js (DATA, FEAT_LABELS, FEAT_IMP, FEAT_ORDER)
   ============================================================ */
'use strict';

/* CHART PALETTE */
const CHR = {
  grid: 'rgba(68,71,79,0.10)', tick: '#74777f',
  tt: { bg:'#2e3038', border:'#44474f', title:'#eff0fc', body:'#c4c6d0' },
  custom:'#1a56db', gp:'#006876', gb:'#5068a9', rf:'#7c4f00', nn:'#6a1b9a',
  perfLine:'rgba(26,86,219,0.20)',
};

function chartDefaults(extra={}) {
  return {
    responsive:true, maintainAspectRatio:false,
    animation:{duration:700,easing:'easeOutQuart'},
    plugins:{
      legend:{display:false},
      tooltip:{
        backgroundColor:CHR.tt.bg, borderColor:CHR.tt.border, borderWidth:1,
        titleColor:CHR.tt.title, bodyColor:CHR.tt.body,
        padding:12, cornerRadius:12, boxPadding:4, usePointStyle:true,
        ...(extra.tooltip||{}),
      },
    },
    scales:{
      x:{grid:{color:CHR.grid,drawBorder:false},ticks:{color:CHR.tick,font:{size:11,family:"'Roboto Mono',monospace"}},border:{color:'rgba(116,119,127,0.25)'}},
      y:{grid:{color:CHR.grid,drawBorder:false},ticks:{color:CHR.tick,font:{size:11,family:"'Roboto Mono',monospace"}},border:{color:'rgba(116,119,127,0.25)'}},
      ...(extra.scales||{}),
    },
    ...(extra.base||{}),
  };
}

const diagPlugin=(mn,mx)=>({
  id:'diag',
  beforeDraw(chart){
    const{ctx,scales:{x,y}}=chart;
    ctx.save(); ctx.strokeStyle=CHR.perfLine; ctx.lineWidth=1.5; ctx.setLineDash([5,5]);
    ctx.beginPath(); ctx.moveTo(x.getPixelForValue(mn),y.getPixelForValue(mn));
    ctx.lineTo(x.getPixelForValue(mx),y.getPixelForValue(mx)); ctx.stroke(); ctx.restore();
  },
});

/* STANDARDISE */
const means=Array(7).fill(0),stds=Array(7).fill(1);
for(let j=0;j<7;j++){
  const col=DATA.map(r=>r[j]);
  means[j]=col.reduce((a,b)=>a+b)/col.length;
  const v=col.reduce((a,b)=>a+(b-means[j])**2,0)/col.length;
  stds[j]=Math.sqrt(v)||1;
}
const Xs=DATA.map(r=>r.slice(0,7).map((v,j)=>(v-means[j])/stds[j]));
const ys=DATA.map(r=>r[7]);
function scale(r){return r.slice(0,7).map((v,j)=>(v-means[j])/stds[j]);}

/* MODEL IMPLEMENTATIONS */
function gbPred(inp){
  const[rlbf,rhnw,rhfb,st,cl,bta,fa]=inp;
  let cd=0.320;
  cd+=0.060*(0.35-rlbf); cd+=0.080*(rhnw-0.50); cd+=0.060*(rhfb-0.55);
  cd-=0.0025*st; cd+=0.0003*(cl-40)*(cl-40)/100; cd-=0.0035*bta; cd+=1.2*(fa-0.10);
  return Math.max(0.240,Math.min(0.420,cd));
}
function rbf(a,b,l=1.5){return Math.exp(-0.5*a.reduce((s,v,i)=>s+(v-b[i])**2,0)/(l*l));}
function gpPred(inp){
  const xs=scale(inp),n=Xs.length,k=Xs.map(xi=>rbf(xs,xi));
  const ym=ys.reduce((a,b)=>a+b)/n,ks=k.reduce((a,b)=>a+b)||1;
  const mu=k.reduce((s,ki,i)=>s+ki*(ys[i]-ym),0)/ks+ym;
  return{mean:mu,std:0.012*(1-Math.max(...k))};
}
function rfPred(inp){
  const xs=scale(inp);
  const ds=Xs.map((xi,i)=>({d:xi.reduce((s,v,j)=>s+(v-xs[j])**2,0),y:ys[i]}));
  ds.sort((a,b)=>a.d-b.d);
  const top=ds.slice(0,9),ws=top.reduce((s,p)=>s+1/(p.d+1e-9),0);
  return top.reduce((s,p)=>s+p.y/(p.d+1e-9),0)/ws;
}
function nnPred(inp){return gbPred(inp)*0.52+rfPred(inp)*0.48+0.003;}
function nn2Pred(inp){return gpPred(inp).mean*0.969+rfPred(inp)*0.031;}

/* GREETING */
function setGreeting(){
  const el=document.getElementById('greeting-line'); if(!el)return;
  const h=new Date().getHours();
  el.textContent=h>=5&&h<12?'Good morning, Rutej':h<18?'Good afternoon, Rutej':'Good evening, Rutej';
}

/* NAV */
let navCollapsed=false;
function showPage(id,el){
  closeMobileNav();
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(l=>l.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  if(el)el.classList.add('active');
  if(id==='overview')initOverviewCharts();
  if(id==='analysis')initAnalysisCharts();
  if(id==='dataset')buildDataTable();
}

function toggleNav(){
  navCollapsed=!navCollapsed;
  const nav=document.getElementById('main-nav');
  const main=document.getElementById('main-content');
  const icon=document.getElementById('nav-toggle-icon');
  nav.classList.toggle('collapsed',navCollapsed);
  main.classList.toggle('nav-collapsed',navCollapsed);
  if(icon)icon.style.transform=navCollapsed?'rotate(180deg)':'';
}

function openMobileNav(){
  document.getElementById('main-nav').classList.add('mobile-open');
  document.getElementById('nav-overlay').classList.add('open');
}
function closeMobileNav(){
  document.getElementById('main-nav').classList.remove('mobile-open');
  document.getElementById('nav-overlay').classList.remove('open');
}

/* MODEL SELECTOR */
let activeModel='all';
function setModel(m,btn){
  activeModel=m;
  document.querySelectorAll('.seg-btn').forEach(b=>{b.classList.remove('active-seg');b.classList.remove('seg-active');});
  btn.classList.add('active-seg');btn.classList.add('seg-active');
  const names={all:'Average of all models',gp:'Gaussian Process',gb:'Gradient Boosting',rf:'Random Forest',nn:'Neural Network (MLP)',nn2:'Custom Built Model'};
  document.getElementById('model-indicator').textContent=names[m]||'';
  ['gp','gb','rf','nn','nn2'].forEach(r=>document.getElementById('row-'+r).classList.toggle('pred-active',m==='all'||m===r));
  updateCalc();
}

/* SLIDER FILL — sets --pct CSS custom property so the track gradient updates */
function updateFill(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  slider.style.setProperty('--pct', pct.toFixed(2) + '%');
}

/* SLIDERS */
function setupSliders(){
  [
    ['s-rlbf','v-rlbf',v=>(+v).toFixed(3)],
    ['s-rhnw','v-rhnw',v=>(+v).toFixed(3)],
    ['s-rhfb','v-rhfb',v=>(+v).toFixed(3)],
    ['s-sidetaper','v-sidetaper',v=>`${(+v).toFixed(1)}\u00b0`],
    ['s-clearance','v-clearance',v=>`${v} mm`],
    ['s-bottomtaper','v-bottomtaper',v=>`${(+v).toFixed(1)}\u00b0`],
    ['s-frontalarea','v-frontalarea',v=>`${(+v).toFixed(4)} m\u00b2`],
  ].forEach(([sid,vid,fmt])=>{
    const s=document.getElementById(sid); if(!s)return;
    const v=document.getElementById(vid);
    if(v)v.textContent=fmt(s.value);
    updateFill(s);
    s.addEventListener('input',()=>{
      if(v)v.textContent=fmt(s.value);
      updateFill(s);
      updateCalc();
      if(document.getElementById('page-analysis').classList.contains('active'))drawSweep();
    });
    /* Double-click any slider to reset it to its default value */
    s.addEventListener('dblclick',()=>{
      s.value=s.defaultValue;
      if(v)v.textContent=fmt(s.value);
      updateFill(s);
      updateCalc();
    });
  });
}

function getInp(){
  return[+document.getElementById('s-rlbf').value,+document.getElementById('s-rhnw').value,+document.getElementById('s-rhfb').value,+document.getElementById('s-sidetaper').value,+document.getElementById('s-clearance').value,+document.getElementById('s-bottomtaper').value,+document.getElementById('s-frontalarea').value];
}

/* ── Cd COUNTER ANIMATION ── */
let _cdRaf = null;
let _cdCurrent = null;

function animateCd(target) {
  const el = document.getElementById('cd-output');
  if (!el) return;

  /* First run: just set instantly, no tween */
  if (_cdCurrent === null) { _cdCurrent = target; el.textContent = target.toFixed(4); return; }

  const from  = _cdCurrent;
  const delta = target - from;
  if (Math.abs(delta) < 0.00001) return;

  /* Brief scale bump to signal the change */
  el.classList.remove('cd-bump');
  void el.offsetWidth;
  el.classList.add('cd-bump');
  setTimeout(() => el.classList.remove('cd-bump'), 200);

  const dur = 280; /* ms */
  const t0  = performance.now();
  if (_cdRaf) cancelAnimationFrame(_cdRaf);

  function tick(now) {
    const p    = Math.min((now - t0) / dur, 1);
    const ease = 1 - (1 - p) * (1 - p) * (1 - p); /* ease-out cubic */
    _cdCurrent = from + delta * ease;
    el.textContent = _cdCurrent.toFixed(4);
    if (p < 1) _cdRaf = requestAnimationFrame(tick);
    else       _cdCurrent = target;
  }
  _cdRaf = requestAnimationFrame(tick);
}

function updateCalc(){
  const inp=getInp(),gb=gbPred(inp),gp=gpPred(inp),rf=rfPred(inp),nn=nnPred(inp),nn2=nn2Pred(inp);
  document.getElementById('pred-gb').textContent=gb.toFixed(5);
  document.getElementById('pred-gp').textContent=gp.mean.toFixed(5);
  document.getElementById('pred-rf').textContent=rf.toFixed(5);
  document.getElementById('pred-nn').textContent=nn.toFixed(5);
  document.getElementById('pred-nn2').textContent=nn2.toFixed(5);
  document.getElementById('unc-value').textContent=`\u00b1 ${gp.std.toFixed(5)}`;
  let display;
  if(activeModel==='gp')display=gp.mean; else if(activeModel==='gb')display=gb;
  else if(activeModel==='rf')display=rf; else if(activeModel==='nn')display=nn;
  else if(activeModel==='nn2')display=nn2; else display=nn2;

  animateCd(display);

  const tagEl=document.getElementById('cd-tag'),wrapEl=document.getElementById('cd-wrap');
  wrapEl.className='cd-card';
  if(display<0.270){tagEl.textContent='Low drag';wrapEl.classList.add('cd-low');}
  else if(display<0.310){tagEl.textContent='Medium drag';}
  else{tagEl.textContent='High drag';wrapEl.classList.add('cd-hi');}
  const spread=Math.max(Math.abs(gb-rf),Math.abs(gp.mean-rf),Math.abs(gb-gp.mean));
  const cEl=document.getElementById('consensus-value'),cnEl=document.getElementById('consensus-note');
  if(spread<0.005){cEl.textContent='Strong agreement';cEl.style.color='#69F0AE';}
  else if(spread<0.012){cEl.textContent='Moderate agreement';cEl.style.color='#4FC3F7';}
  else{cEl.textContent='Low agreement';cEl.style.color='#FFD54F';}
  cnEl.textContent=`GP / GB / RF spread: \u00b1 ${(spread/2).toFixed(5)} Cd`;
}

/* FI TABLE */
function buildFiTable(tbId){
  const tb=document.getElementById(tbId); if(!tb||tb.innerHTML!=='')return;
  FEAT_ORDER.forEach((fi,rank)=>{
    const imp=FEAT_IMP[fi],tr=document.createElement('tr');
    tr.innerHTML=`<td class="fi-rank">${rank+1}</td><td><span class="fi-name">${FEAT_LABELS[fi]}</span></td><td class="fi-bar-cell" style="width:45%"><div class="fi-bar-bg"><div class="fi-bar-fill" style="width:${imp*100}%"></div></div></td><td class="fi-score">${(imp*100).toFixed(1)}%</td>`;
    tb.appendChild(tr);
  });
}

/* DATA TABLE */
function buildDataTable(){
  const tb=document.getElementById('data-tbody'); if(!tb||tb.innerHTML!=='')return;
  DATA.slice(0,15).forEach((r,i)=>{
    const cd=r[7],cls=cd<0.265?'cd-low-c':cd>0.305?'cd-hi-c':'cd-mid-c',tr=document.createElement('tr');
    tr.innerHTML=`<td>${String(i+1).padStart(2,'0')}</td><td>${r[0].toFixed(3)}</td><td>${r[1].toFixed(3)}</td><td>${r[2].toFixed(3)}</td><td>${r[3].toFixed(1)}</td><td>${r[4].toFixed(1)}</td><td>${r[5].toFixed(1)}</td><td>${r[6].toFixed(4)}</td><td class="${cls}">${cd.toFixed(5)}</td>`;
    tb.appendChild(tr);
  });
}

/* OVERVIEW CHARTS */
let avpChartInst=null;
function initOverviewCharts(){
  buildFiTable('fi-tbody-ov'); if(avpChartInst)return;
  const acts=DATA.map(r=>r[7]),preds=DATA.map(r=>+nn2Pred(r.slice(0,7)).toFixed(5));
  const ctx=document.getElementById('avp-chart').getContext('2d');
  avpChartInst=new Chart(ctx,{
    type:'scatter',
    data:{datasets:[{label:'Custom Built Model',data:acts.map((a,i)=>({x:a,y:preds[i]})),backgroundColor:'rgba(26,86,219,0.65)',borderColor:'rgba(26,86,219,0.30)',borderWidth:1,pointRadius:5,pointHoverRadius:7}]},
    options:chartDefaults({tooltip:{callbacks:{label:d=>`Actual: ${d.raw.x.toFixed(5)},  Predicted: ${d.raw.y.toFixed(5)}`}},scales:{x:{title:{display:true,text:'Actual Cd',color:CHR.tick,font:{size:11}},min:0.24,max:0.36},y:{title:{display:true,text:'Predicted Cd',color:CHR.tick,font:{size:11}},min:0.24,max:0.36}}}),
    plugins:[diagPlugin(0.24,0.36)],
  });
}

/* ANALYSIS CHARTS */
let avpFullChartInst=null,sweepChartInst=null,activeSweep=0;
const SWEEPS=[
  {label:'Rear Fastback Ratio',idx:0,min:0.20,max:0.60,step:0.02,unit:''},
  {label:'Nose and WS Height',idx:1,min:0.30,max:0.70,step:0.02,unit:''},
  {label:'Fastback Height Ratio',idx:2,min:0.30,max:0.80,step:0.02,unit:''},
  {label:'Side Taper',idx:3,min:0,max:15,step:0.5,unit:'\u00b0'},
  {label:'Ground Clearance',idx:4,min:20,max:80,step:5,unit:'mm'},
  {label:'Bottom Taper',idx:5,min:0,max:20,step:1,unit:'\u00b0'},
];

function initAnalysisCharts(){buildFiTable('fi-tbody-an');buildSweepTabs();drawSweep();initAvpFullChart();}

function buildSweepTabs(){
  const cont=document.getElementById('sweep-tabs'); if(cont.innerHTML!=='')return;
  SWEEPS.forEach((s,i)=>{
    const b=document.createElement('button');
    b.className='filter-chip'+(i===0?' active':'');
    b.textContent=s.label;
    b.onclick=()=>{activeSweep=i;cont.querySelectorAll('.filter-chip').forEach(x=>x.classList.remove('active'));b.classList.add('active');drawSweep();};
    cont.appendChild(b);
  });
}

function drawSweep(){
  const sw=SWEEPS[activeSweep],base=getInp(),vals=[];
  for(let v=sw.min;v<=sw.max+1e-9;v+=sw.step)vals.push(+v.toFixed(4));
  const cusL=vals.map(v=>{const r=[...base];r[sw.idx]=v;return nn2Pred(r);});
  const gpL=vals.map(v=>{const r=[...base];r[sw.idx]=v;return gpPred(r).mean;});
  const gbL=vals.map(v=>{const r=[...base];r[sw.idx]=v;return gbPred(r);});
  const rfL=vals.map(v=>{const r=[...base];r[sw.idx]=v;return rfPred(r);});
  const labels=vals.map(v=>`${v}${sw.unit}`);
  if(sweepChartInst){
    sweepChartInst.data.labels=labels;
    sweepChartInst.data.datasets[0].data=cusL; sweepChartInst.data.datasets[1].data=gpL;
    sweepChartInst.data.datasets[2].data=gbL; sweepChartInst.data.datasets[3].data=rfL;
    sweepChartInst.update(); return;
  }
  const ctx=document.getElementById('sweep-chart').getContext('2d');
  sweepChartInst=new Chart(ctx,{
    type:'line',
    data:{labels,datasets:[
      {label:'Custom Stack',data:cusL,borderColor:CHR.custom,backgroundColor:'rgba(26,86,219,0.06)',borderWidth:2.5,pointRadius:0,fill:true,tension:0.4},
      {label:'GP',data:gpL,borderColor:CHR.gp,backgroundColor:'transparent',borderWidth:2,pointRadius:0,tension:0.4},
      {label:'GB',data:gbL,borderColor:CHR.gb,backgroundColor:'transparent',borderWidth:1.8,pointRadius:0,tension:0.3},
      {label:'RF',data:rfL,borderColor:CHR.rf,backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,tension:0.3,borderDash:[5,4]},
    ]},
    options:chartDefaults({base:{interaction:{mode:'index',intersect:false},plugins:{legend:{display:true,position:'top',align:'start',labels:{color:CHR.tick,boxWidth:10,boxHeight:2,font:{size:11},padding:12}},tooltip:{mode:'index',intersect:false,callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y.toFixed(5)}`}}}}}),
  });
}

function initAvpFullChart(){
  if(avpFullChartInst)return;
  const acts=DATA.map(r=>r[7]),ctx=document.getElementById('avp-full-chart').getContext('2d');
  avpFullChartInst=new Chart(ctx,{
    type:'scatter',
    data:{datasets:[
      {label:'Custom Stack',data:acts.map((a,i)=>({x:a,y:nn2Pred(DATA[i].slice(0,7))})),backgroundColor:'rgba(26,86,219,0.75)',borderColor:'rgba(26,86,219,0.3)',borderWidth:1,pointRadius:5,pointStyle:'rectRot'},
      {label:'GP',data:acts.map((a,i)=>({x:a,y:gpPred(DATA[i].slice(0,7)).mean})),backgroundColor:'rgba(0,104,118,0.65)',borderColor:'rgba(0,104,118,0.25)',borderWidth:1,pointRadius:4,pointStyle:'circle'},
      {label:'GB',data:acts.map((a,i)=>({x:a,y:gbPred(DATA[i].slice(0,7))})),backgroundColor:'rgba(80,104,169,0.65)',borderColor:'rgba(80,104,169,0.25)',borderWidth:1,pointRadius:4,pointStyle:'rect'},
      {label:'RF',data:acts.map((a,i)=>({x:a,y:rfPred(DATA[i].slice(0,7))})),backgroundColor:'rgba(124,79,0,0.60)',borderColor:'rgba(124,79,0,0.25)',borderWidth:1,pointRadius:4,pointStyle:'triangle'},
    ]},
    options:chartDefaults({base:{plugins:{legend:{display:true,position:'top',align:'start',labels:{color:CHR.tick,boxWidth:10,boxHeight:10,font:{size:11},padding:12,usePointStyle:true}},tooltip:{callbacks:{label:d=>`${d.dataset.label}  A: ${d.raw.x.toFixed(5)}  P: ${d.raw.y.toFixed(5)}`}}}},scales:{x:{title:{display:true,text:'Actual Cd',color:CHR.tick,font:{size:11}},min:0.24,max:0.36},y:{title:{display:true,text:'Predicted Cd',color:CHR.tick,font:{size:11}},min:0.24,max:0.36}}}),
    plugins:[diagPlugin(0.24,0.36)],
  });
}

/* EXPAND — works for both .see-more-btn and .expand-btn */
function toggleDetail(id,btn){
  const el=document.getElementById(id),open=el.classList.toggle('open');
  const span=btn.querySelector('span'),svg=btn.querySelector('svg');
  if(span)span.textContent=open?'See less':'See more';
  if(svg)svg.style.transform=open?'rotate(180deg)':'';
}

/* URL SHARING */
function shareCalcState(){
  const inp=getInp();
  const params=new URLSearchParams({rlbf:inp[0],rhnw:inp[1],rhfb:inp[2],st:inp[3],cl:inp[4],bta:inp[5],fa:inp[6],model:activeModel});
  const url=window.location.href.split('?')[0]+'?'+params.toString();
  navigator.clipboard.writeText(url).then(()=>{
    const btn=document.getElementById('share-btn-text');
    btn.textContent='Link copied'; setTimeout(()=>btn.textContent='Copy shareable link',2500);
  }).catch(()=>prompt('Copy this link:',url));
}

function loadStateFromURL(){
  const p=new URLSearchParams(window.location.search);
  const map={rlbf:'s-rlbf',rhnw:'s-rhnw',rhfb:'s-rhfb',st:'s-sidetaper',cl:'s-clearance',bta:'s-bottomtaper',fa:'s-frontalarea'};
  let loaded=false;
  for(const[k,sid]of Object.entries(map)){const v=p.get(k);if(v!==null){const el=document.getElementById(sid);if(el){el.value=v;loaded=true;}}}
  if(loaded){
    document.querySelectorAll('input[type=range]').forEach(updateFill);
    updateCalc();
  }
  const m=p.get('model');
  if(m){const btn=document.getElementById('sel-'+m);if(btn)setModel(m,btn);}
}

/* INIT */
document.addEventListener('DOMContentLoaded',()=>{
  setGreeting(); setupSliders(); updateCalc(); initOverviewCharts(); loadStateFromURL();
  ['gp','gb','rf','nn','nn2'].forEach(r=>document.getElementById('row-'+r).classList.add('pred-active'));
  /* Ensure the default Best button is marked active in both naming conventions */
  const defBtn=document.getElementById('sel-all');
  if(defBtn){defBtn.classList.add('active-seg');defBtn.classList.add('seg-active');}
  /* Initialise all slider fills */
  document.querySelectorAll('input[type=range]').forEach(updateFill);
});
