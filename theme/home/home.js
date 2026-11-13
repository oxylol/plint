/* Plint home: draws the page in theme/home/index.html. Read home.css first for the language. */
(async()=>{
const root=document.documentElement, $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];
const calm=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
const live=()=>root.dataset.flk==='on'&&!calm();   // may things move?
const rnd=(a,b)=>a+Math.random()*(b-a), pick=a=>a[Math.floor(Math.random()*a.length)];
const seen=e=>{const r=e.getBoundingClientRect();return r.bottom>0&&r.top<innerHeight};

/* ── pixel art: shapes drawn as vectors, then sampled onto a grid of squares ── */
const ST=[[6,54,122,82,8],[20,30,98,48,6],[44,8,106,24,5]];   // three pieces: the plinth first, then the two stones upward: x0,y0,x1,y1,cut
const ALL=ST.map((_,i)=>i), STONES=ALL.slice(1);
const hex=([x0,y0,x1,y1,c])=>[[x0+c,y0],[x1,y0],[x1,y1-c],[x1-c,y1],[x0,y1],[x0,y0+c]];
const STROKES=[[[[5,40],[5,5],[29,5],[29,25],[5,25]]],[[[51,0],[51,35],[80,35]]],[[[97,0],[97,40]]],[[[119,40],[119,4],[143,36],[143,0]]],[[[160,5],[202,5]],[[181,5],[181,40]]]];
const ART={
  mark:{W:128,H:88,draw(c,a={}){const off=a.off||{},g=v=>Math.round(v/4)*4;
    (a.show||ALL).forEach(i=>{const [dx,dy]=off[i]||[0,0];c.beginPath();hex(ST[i]).forEach(([x,y],j)=>c[j?'lineTo':'moveTo'](x+dx,y+dy));c.closePath();c.fill()});
    if(a.dust!=null){const [x0,,x1,y1]=ST[a.dust];[[x0-8,y1-4],[x0-16,y1-12],[x1+4,y1-4],[x1+12,y1-10]].forEach(([x,y])=>c.fillRect(g(x),g(y),4,4))}}},
  build:{W:128,H:128,draw(c,a){c.translate(0,40);ART.mark.draw(c,a)}},   // the mark with headroom for falling stones
  word:{W:202,H:40,draw(c,vis=[1,1,1,1,1]){c.lineWidth=10;c.lineJoin='bevel';STROKES.forEach((l,i)=>vis[i]&&l.forEach(s=>{c.beginPath();s.forEach(([x,y],j)=>c[j?'lineTo':'moveTo'](x,y));c.stroke()}))}}
};
const inkOf=k=>k==='bg'?getComputedStyle(root).getPropertyValue('--bg'):`rgb(${getComputedStyle(root).getPropertyValue('--ph')})`;
function pix(cv,art,arg,cell,gap,ink,keep){
  const A=ART[art], dpr=Math.min(window.devicePixelRatio||1,3), w=cv.clientWidth; if(!w)return;
  cv.style.aspectRatio=A.W+'/'+A.H;
  const W=Math.round(w*dpr), H=Math.round(w*dpr*A.H/A.W);
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H}
  const ctx=cv.getContext('2d'); if(!keep)ctx.clearRect(0,0,W,H); if(arg===null)return;
  const cols=Math.ceil(A.W/cell), rows=Math.ceil(A.H/cell), S=3, off=document.createElement('canvas');
  off.width=cols*S; off.height=rows*S; const o=off.getContext('2d'); o.fillStyle=o.strokeStyle='#fff'; o.scale(S/cell,S/cell); A.draw(o,arg);
  const d=o.getImageData(0,0,off.width,off.height).data, s=W/A.W, g=gap?Math.max(1,Math.round(cell*s*gap)):0;
  ctx.fillStyle=inkOf(ink);
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){let a=0;
    for(let y=0;y<S;y++)for(let x=0;x<S;x++)a+=d[((r*S+y)*off.width+c*S+x)*4+3];
    if(a>S*S*127){const x0=Math.round(c*cell*s),x1=Math.round((c+1)*cell*s),y0=Math.round(r*cell*s),y1=Math.round((r+1)*cell*s);ctx.fillRect(x0,y0,x1-x0-g,y1-y0-g)}}
}
// the finished mark: the plinth solid, the stones dot matrix. vis = which parts are lit, sharp = which have set (the rest are coarse
// and solid), flip = parts drawn the other way round for a moment (a stone goes solid, the plinth dotted: it is being read)
function mk(cv,art,gap,{ink,vis=ALL,sharp=ALL,flip=[]}={}){
  const S=f=>({show:ALL.filter(i=>vis.includes(i)&&f(i))}), sh=i=>sharp.includes(i), dot=i=>!!i!==flip.includes(i);
  pix(cv,art,S(i=>sh(i)&&dot(i)),2,gap,ink);
  pix(cv,art,S(i=>sh(i)&&!dot(i)),2,0,ink,true);
  pix(cv,art,S(i=>!sh(i)),4,0,ink,true);
}
const B4=[0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5];
function dither(cv,ph){   // an ordered-dither fade, the ground under the cairn; with a phase it drifts like mist
  const dpr=Math.min(window.devicePixelRatio||1,3), w=cv.clientWidth, rows=+cv.dataset.rows, c=3, cols=Math.floor(w/c); if(!w)return;
  const W=Math.round(w*dpr), H=Math.round(rows*c*dpr);
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;cv.style.height=rows*c+'px'}
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H); ctx.fillStyle=inkOf();
  for(let r=0;r<rows;r++)for(let x=0;x<cols;x++){const d=(cv.dataset.flip?(r+1)/rows:1-r/rows)*.92*(ph==null?1:1+.24*Math.sin(x*.09+r*.5-ph));
    if(d>(B4[(r%4)*4+x%4]+.5)/16)ctx.fillRect(Math.round(x*c*dpr),Math.round(r*c*dpr),Math.round((c-1)*dpr),Math.round((c-1)*dpr))}
}
// a static canvas in its current state: _vis = lit parts (unset: all), _flip as in mk()
function paint(cv){const d=cv.dataset, v=cv._vis;
  if(d.art==='word')return pix(cv,'word',v?[0,1,2,3,4].map(i=>+v.includes(i)):undefined,+d.cell,+d.gap,d.ink);
  if(+d.gap&&!d.arg)return mk(cv,d.art,+d.gap,{ink:d.ink,vis:v||ALL,flip:cv._flip||[]});
  pix(cv,d.art,d.arg?JSON.parse(d.arg):v?{show:v}:undefined,+d.cell,+d.gap,d.ink)}
const statics=()=>{$$('canvas.pix[data-art]').forEach(paint);$$('canvas.dith').forEach(c=>dither(c))};

/* ── lights: each one strikes on by itself ── */
// one light striking: on, a random stutter or two, then steady. [ms, on?]
const stutter=()=>{const ev=[[0,1]];let t=0;for(let k=Math.floor(rnd(1,3.5));k>0;k--){t+=rnd(30,85);ev.push([t,0]);t+=rnd(25,95);ev.push([t,1])}return ev};
function strike(el,delay=0){
  el.classList.add('on'); if(!live()||!el.animate)return;
  const ev=stutter(), T=ev[ev.length-1][0]+40;
  el.animate([{opacity:0,offset:0,easing:'step-end'},...ev.map(([t,on],i)=>({opacity:on?(i<ev.length-1?rnd(.45,1):1):rnd(0,.16),offset:(t+20)/T,easing:'step-end'}))],{duration:T,delay,fill:'backwards'});
}
function parts(cv,n,ordered,tm=setTimeout){   // a canvas made of parts (stones, letters): each part strikes by itself
  if(!live()){cv._vis=null;return paint(cv)}
  cv._vis=[];paint(cv);let t=0;
  const order=[...Array(n).keys()]; if(!ordered)order.sort(()=>Math.random()-.5);
  order.forEach(i=>{stutter().forEach(([dt,on])=>tm(()=>{cv._vis=(cv._vis||[]).filter(j=>j!==i);if(on)cv._vis.push(i);paint(cv)},t+dt));t+=rnd(60,140)});
  tm(()=>{cv._vis=null;paint(cv)},t+700);
}
const spin=(s,ms)=>{if(!live()||!/\d/.test(s.dataset.v))return;const id=setInterval(()=>{s.textContent=Math.floor(Math.random()*10)},50);setTimeout(()=>{clearInterval(id);s.textContent=s.dataset.v},ms)};
const blip=el=>el.animate&&el.animate([{opacity:.12},{opacity:.12}],{duration:90});
const flash=el=>live()&&el.animate&&el.animate([{opacity:.2,easing:'step-end'},{opacity:1,offset:.35,easing:'step-end'},{opacity:.2,offset:.6,easing:'step-end'},{opacity:1,offset:.8}],{duration:170});

/* ── the cairn going up ── */
// on any canvas: the plinth strikes coarse and sets down, then each stone drops in steps, lands a little off with dust and is nudged home
function stack(cv,art,at,t,fl,each){
  const D=(cell,arg,keep)=>pix(cv,art,arg,cell,0,undefined,keep), placed=[];
  const frame=(i,off,cell,dust)=>{D(4,{show:placed.slice(),dust});if(i!=null)D(cell,{show:[i],off:{[i]:off}},true)};   // placed parts fine, the one in the air coarse
  at(t,()=>D(13,{show:[0]})); if(fl)at(t+70,()=>D(2,null)); at(t+130,()=>D(13,{show:[0]})); at(t+260,()=>D(8,{show:[0]})); at(t+350,()=>{placed.push(0);frame();each&&each(0)});
  t+=440;
  STONES.forEach(i=>{const dx=[0,10,-8,6][i];
    at(t,()=>frame(i,[dx,-44],8)); at(t+60,()=>frame(i,[dx,-21],8)); at(t+115,()=>frame(i,[dx,-6],8));
    at(t+165,()=>frame(i,[dx,0],4,i)); at(t+240,()=>{placed.push(i);frame();each&&each(i)}); t+=270});
  return t;
}
// then it sets: part by part it goes dark for a frame and comes back sharp
function sharpen(cv,art,at,t,fl){const sharp=[];
  ALL.forEach(i=>{if(fl)at(t,()=>mk(cv,art,.2,{sharp,vis:ALL.filter(j=>j!==i)}));const u=t+rnd(40,75);at(u,()=>{sharp.push(i);mk(cv,art,.2,{sharp})});t=u+rnd(45,85)});
  return t}

/* ── 01 intro ── */
const LOG=()=>{const S=D.system, ar=S.disks.find(d=>d.name==='ARRAY'), z=n=>String(n).padStart(2,'0'), dn=mons().length-upN();   // the boot log reads the same figures
 return ['PLINT SYSTEM INITIALIZATION (POST) // BUILD 2026.11','INITIALIZING HOME SERVER CONTROLLER…','',
 ['0x0000','0.000','Memory check… '+(S.mem.total_mb/1024).toFixed(1)+' GB OK'],['0x0008','0.004','Mapping array… '+(ar?gb(ar.total_gb)+' MOUNTED':'NOT FOUND')],['0x0014','0.009','Docker socket proxy… ONLINE'],
 ['0x0021','0.014','Enumerating stacks… '+z(S.stacks)+' FOUND'],['0x0030','0.021','Starting containers… '+z(S.containers.running)+' RUNNING'],['0x0042','0.027','Wildcard certificate… VALID'],
 ['0x0050','0.034','AdGuard DNS rewrite… LOCKED'],['0x0063','0.041','Tailnet… '+z(D.tailscale.devices.length)+' NODES'],['0x0071','0.047','Kopia manifest 03:30… VERIFIED'],
 ['0x0080','0.053','Uptime Kuma monitors… '+z(upN())+' UP'],['0x0091','0.064','AMP instance manager… '+z(D.games.servers.length)+' SERVERS'],['0x00A2','0.071','Hardware faults: 000'],
 ['0x00B0','0.083','Diagnostics complete. '+(dn?dn+' UNIT'+(dn>1?'S':'')+' DOWN':'ALL SYSTEMS NOMINAL')],['0x00C2','0.096','TRANSFERRING EXECUTION TO HOME SCREEN…']]};
const stage=$('#stage'), brand=$('#brand'), mark=$('#mark'), word=$('#word'), log=$('#log');
let timers=[], done=false;
const at=(t,f)=>timers.push(setTimeout(f,t));
const Wd=v=>pix(word,'word',v,2,.2);
function finish(){timers.forEach(clearTimeout);timers=[];done=true;brand.classList.add('up');stage.classList.add('end');mk(mark,'build',.2);Wd([1,1,1,1,1]);$$('p',log).forEach(p=>p.classList.add('on'))}
function intro(){
  timers.forEach(clearTimeout);timers=[];done=false;
  log.innerHTML=LOG().map(l=>typeof l==='string'?`<p>${l||'&nbsp;'}</p>`:`<p><span class="a">${l[0]} [${l[1]}]</span> ${l[2]}</p>`).join('')+'<p><i class="cursor"></i></p>';
  brand.classList.remove('up');stage.classList.remove('lit','end');pix(mark,'build',null,2,0);Wd(null);
  const fl=root.dataset.flk==='on';
  let t=stack(mark,'build',at,180,fl);
  t=sharpen(mark,'build',at,t+60,fl); at(t+30,()=>stage.classList.add('lit'));   // set, and one surge of light
  at(t+250,()=>brand.classList.add('up'));
  // then the name: each letter is its own light, out of order
  const vis=[0,0,0,0,0]; let w=t+420;
  [0,3,2,4,1].forEach(i=>{(fl?stutter():[[0,1]]).forEach(([dt,on])=>at(w+dt,()=>{vis[i]=on;Wd(vis)}));w+=rnd(105,155)});
  // the log line by line, the labels last
  w+=260; $$('p',log).forEach(p=>{at(w,()=>strike(p));w+=36});
  at(w+120,()=>stage.classList.add('end'));
  at(w+200,()=>{done=true});
  at(w+950,leave);
}
// the intro is shown once per visit; a tap skips it. Then it switches off like a tube and the home page comes on
let left=false;
function leave(){if(left)return;left=true;timers.forEach(clearTimeout);timers=[];done=true;try{sessionStorage.plintBoot=1}catch(e){}
  const off=()=>{stage.hidden=true;root.classList.remove('boot');lights()};
  if(!live()||stage.hidden)return off();
  stage.classList.add('out');setTimeout(off,300)}
stage.addEventListener('click',leave);

// the figures: what the collector on the server wrote (scripts/theme-data.sh, every 2 minutes), read before anything is drawn
// and again every minute. A file that cannot be read leaves its instrument empty, not the page broken
const SNAP={system:{generated:new Date().toISOString(),uptime_s:0,load:[0,0,0],cores:[0,0,0,0],peak:[0,0,0,0],loop:{at:'',frames:[]},mem:{total_mb:1,used_mb:0,tenants:[]},disks:[],containers:{running:0,total:0},stacks:0},
  dns:{hours:{},days:{queries:[],blocked:[],avg_ms:0}},uptime:{days:[],monitors:[]},tailscale:{devices:[]},storage:{generated:new Date().toISOString(),disks:[]},
  games:{servers:[]},'games-history':{samples:[]},speedtest:{results:[]},backups:{snapshots:[]}};
let READ=0;
await Promise.all(Object.keys(SNAP).map(async f=>{try{const r=await fetch('/_plint/data/'+f+'.json',{cache:'no-store'});if(r.ok){SNAP[f]=await r.json();READ++}}catch(e){}}));
if(READ)root.dataset.src='live';
const D=SNAP, FILES=Object.keys(SNAP), DRAW=[], SIG={}, http=location.protocol.startsWith('http'), stills=[];
const sig=o=>JSON.stringify(Object.assign({},o,{generated:0}));
FILES.forEach(f=>SIG[f]=sig(D[f]));
const on=(files,fn)=>{DRAW.push([files.split(' '),fn]);try{fn()}catch(e){console.error(e)}};   // draw now, and again whenever one of these files changes
async function pull(){
  if(!http||document.hidden)return; const ch=[]; let ok=0;
  await Promise.all(FILES.map(async f=>{try{const r=await fetch('/_plint/data/'+f+'.json',{cache:'no-store'});if(!r.ok)return;
    const o=await r.json(),s=sig(o);ok++;D[f].generated=o.generated;if(s===SIG[f])return;SIG[f]=s;D[f]=o;ch.push(f)}catch(e){}}));
  if(ok)root.dataset.src='live';
  DRAW.forEach(([fs,fn])=>{if(fs.some(f=>ch.includes(f)))try{fn()}catch(e){console.error(e)}});
}
const p2=n=>String(n).padStart(2,'0'), MON=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'], DAY=['SUN','MON','TUE','WED','THU','FRI','SAT'];
const iso=s=>new Date(String(s).replace(/\.\d+/,''));   // the collector's timestamps, without the fraction some browsers refuse
const esc=s=>String(s).replace(/[&<>"]/g,c=>'&#'+c.charCodeAt(0)+';');
const dm=d=>p2(d.getDate())+' '+MON[d.getMonth()], hm=d=>p2(d.getHours())+':'+p2(d.getMinutes()), ymd=d=>d.getFullYear()+'-'+p2(d.getMonth()+1)+'-'+p2(d.getDate());
const nu=n=>n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e4?(n/1e3).toFixed(1)+'K':String(Math.round(n));
const gb=v=>v>=1000?(v/1000).toFixed(1)+' TB':v>=100?Math.round(v)+' GB':(+v).toFixed(1)+' GB';
const pct=(a,b)=>b?Math.round(a/b*100):0;
const url=sub=>http?'https://'+sub+'.'+location.hostname.replace(/^dash\./,''):'#';
const AR='<svg class="ar" viewBox="0 0 7 7" aria-hidden="true"><path d="M1 6L6 1M2.5 1H6v3.5"/></svg>';
// a dot numeral: every digit is its own light, and spins when the figure changes
const dig=(e,v,sub)=>{v=String(v);const k=v+'|'+(sub||'');if(e.dataset.k===k)return;const first=!e.dataset.k;e.dataset.k=k;
  e.innerHTML=[...v].map(c=>`<span class="dg" data-v="${c}">${c}</span>`).join('')+(sub?`<sub>${sub}</sub>`:'');
  if(!first&&live()&&seen(e))$$('.dg',e).forEach(s=>spin(s,rnd(160,420)))};
// figures used in more than one place
const mons=()=>D.uptime.monitors, upN=()=>mons().filter(m=>m.up).length;
const mon=n=>mons().find(m=>m.name===n), runs=n=>D.system.mem.tenants.some(t=>t.kind==='container'&&t.name===n);
const cpuMean=()=>Math.round(D.system.cores.reduce((a,b)=>a+b,0)/D.system.cores.length);
const lastTest=()=>D.speedtest.results.filter(r=>r.ok).slice(-1)[0];
function dns24(){   // the last 24 local hours of DNS, by hour of the day. The hour that is running now is not complete
  const hs=D.dns.hours, last=Object.keys(hs).sort().pop(), H=Array(24).fill(null); if(!last)return {H,now:new Date().getHours(),q:0,b:0};
  const end=new Date(+last.slice(0,4),+last.slice(5,7)-1,+last.slice(8,10),+last.slice(11,13)); let q=0,b=0;
  for(let k=0;k<24;k++){const d=new Date(end-k*3600e3),v=hs[ymd(d)+'T'+p2(d.getHours())];if(v){H[d.getHours()]={q:v[0],b:v[1]};q+=v[0];b+=v[1]}}
  return {H,now:end.getHours(),q,b};
}
function nights(){   // backups by the night they ran. Kopia keeps every night for two weeks and thins the older ones
  const by={}; D.backups.snapshots.forEach(s=>{const at=iso(s.start),k=ymd(at),n=by[k]||(by[k]={n:0,size:0,err:0,at});n.n++;n.size+=s.size;n.err+=s.errors});
  const keys=Object.keys(by).sort(), t=new Date(), out=[]; if(!by[ymd(t)])t.setDate(t.getDate()-1);
  for(let k=13;k>=0;k--){const d=new Date(t.getFullYear(),t.getMonth(),t.getDate()-k);out.push({d,r:by[ymd(d)]})}
  return {out,oldest:keys.length?by[keys[0]].at:null,last:keys.length?by[keys[keys.length-1]]:null};
}

// the registry: every unit, a tap away. [name under the domain, port, title, what it is (or a live figure), lamp: a monitor in Uptime Kuma or c:<container>]
const REG=[['EVERYDAY'],['games',8080,'Games',()=>{const s=D.games.servers,r=s.filter(x=>x.running);return r.length?r.length+' running · '+r.reduce((a,x)=>a+x.players,0)+' players':'AMP · '+s.length+' servers, all stopped'},'Games'],
 ['table',3200,'D&D Table','campaigns, dice, DM screen','c:table'],['ai',5220,'Plint Chat','Claude Code from the phone',''],['cnc',5178,'CNC Studio','G-code simulator','c:cnc-studio'],['twin',5210,'CNC Twin','physics twin','c:cnc-twin'],
 ['poly',5190,'Polybot','Polymarket, paper mode','c:polybot'],['music',5200,'Hertz','music from waves','c:hertz'],['accrete',5240,'Accrete','a star from dust','c:accrete'],['stremio',8095,'Stremio','movies & series','Stremio'],['tv',7700,'TV','the Mac on the TV',''],
 ['HEALTH'],['status',3001,'Status',()=>'Uptime Kuma · '+upN()+' of '+mons().length+' up','c:uptime-kuma'],['dns',3000,'AdGuard',()=>{const d=dns24();return (d.q?(d.b/d.q*100).toFixed(1):'0')+' % blocked, last 24 h'},'DNS'],
 ['backup',51515,'Backups',()=>{const n=nights().last;return 'Kopia · last '+(n?dm(n.at)+' '+hm(n.at):'none')},'Backups'],['speed',8075,'Internet',()=>{const r=lastTest();return r?Math.round(r.down)+' down · '+Math.round(r.up)+' up':'no test yet'},'c:speedtest-tracker'],
 ['ntfy',8070,'Notifications','ntfy · messages to the phone','Notifications'],
 ['SYSTEM'],['host',9090,'Host','Cockpit · the machine','Host'],['stacks',5001,'Stacks',()=>'Dockge · '+D.system.stacks+' stacks','Stacks'],['metrics',8090,'Metrics','Beszel · history','Metrics']];
$('#reg').innerHTML='<div class="rg">'+REG.map((r,i)=>r.length===1?`${i?'</div><div class="rg">':''}<h4>${r[0]}</h4>`:
 `<a href="${url(r[0])}" data-sub="${r[0]}"><span class="p">:${r[1]}</span><span class="n"><b>${r[2]}</b><small></small></span><span class="s"><span class="blk"></span><span class="st"></span></span></a>`).join('')+'</div>';
on('uptime system dns speedtest backups games',()=>REG.forEach(r=>{if(r.length===1)return;const a=$(`#reg a[data-sub="${r[0]}"]`),k=r[4];
  const up=!k?null:k.startsWith('c:')?runs(k.slice(2)):!!(mon(k)||{}).up;
  $('small',a).textContent=typeof r[3]==='function'?r[3]():r[3];
  $('.s',a).classList.toggle('off',!up);$('.st',a).textContent=up==null?'NOT WATCHED':up?'ONLINE':'OFFLINE';a.classList.toggle('lit',!!up)}));
$$('[data-go]').forEach(a=>{a.href=url(a.dataset.go);a.insertAdjacentHTML('beforeend',AR)});

// the ticker
on('system uptime games speedtest dns tailscale storage',()=>{const dn=mons().filter(m=>!m.up), run=D.games.servers.filter(s=>s.running), d=dns24(), t=lastTest(), ts=D.tailscale.devices, ds=D.system.disks;
  const u=(a,b)=>pct(a,b)+' %', ru=s=>`<span class="ru">${s}</span>`;
  const T=[dn.length?`<span class="inv">${dn.length} DOWN: ${esc(dn.map(m=>m.name).join(', ').toUpperCase())}</span>`:'ALL SYSTEMS NOMINAL',dn.length?'':ru('ВСЕ СИСТЕМЫ В НОРМЕ'),
    upN()+' OF '+mons().length+' UNITS ONLINE',...(run.length?run.map(s=>esc(s.name.toUpperCase())+' · '+s.players+' PLAYERS'):['GAME SERVERS IDLE']),'NEXT BACKUP 03:30',
    t?Math.round(t.down)+' MBIT DOWN':'',...ds.map(x=>x.name+' '+u(x.used_gb,x.total_gb)),ru('ПАМЯТЬ '+u(D.system.mem.used_mb,D.system.mem.total_mb)),'MEMORY '+u(D.system.mem.used_mb,D.system.mem.total_mb),
    nu(d.q)+' DNS LOOKUPS / 24 H',ts.filter(x=>x.online).length+' OF '+ts.length+' TAILNET NODES'].filter(Boolean);
  const h=T.map(t=>t.startsWith('<span class="ru"')?t+'<i class="blk ru"></i>':`<span>${t}</span><i class="blk"></i>`).join('').repeat(2), el=$('#tick');
  if(el._h!==h){el._h=h;el.innerHTML=h}});
$('#bar').innerHTML=[3,1,1,2,1,3,2,1,1,1,3,2,1,2,1,1,3,1,2,3,1,1,2,1,1,3,1,2,2,1,3,1,1].map((w,i)=>`<i style="width:${w*1.5}px${i%2?';background:none':''}"></i>`).join('');

// the three big figures, and the same ones inside the apps
on('system uptime dns',()=>{const d=dns24(), n=mons().length, rec=mons().reduce((a,m)=>Math.max(a,m.down.filter(v=>v!=null).length),0);
  dig($('#bUnits'),upN(),'/'+n); dig($('#bCpu'),cpuMean(),'%'); dig($('#bUp'),Math.floor(D.system.uptime_s/86400),'D')});
// where the figures come from, and how old they are
const srcAge=()=>{const el=$('#srcT'), g=iso(D.system.generated), age=Math.max(0,Math.floor((Date.now()-g)/1e3)), liveNow=root.dataset.src==='live', stale=liveNow&&age>420;
  const t=!liveNow?'NO DATA · THE COLLECTOR HAS NOT ANSWERED':(stale?'DATA STALE':'DATA LIVE')+' · READ '+hm(g)+':'+p2(g.getSeconds())+' · '+Math.floor(age/60)+':'+p2(age%60)+' AGO';
  if(el.textContent!==t)el.textContent=t; el.classList.toggle('inv',stale||!liveNow)};

/* ── the system cairn: each stone a layer of the server ── */
const LAYERS=['01 PLINTH','02 DOCKER','03 UNITS'];
const cairn=$('#cairn'), cm=$('canvas.m',cairn);
cairn.insertAdjacentHTML('beforeend',LAYERS.map((l,i)=>{const [,y0,x1,y1]=ST[i];
  return `<div class="row" style="left:calc(var(--w)*${(x1/128).toFixed(3)} + 7px);top:calc(var(--w)*${((y0+y1)/2/128).toFixed(3)})"><b>${l}</b><span></span></div>`}).join(''));
const rows=$$('.row',cairn);
on('system uptime',()=>{const n=mons().length, u=upN(), v=['HOST · '+Math.floor(D.system.uptime_s/86400)+' D',D.system.containers.running+' CONTAINERS',u+' / '+n];
  rows.forEach((r,i)=>{const e=r.lastElementChild;if(e.textContent!==v[i])e.textContent=v[i];e.classList.toggle('inv',i===2&&u<n)})});   // inverse only when a unit is down
let ct=[]; const cat=(f,t)=>ct.push(setTimeout(f,t)), cstop=()=>{ct.forEach(clearTimeout);ct=[];cm._flip=[]};
function raise(){   // first light: the plinth, then each stone, strikes on its own and brings its callout
  if(!live()){rows.forEach(r=>r.classList.add('on'));cm._vis=null;return paint(cm)}
  cstop();cm._busy=true;parts(cm,ALL.length,true,cat);
  rows.forEach((r,i)=>cat(()=>strike(r),170+i*190+rnd(0,90)));
  cat(()=>{cm._busy=false},1700);
}
function rebuild(){   // a tap: it comes down and is stacked again
  if(!live()||!cairn.classList.contains('on'))return;
  cstop();cm._busy=true;cm._vis=null;rows.forEach(r=>r.classList.remove('on'));
  const a=(t,f)=>cat(f,t), t=sharpen(cm,'mark',a,stack(cm,'mark',a,0,true,i=>strike(rows[i]))+60,true);
  cat(()=>{cm._busy=false;paint(cm)},t+60);
}
cairn.addEventListener('click',rebuild);
function poll(){   // every few seconds the server is asked how it is: the cairn answers stone by stone, then every lamp in the registry
  if(!live()||document.hidden)return;
  if(!cm._busy&&cairn.classList.contains('on')&&seen(cairn)){
    ALL.forEach((i,k)=>cat(()=>{cm._flip=[i];paint(cm);blip(rows[i].lastElementChild)},k*140));
    cat(()=>{cm._flip=[];paint(cm)},ALL.length*140)}
  $$('#reg a.on.lit .blk').filter(seen).forEach((b,k)=>setTimeout(()=>blip(b),700+k*60));
}
setInterval(poll,9000);

/* ── alive: clock and seconds, the data read again every minute, mist, a faulty letter now and then ── */
const secs=$('#secs'); secs.innerHTML='<i></i>'.repeat(60); const sc=$$('i',secs);
const setT=(el,v)=>{if(el.textContent===v)return;if(!live()||!el.dataset.set){el.textContent=v;el.dataset.set=1;return}
  let k=0;const id=setInterval(()=>{el.textContent=++k>4?v:String(Math.floor(rnd(10,99)));if(k>4)clearInterval(id)},55)};
const clock=()=>{const d=new Date(),p=n=>String(n).padStart(2,'0'),s=d.getSeconds();setT($('#hh'),p(d.getHours()));setT($('#mm'),p(d.getMinutes()));
  $('#dt').textContent=d.toLocaleDateString('en-GB',{weekday:'short',day:'2-digit',month:'short'}).replace(',','').toUpperCase();
  sc.forEach((c,i)=>c.className=i<s?'p':i===s?'n':'');srcAge()};
clock();setInterval(clock,1000);
pull();setInterval(pull,60000);document.addEventListener('visibilitychange',()=>document.hidden||pull());
let mist=0; setInterval(()=>{if(!live()||document.hidden)return;mist+=.22;$$('canvas.dith').forEach(c=>seen(c)&&dither(c,mist))},170);
(function glitch(){setTimeout(()=>{const c=$('.mast canvas');
  if(live()&&!document.hidden&&c.classList.contains('on')&&!c._vis&&seen(c)){const i=Math.floor(rnd(0,5));c._vis=[0,1,2,3,4].filter(j=>j!==i);paint(c);setTimeout(()=>{c._vis=null;paint(c)},rnd(50,110))}
  glitch()},rnd(7000,15000))})();
document.addEventListener('click',e=>{const b=e.target.closest('.reg a,[data-go]');if(!b)return;flash(b);
  if(http||!b.matches('.reg a,[data-go]'))return;   // served by Plint these are real links; opened as a file they only select
  e.preventDefault();if(b.matches('.reg a')){$$('.reg a.sel').forEach(a=>a.classList.remove('sel'));b.classList.add('sel')}});

/* ── the instruments: the server's own figures, drawn. Each one draws itself again when its file changes ── */
const svg=(vb,inner)=>`<svg viewBox="${vb}" role="img">${inner}</svg>`;
const say=(ins,t)=>{const r=$('.rd',ins);r.textContent=t;flash(r)};
const awake=el=>live()&&!document.hidden&&seen(el)&&el.closest('.ins').classList.contains('on');
const hms=d=>hm(d)+':'+p2(d.getSeconds());
document.addEventListener('click',e=>{const m=e.target.closest('.ins [data-t]');if(m)say(m.closest('.ins'),m.dataset.t)});

// seats: one little person per player slot on every game server; a taken seat is lit, and its player hops or waves now and then
(()=>{const el=$('#seats'), ins=el.closest('.ins');
  const F=['..###..','..###..','.#####.','#.###.#','#.###.#','..###..','..#.#..','..#.#..','.##.##.'], W=['#.###.#','#.###.#','.#####.','..###..','..###..','..###..','..#.#..','..#.#..','.##.##.'];
  const sp=(id,rows)=>`<g id="${id}">`+rows.map((r,y)=>[...r].map((ch,x)=>ch==='#'?`<rect x="${x}" y="${y}" width="1.03" height="1.03"/>`:'').join('')).join('')+'</g>';
  el.insertAdjacentHTML('beforebegin',`<svg width="0" height="0" style="position:absolute"><defs>${sp('pf',F)}${sp('pw',W)}</defs></svg>`);
  on('games games-history',()=>{const S=D.games.servers, H=D['games-history'].samples, t0=Date.now()/1e3-86400;
    el.innerHTML=S.map(s=>{const sl=Array(48).fill(null), mx=Math.min(s.max||10,40), n=s.running?Math.min(s.players,mx):0;   // the last 24 h in half hours: null not known, -1 stopped, else players
      H.forEach(([t,m])=>{const k=Math.floor((t-t0)/1800),v=m[s.id];if(k>=0&&k<48&&v!=null)sl[k]=Math.max(sl[k]==null?-1:sl[k],v)});
      const ran=sl.filter(v=>v!=null&&v>=0).length, top=Math.max(0,...sl.map(v=>v||0));
      const t=`${s.name} · ${s.module} · ${s.running?s.players+' OF '+s.max+' SEATS TAKEN':'STOPPED'} · LAST 24 H: ${ran?'UP '+ran/2+' H, MOST '+top+' AT ONCE':'NOT RUN'}`;
      return `<div class="srv" data-t="${esc(t.toUpperCase())}"><span class="u${s.running?'':' faint'}">${s.running?'<span class="blk"></span>&nbsp; ':''}${esc(s.name)}</span><span${s.running?'':' class="faint"'}>${s.running?s.players+' / '+s.max:'STOPPED'}</span>`+
        (ran||s.running?svg(`0 -1.5 ${mx*9-2} 10.5`,Array.from({length:mx},(_,i)=>`<use href="#pf" x="${i*9}" class="${i<n?'k':'z'}"/>`).join('')):   // a server that has not run all day folds its seats away
          svg(`0 0 ${mx*9-2} 2`,Array.from({length:mx},(_,i)=>`<rect x="${i*9}" y="0" width="7" height="1.6" class="z"/>`).join('')))+
        (ran||s.running?`<div class="tape">${sl.map(v=>`<i class="${v>0?'pl':v===0?'on':''}"></i>`).join('')}</div>`:'')+'</div>'}).join('');
    const run=S.filter(s=>s.running);
    $('.rd',ins).textContent=`${run.reduce((a,s)=>a+s.players,0)} OF ${S.reduce((a,s)=>a+s.max,0)} SEATS TAKEN · ${run.length} OF ${S.length} SERVERS RUNNING`;
  });
  setInterval(()=>{if(!awake(el))return;const U=$$('use.k',el);if(!U.length)return;const u=pick(U);
    if(Math.random()<.5){u.setAttribute('y',-1.5);setTimeout(()=>u.setAttribute('y',0),150)}   // a hop
    else [0,300].forEach(t=>{setTimeout(()=>u.setAttribute('href','#pw'),t);setTimeout(()=>u.setAttribute('href','#pf'),t+170)})},2100);   // or a wave
  ins._lit=()=>$$('.srv',el).forEach((r,i)=>{strike(r,i*130+rnd(0,60));$$('use.k',r).forEach((u,j)=>strike(u,i*130+200+j*60))});
})();

// cores: a level meter per core, like a tape deck. The server records ten seconds of load every two minutes; the meters play that recording
// in a loop. The ghost segment above is the highest two-minute mean of the last hour
(()=>{const el=$('#cores'), ins=el.closest('.ins'), SEG=20; let V=[],T=[],f=0;
  const lvl=v=>v>0?Math.max(1,Math.round(v/100*SEG)):0;
  const step=()=>{const S=D.system, fr=S.loop.frames, i=f%fr.length, vals=live()?fr[i]:S.cores;
    V.forEach((g,c)=>{const L=lvl(vals[c]),P=lvl(S.peak[c])-1;$$('rect[data-k]',g).forEach((r,k)=>r.setAttribute('class',k<L?'k':k===P?'pk':'z'))});
    T.forEach((r,k)=>r.setAttribute('class',live()&&k===i?'k':'z'))};
  on('system',()=>{const S=D.system, n=S.cores.length, w=176/n, fr=S.loop.frames; let s='';
    for(let c=0;c<n;c++){s+=`<g class="vu" data-c="${c}"><rect class="hit" x="${c*w}" y="0" width="${w}" height="96"/>`;
      for(let k=0;k<SEG;k++)s+=`<rect data-k="${k}" x="${c*w+4}" y="${((SEG-1-k)*4.2).toFixed(1)}" width="${w-8}" height="3"/>`;
      s+=`<text x="${c*w+w/2}" y="93" text-anchor="middle">${n>6?c:'CORE '+c}</text></g>`}
    s+='<g class="tp">'+fr.map((_,i)=>`<rect x="${4+i*7}" y="101.5" width="5" height="4"/>`).join('')+`</g><text x="172" y="105.5" text-anchor="end">REC ${hms(iso(S.loop.at))} · ${fr.length} S</text>`;
    el.innerHTML=svg('0 0 176 108',s); V=$$('.vu',el); T=$$('.tp rect',el); f=0; step();
    const pk=Math.max(...S.peak);
    $('.rd',ins).textContent=`MEAN ${cpuMean()} % · 1 H PEAK ${pk} % ON CORE ${S.peak.indexOf(pk)} · LOAD ${S.load[0]}`;
  });
  setInterval(()=>{if(!awake(el))return;f++;step()},700);
  stills.push(step);
  el.addEventListener('click',e=>{const g=e.target.closest('.vu');if(!g)return;const c=+g.dataset.c,S=D.system,fs=S.loop.frames.map(x=>x[c]);
    say(ins,`CORE ${c} · MEAN ${S.cores[c]} % · 1 H PEAK ${S.peak[c]} % · RECORDED ${Math.min(...fs)}–${Math.max(...fs)} %`)});
  ins._lit=()=>V.forEach((g,i)=>strike(g,i*90+rnd(0,40)));
})();

// memory map: 320 cells, filled in order like an old defragmenter; the three biggest tenants have a cell shape each
(()=>{const el=$('#mem'), ins=el.closest('.ins'); let U=[],hd=0;
  const sz=mb=>mb>=1000?(mb/1024).toFixed(1)+' GB':Math.round(mb)+' MB';
  on('system',()=>{const M=D.system.mem, cell=M.total_mb/320, T=M.tenants, top=T.slice(0,3), rest=Math.max(0,M.used_mb-top.reduce((a,t)=>a+t.mb,0)), free=M.total_mb-M.used_mb;
    const K=top.map((t,i)=>['abc'[i],t.name.toUpperCase(),t.mb]).concat([['d',Math.max(0,T.length-3)+' MORE + SYSTEM',rest]]), cells=[];
    K.forEach(k=>{k[3]=Math.max(1,Math.round(k[2]/cell));for(let i=0;i<k[3]&&cells.length<320;i++)cells.push(k)});
    const nf=320-cells.length, ft=`FREE + CACHE · ${sz(free)} · ${nf} CELLS`;
    let s='<defs><g id="ma"><rect width="9" height="9"/></g><g id="mb"><path d="M0 0h9v9h-9zM3 3v3h3v-3z" fill-rule="evenodd"/></g><g id="mc"><rect width="4" height="4"/><rect x="5" y="5" width="4" height="4"/></g><g id="md"><rect x="2.5" y="2.5" width="4" height="4"/></g></defs>';
    for(let r=0;r<10;r++){s+='<g class="mr">';for(let c=0;c<32;c++){const k=cells[r*32+c],x=c*11,y=r*11;
      s+=k?`<use href="#m${k[0]}" class="k" x="${x}" y="${y}" data-t="${esc(k[1])} · ${sz(k[2])} · ${k[3]} CELLS"/>`:`<rect class="z" x="${x}" y="${y}" width="9" height="9" data-t="${ft}"/>`}s+='</g>'}
    el.innerHTML=svg('0 0 350 108',s)+'<ul class="lg">'+K.map(k=>`<li><svg viewBox="0 0 9 9"><use href="#m${k[0]}" class="k"/></svg>${esc(k[1])} <b>${sz(k[2])}</b></li>`).join('')+`<li><svg viewBox="0 0 9 9"><rect class="z" width="9" height="9"/></svg>FREE + CACHE <b>${sz(free)}</b></li></ul>`;
    U=$$('.mr use',el);
    $('#memH').textContent=(M.used_mb/1024).toFixed(1)+' / '+(M.total_mb/1024).toFixed(1)+' GB';
    $('.rd',ins).textContent=`ONE CELL = ${Math.round(cell)} MB · ${T.length} TENANTS`;
  });
  setInterval(()=>{if(!awake(el)||!U.length)return;hd=(hd+Math.ceil(rnd(0,4)))%U.length;blip(U[hd])},130);   // the head reading its way through
  ins._lit=()=>$$('.mr',el).forEach((g,i)=>strike(g,i*70+rnd(0,40)));
})();

// core sample: each disk drilled out and stood up, on its own scale. Every kind of data is a layer with its own grain; measured every 6 hours
(()=>{const el=$('#core'), ins=el.closest('.ins'), B=184, T=30;
  const TX={SYSTEM:'s',BACKUPS:'s','APP DATA':'p1',GAMES:'p1',DOCKER:'p2',MEDIA:'p2',HOME:'p3',OTHER:'p3',TMP:'p4'}, g1=v=>v>=100?Math.round(v):(+v).toFixed(1);
  on('storage',()=>{const S=D.storage.disks;
    let s='<defs><pattern id="p1" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="3" height="3"/></pattern>'+
      '<pattern id="p2" width="6" height="6" patternUnits="userSpaceOnUse"><rect y="4" width="2" height="2"/><rect x="2" y="2" width="2" height="2"/><rect x="4" width="2" height="2"/></pattern>'+
      '<pattern id="p3" width="6" height="6" patternUnits="userSpaceOnUse"><rect x="2" y="2" width="2" height="2"/></pattern>'+
      '<pattern id="p4" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="1.6"/></pattern></defs>';
    S.forEach((d,di)=>{const X=di*150+6, x1=X+40, tube=`${X+8},${T} ${x1},${T} ${x1},${B-8} ${x1-8},${B} ${X},${B} ${X},${T+8}`;
      s+=`<clipPath id="cc${di}"><polygon points="${tube}"/></clipPath><polygon class="tube" points="${tube}"/><text class="b" x="${X}" y="9">${esc(d.name)}</text><text x="${X}" y="19.5">${gb(d.used_gb)} OF ${gb(d.total_gb)}</text>`;
      let y=B, ly=B+6;
      d.layers.map(l=>[l.name,l.gb,TX[l.name]||'p3']).concat([['FREE',Math.max(0,d.total_gb-d.used_gb),'f']]).forEach(([n,v,p])=>{
        const h=p==='f'?y-T:Math.max(1.8,v/d.total_gb*(B-T)), y0=y-h, m=y0+h/2; ly=Math.min(ly-9.5,m);
        s+=`<g class="ly" data-t="${esc(d.name)} · ${esc(n)} · ${gb(v)} · ${pct(v,d.total_gb)} % OF THE DISK"><rect class="hit" x="${X}" y="${y0.toFixed(1)}" width="40" height="${h.toFixed(1)}"/><rect class="hit" x="${x1}" y="${(ly-5).toFixed(1)}" width="104" height="10"/>`+
          (p==='f'?'':`<rect clip-path="url(#cc${di})" x="${X}" y="${(y0+.8).toFixed(1)}" width="40" height="${(h-.8).toFixed(1)}" ${p==='s'?'class="k"':`fill="url(#${p})"`}/>`)+
          `<path class="ld" d="M${x1+3} ${m.toFixed(1)}h3L${x1+10} ${ly.toFixed(1)}h3"/><text x="${x1+16}" y="${(ly+2.6).toFixed(1)}"><tspan class="${p==='f'?'f':'b'}">${esc(n)}</tspan><tspan dx="5"${p==='f'?' class="f"':''}>${g1(v)}</tspan></text></g>`; y=y0})});
    el.innerHTML=svg('0 0 300 196',s);
    $('.rd',ins).textContent=S.map(d=>`${d.name} ${pct(d.used_gb,d.total_gb)} % FULL`).join(' · ')+' · MEASURED '+hm(iso(D.storage.generated));
  });
  ins._lit=()=>$$('.ly',el).forEach((g,i)=>strike(g,i*110+rnd(0,60)));   // deposited from the bottom up
})();

// day dial: 24 spokes, one per hour, 00 at the top; the longer the spoke, the more DNS that hour. A sweep goes round like a radar.
// Two series, never both at once: what was blocked, or everything that was asked
(()=>{const el=$('#dial'), key=$('#dkey'), ins=el.closest('.ins'); let G=[],ser='b',selH=null,dd=null,sw=0;
  const val=h=>dd.H[h]?dd.H[h][ser]:null;
  const sel=h=>{selH=h;G.forEach(g=>g.classList.toggle('sel',+g.dataset.h===h));const v=val(h),t=v==null?'—':nu(v),n=$('.dn',el);
    $('.dh',el).textContent=p2(h)+' H';$('.ds',el).textContent=h===dd.now?'SO FAR':'';n.textContent=t;n.style.fontSize=t.length>4?'16px':'21px'};
  const draw=()=>{dd=dns24(); const full=dd.H.map((x,h)=>x&&h!==dd.now?[x[ser],h]:null).filter(Boolean), mx=Math.max(1,...full.map(x=>x[0])); let s='';
    for(let h=0;h<24;h++){const v=val(h), n=v==null?0:Math.min(7,Math.max(1,Math.round(v/mx*7)));
      s+=`<g class="sp${h===dd.now?' now':''}" data-h="${h}" transform="rotate(${h*15+7.5})"><rect class="hit" x="-12" y="-98" width="24" height="72"/>`;
      for(let k=0;k<7;k++){const r=31+k*8.6,w=r*.2;s+=`<rect class="${k<n?'k':'z'}" x="${(-w/2).toFixed(1)}" y="${(-r-6).toFixed(1)}" width="${w.toFixed(1)}" height="6"/>`}
      s+='<rect class="nw" x="-1.5" y="-97.5" width="3" height="4"/></g>'}
    [['00',0,-101],['06',104,3],['12',0,108],['18',-104,3]].forEach(([t,x,y])=>s+=`<text x="${x}" y="${y}" text-anchor="middle">${t}</text>`);
    s+='<rect class="k" x="-2.5" y="-106.5" width="5" height="5" transform="rotate(52.5) rotate(45 0 -104)"/>';   // the 03:30 backup
    s+='<text class="dh" y="-8" text-anchor="middle"></text><text class="dn" y="11" text-anchor="middle"></text><text class="ds" y="19.5" text-anchor="middle"></text>';
    el.innerHTML=svg('-113 -113 226 226',s); G=$$('.sp',el); sel(selH==null?dd.now:selH);
    const pk=full.reduce((a,x)=>x[0]>a[0]?x:a,full[0]||[0,0]), qt=full.reduce((a,x)=>x[0]<a[0]?x:a,full[0]||[0,0]), rb=k=>`<i class="rb${ser===k?' on':''}"></i>`;
    key.innerHTML=`<li data-ser="b"><b>${rb('b')}BLOCKED</b>${nu(dd.b)} · ${dd.q?(dd.b/dd.q*100).toFixed(1):0} %</li><li data-ser="q"><b>${rb('q')}ALL LOOKUPS</b>${nu(dd.q)}</li>`+
      `<li><b>PEAK</b>${p2(pk[1])} H · ${nu(pk[0])}</li><li><b>QUIET</b>${p2(qt[1])} H · ${nu(qt[0])}</li><li><b>BACKUP</b><svg class="dmd" viewBox="0 0 8 8"><rect x="1.4" y="1.4" width="5.2" height="5.2" transform="rotate(45 4 4)"/></svg>03:30</li>`};
  on('dns',draw);
  el.addEventListener('click',e=>{const g=e.target.closest('.sp');if(g){sel(+g.dataset.h);flash($('.dn',el))}});
  key.addEventListener('click',e=>{const l=e.target.closest('[data-ser]');if(!l||l.dataset.ser===ser)return;ser=l.dataset.ser;draw();flash(key);if(live())G.forEach((g,i)=>strike(g,i*16+rnd(0,14)))});
  setInterval(()=>{if(!awake(el))return;G.forEach(g=>g.classList.remove('s0','s1','s2'));sw=(sw+1)%24;[0,1,2].forEach(k=>G[(sw-k+24)%24].classList.add('s'+k))},330);
  ins._lit=()=>G.forEach((g,i)=>strike(g,i*26+rnd(0,20)));
})();

// constellation: the tailnet as a star chart. Plint in the middle; the further out a node, the longer its ping (log scale).
// A node that does not answer sits hollow on the rim
(()=>{const el=$('#stars'), ins=el.closest('.ins'), cx=160, cy=100, rad=a=>a*Math.PI/180, R=ms=>34+24*Math.log10(Math.min(200,Math.max(1,ms))), SL=[-32,208,28,152,78,238,-78,118];
  let P=[],N=[],nd=[],busy=false,sd=7; const rs=()=>(sd=sd*16807%2147483647)/2147483647;
  let sky='<g class="sts">';
  for(let i=0;i<70;i++){const x=rs()*316+2,y=rs()*196+2,b=rs(),d=rs()*4; if(Math.hypot(x-cx,y-cy)<24)continue;
    sky+=`<rect class="st${b>.8?' tw':''}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${b>.6?1.6:1}" height="${b>.6?1.6:1}" style="animation-delay:-${d.toFixed(2)}s"/>`}
  sky+='<path class="tk" d="M3 7h8M7 3v8M309 7h8M313 3v8M3 193h8M7 189v8M309 193h8M313 189v8"/></g>';
  [[1,34],[10,58],[100,82]].forEach(([ms,r])=>sky+=`<g class="rg"><circle cx="${cx}" cy="${cy}" r="${r}"/><text x="${cx}" y="${cy-r-3}" text-anchor="middle">${ms} MS</text></g>`);
  sky+='<text x="16" y="9.5">DISTANCE = PING</text>';
  ST.forEach(p=>sky+=`<polygon class="k" points="${hex(p).map(([x,y])=>(cx-12.8+x*.2).toFixed(1)+','+(cy-8.8+y*.2).toFixed(1)).join(' ')}"/>`);
  on('tailscale',()=>{N=D.tailscale.devices.filter(d=>!d.self).map((d,i)=>{const a=SL[i%SL.length]+(i>=SL.length?13:0), ok=d.online&&d.ms!=null, r=ok?R(d.ms):92;
      const nm=d.name.toUpperCase(), seen_=d.lastSeen?iso(d.lastSeen):null;
      return {a,ok,off:!d.online,x:cx+r*Math.cos(rad(a)),y:cy+r*Math.sin(rad(a)),name:nm.length>13?nm.slice(0,12)+'…':nm,full:nm,os:(d.os||'').toUpperCase(),
        sub:ok?(d.ms<1?'<1 MS':Math.round(d.ms)+' MS'):d.online?'NO PING':'OFF'+(seen_?' · '+Math.floor((Date.now()-seen_)/864e5)+' D':''),
        tip:ok?(d.direct?'DIRECT':'VIA RELAY'):d.online?'ONLINE, DID NOT ANSWER THE PING':seen_?'LAST SEEN '+dm(seen_):'OFFLINE'}});
    let s=sky;
    N.forEach(n=>{const L=Math.cos(rad(n.a))<0,tx=(n.x+(L?-7:7)).toFixed(1),an=L?'end':'start';
      s+=`<g class="nd${n.off?' off':''}" data-t="${esc(n.full+(n.os?' · '+n.os:'')+(n.ok?' · '+n.sub:'')+' · '+n.tip)}"><line class="ln" x1="${(cx+20*Math.cos(rad(n.a))).toFixed(1)}" y1="${(cy+20*Math.sin(rad(n.a))).toFixed(1)}" x2="${n.x.toFixed(1)}" y2="${n.y.toFixed(1)}"/>`+
        `<rect class="hit" x="${(n.x-(L?66:10)).toFixed(1)}" y="${(n.y-12).toFixed(1)}" width="76" height="24"/><rect class="nb" x="${(n.x-3).toFixed(1)}" y="${(n.y-3).toFixed(1)}" width="6" height="6"/>`+
        `<text class="b" x="${tx}" y="${(n.y-.5).toFixed(1)}" text-anchor="${an}">${esc(n.name)}</text><text x="${tx}" y="${(n.y+7.5).toFixed(1)}" text-anchor="${an}">${esc(n.sub)}</text></g>`});
    s+='<rect class="k pkt" width="3.4" height="3.4" style="display:none"/>';
    el.innerHTML=svg('0 0 320 200',s); nd=$$('.nd',el); busy=false;
    $('.rd',ins).textContent=`${N.filter(n=>n.ok).length} OF ${N.length} NODES ANSWER`;
  });
  const trip=i=>{busy=true;const pk=$('.pkt',el),n=N[i],x0=cx+20*Math.cos(rad(n.a)),y0=cy+20*Math.sin(rad(n.a)),st=6;let k=0;   // a packet goes out, the node answers, it comes back
    const id=setInterval(()=>{if(!pk.isConnected){clearInterval(id);return}const f=k<=st?k/st:(2*st-k)/st;pk.style.display='';pk.setAttribute('x',(x0+(n.x-x0)*f-1.7).toFixed(1));pk.setAttribute('y',(y0+(n.y-y0)*f-1.7).toFixed(1));
      if(k===st&&nd[i])blip(nd[i]); if(++k>2*st){clearInterval(id);pk.style.display='none';busy=false}},90)};
  setInterval(()=>{const up=N.map((n,i)=>n.ok?i:-1).filter(i=>i>=0);if(awake(el)&&!busy&&up.length)trip(pick(up))},1700);
  ins._lit=()=>{strike($('.sts',el));$$('.rg',el).forEach((r,i)=>strike(r,200+i*140));nd.forEach((n,i)=>strike(n,620+i*170+rnd(0,80)))};
})();

// punch card: thirty days of uptime from Uptime Kuma. A lit card, one row per monitor, one column per day; a hole is a day with downtime.
// Where no slot is printed, nothing was recorded yet
(()=>{const el=$('#card'), ins=el.closest('.ins'), x0=66, px=9.6, y0=22, py=11.6; let M=[],days=[],W=362,Hc=0;
  const day=c=>new Date(days[c]*1e3), dl=c=>p2(day(c).getUTCDate())+' '+MON[day(c).getUTCMonth()];
  on('uptime',()=>{M=mons(); days=D.uptime.days; Hc=y0+M.length*py+16; let holes=0, rec=0, most=0;
    let s=`<polygon class="cd" points="16,0 ${W},0 ${W},${Hc} 0,${Hc} 0,16"/>`;
    days.forEach((_,c)=>s+=`<text x="${(x0+c*px+2.5).toFixed(1)}" y="14" text-anchor="middle">${day(c).getUTCDate()}</text>`);
    M.forEach((m,r)=>{s+=`<text class="rl" x="9" y="${(y0+r*py+7).toFixed(1)}">${esc(m.name.toUpperCase().slice(0,13))}</text>`; most=Math.max(most,m.down.filter(v=>v!=null).length);
      m.down.forEach((v,c)=>{if(v==null)return;rec++;if(v>0)holes++;s+=`<rect class="${v>0?'h':'u'}" data-c="${c}" x="${(x0+c*px).toFixed(1)}" y="${(y0+r*py).toFixed(1)}" width="5" height="8"/>`})});
    s+=`<text x="9" y="${(Hc-6).toFixed(1)}">PLINT PL-01 · UPTIME · ${dl(0)} – ${dl(days.length-1)} · A HOLE = DOWN THAT DAY</text><path class="k rdm" d="M${(x0+(days.length-1)*px-1).toFixed(1)} -9h7l-3.5 6z"/>`;
    el.innerHTML=svg(`0 -10 ${W} ${(Hc+10).toFixed(1)}`,s);
    $('.rd',ins).textContent=`${holes} HOLE${holes===1?'':'S'} IN ${rec} · ${most} DAYS ON RECORD`;
  });
  el.addEventListener('click',e=>{const b=$('svg',el).getBoundingClientRect(),k=W/b.width,x=(e.clientX-b.left)*k,y=(e.clientY-b.top)*k-10;
    const c=Math.floor((x-x0+2.3)/px),r=Math.floor((y-y0+1.8)/py); if(c<0||c>=days.length||r<0||r>=M.length)return;
    const v=M[r].down[c], min=Math.round(v*1440);
    say(ins,`${M[r].name.toUpperCase()} · ${dl(c)} · ${v==null?'NO RECORD':v>0?'DOWN '+(min>=90?(min/60).toFixed(1)+' H':Math.max(1,min)+' MIN'):c===days.length-1?'UP SO FAR TODAY':'UP ALL DAY'}`)});
  ins._lit=()=>$$('.u,.h',el).forEach(h=>strike(h,300+h.dataset.c*30+rnd(0,40)));   // punched day by day
})();

// internet: the last 28 speed tests, one column each, in cells
(()=>{const el=$('#net'), ins=el.closest('.ins');
  on('speedtest',()=>{const R=D.speedtest.results.slice(-28), ok=R.filter(r=>r.ok), cell=Math.max(500,Math.ceil(Math.max(0,...ok.map(r=>r.down))/500)*500)/10, last=ok[ok.length-1], lab=[]; let s='', dy='';
    R.forEach((r,i)=>{const d=iso(r.at), n=r.ok?Math.min(10,Math.max(1,Math.round(r.down/cell))):0;
      s+=`<g data-t="${dm(d)} ${hm(d)} · ${r.ok?Math.round(r.down)+' DOWN · '+Math.round(r.up)+' UP · '+(+r.ping).toFixed(1)+' MS':'TEST FAILED'}"><rect class="hit" x="${i*12-1.5}" y="0" width="12" height="112"/>`;
      for(let k=0;k<10;k++)s+=`<rect x="${i*12}" y="${(9-k)*10}" width="9" height="7"${k>=n?' class="z"':i===R.length-1&&k===n-1?' class="now"':''}/>`;
      s+='</g>'; if(ymd(d)!==dy){dy=ymd(d);lab.push([i,DAY[d.getDay()]])}});
    if(lab.length>1&&lab[1][0]-lab[0][0]<3)lab.shift();
    lab.forEach(([i,t])=>s+=`<text x="${i*12}" y="110">${t}</text>`);
    el.innerHTML=(last?`<div class="read"><span>DOWN ${Math.round(last.down)} MBIT</span><span class="dim">UP ${Math.round(last.up)} MBIT</span><span class="dim">PING ${(+last.ping).toFixed(1)} MS</span></div>`:'')+`<svg class="plot" viewBox="0 0 336 112" role="img">${s}</svg>`;
    $('.rd',ins).textContent=`ONE CELL = ${cell} MBIT DOWN · A TEST EVERY 6 H`+(last?' · LAST '+hm(iso(last.at)):'');
  });
})();

// backups: the last fourteen nights, one block each; the blinking one is the next run
(()=>{const el=$('#nights'), ins=el.closest('.ins');
  on('backups',()=>{const N=nights(), nx=new Date(); if(nx.getHours()*60+nx.getMinutes()>=210)nx.setDate(nx.getDate()+1);
    el.innerHTML=N.out.map(({d,r})=>`<i class="${r&&!r.err?'':'miss'}" data-t="${dm(d)} · ${r?r.n+' SOURCES · '+gb(r.size/1e9)+' COVERED · '+r.err+' ERRORS':'NO SNAPSHOT'}"></i>`).join('')+`<i class="wait" data-t="NEXT RUN · ${dm(nx)} 03:30"></i>`;
    const kept=N.out.filter(x=>x.r&&!x.r.err).length;
    $('.rd',ins).textContent=`${kept} OF ${N.out.length} NIGHTS KEPT`+(N.last?' · '+gb(N.last.size/1e9)+' COVERED':'')+(N.oldest?' · OLDEST KEPT '+dm(N.oldest):'');
  });
  el.addEventListener('click',e=>{if(e.target!==el)return;let best=null,bd=1e9;   // a tap between two blocks reads the nearest one
    $$('i',el).forEach(i=>{const b=i.getBoundingClientRect(),d=Math.hypot(e.clientX-b.left-b.width/2,e.clientY-b.top-b.height/2);if(d<bd){bd=d;best=i}});
    if(best)say(ins,best.dataset.t)});
})();

/* ── switching on: one light after another, in page order ── */
const ITEMS='.tick,.mast .id>*,.mast .clock>*,.secs,.cairn,.big>div,.src,.grp,.reg h4,.reg a,.ins,.plate .dith,.plate .pl>*';
$$('.fl').forEach(e=>e.classList.remove('fl')); $$(ITEMS).forEach(e=>e.classList.add('fl'));
function light(e){   // switch one item on, and every light inside it by itself
  if(e===cairn){e.classList.add('on');return raise()}
  const art=c=>parts(c,c.dataset.art==='word'?5:ALL.length,c.dataset.art!=='word');
  if(e.matches('canvas.pix')){e.classList.add('on');return art(e)}
  strike(e); if(!live())return;
  e._lit&&e._lit();
  $$('canvas.pix[data-art]',e).forEach(c=>c.dataset.arg||art(c));
  $$('.blk,.inv',e).forEach(s=>strike(s,rnd(140,520)));
  $$('.dg',e).forEach(s=>{const d=rnd(60,380);strike(s,d);spin(s,d+rnd(200,520))});
  [['.nights i',34],['.bar i',16],['.plot g',26],['.tape i',9],['.acts>*',110]].forEach(([sel,step])=>$$(sel,e).forEach((s,i)=>strike(s,i*step+rnd(0,step))));
  $$('.dots',e).forEach(b=>{b.dataset.v=b.dataset.v||parseFloat(b.style.getPropertyValue('--v'));let k=0;b.style.setProperty('--v','0%');   // bars fill in steps
    const id=setInterval(()=>{k++;b.style.setProperty('--v',b.dataset.v*k/6+'%');if(k>=6)clearInterval(id)},70)});
}
let queue=[], running=null;
const next=()=>{let e; while((e=queue.shift())){if(seen(e)){light(e);break} e.classList.add('on');$$('.row',e).forEach(r=>r.classList.add('on'))}   // what was scrolled past is simply on
  running=queue.length?setTimeout(next,queue.length>14?40:rnd(55,125)):null};
const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){queue.push(e.target);io.unobserve(e.target)}});
  queue.sort((a,b)=>a.compareDocumentPosition(b)&4?-1:1); if(!running&&queue.length)next()},{rootMargin:'0px 0px -4% 0px'});
const lights=()=>{statics();$$('.fl').forEach(e=>io.observe(e))};
let booted=0; try{booted=sessionStorage.plintBoot}catch(e){}
const go=()=>{if(booted||calm()){stage.hidden=true;left=true;root.classList.remove('boot');return lights()}statics();intro()};
(document.fonts&&document.fonts.ready?document.fonts.ready:Promise.resolve()).then(go);
let rz, lw=innerWidth; addEventListener('resize',()=>{if(innerWidth===lw)return;lw=innerWidth;clearTimeout(rz);rz=setTimeout(()=>{statics();if(!left)finish()},150)});
})();
