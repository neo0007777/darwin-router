/* Darwin Router — deterministic 60-second motion design.
 * npm ci && npm run render; requires ffmpeg + ffprobe on PATH.
 * No model calls, network requests, or live inference are made by this renderer.
 */
const { createCanvas, loadImage } = require('@napi-rs/canvas');
const fs = require('node:fs');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { once } = require('node:events');

const ROOT = path.resolve(__dirname, '..');
const BUILD = path.join(__dirname, '.build');
const ASSETS = path.join(ROOT, 'assets');
const W = 1920, H = 1080, FPS = 30, DURATION = 60;
const C = { bg:'#080d10', panel:'#0f191d', panel2:'#121e23', line:'#2b3d43', white:'#f8f5ef', muted:'#a9b8ba', mint:'#92f5bc', blue:'#8c9fff', gold:'#e8d69e' };
const chapters = [
  {start:0, end:6, label:'INTRO'},
  {start:6, end:15, label:'ROUTE'},
  {start:15, end:24, label:'JUDGE'},
  {start:24, end:33, label:'IMPROVE'},
  {start:33, end:42, label:'RESULTS'},
  {start:42, end:52, label:'IN ACTION'},
  {start:52, end:60, label:'DARWIN'}
];
fs.mkdirSync(BUILD, {recursive:true});
// macOS system typography; portable fallback is included in the font stack.
const FONT = 'Helvetica Neue, Arial, sans-serif';
const MONO = 'Menlo, monospace';
const canvas = createCanvas(W,H), ctx = canvas.getContext('2d');
let hero, dashboard;

function cmd(args) {
  const p = spawnSync('ffmpeg', ['-hide_banner','-loglevel','error','-y',...args], {stdio:'inherit'});
  if (p.status !== 0) throw new Error(`ffmpeg failed (${p.status})`);
}
const clamp = (v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const ease = v => {v=clamp(v); return 1-Math.pow(1-v,3);};
const smooth = v => {v=clamp(v); return v*v*(3-2*v);};
const lerp = (a,b,t)=>a+(b-a)*t;

function text(s,x,y,size=32,color=C.white,weight=400,align='left',font=FONT) {
  // Canvas font parsing requires discrete CSS weights on some native backends.
  ctx.fillStyle=color; ctx.font=`${Math.round(weight/100)*100} ${size}px ${font}`;
  ctx.textAlign=align; ctx.textBaseline='top'; ctx.fillText(s,x,y);
}
function tracked(s,x,y,size=22,color=C.muted,spacing=3) {
  ctx.fillStyle=color; ctx.font=`500 ${size}px ${FONT}`; ctx.textAlign='left'; ctx.textBaseline='top';
  for (const ch of s) {ctx.fillText(ch,x,y); x+=ctx.measureText(ch).width+spacing;}
}
function line(x1,y1,x2,y2,color=C.line,width=2) {
  ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
}
function round(x,y,w,h,r=20,fill=C.panel,stroke=C.line) {
  ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();
  if(stroke) {ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}
}
function circle(x,y,r,fill,stroke=null,width=2) {
  ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
  if(fill){ctx.fillStyle=fill;ctx.fill();}
  if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke();}
}
function glow(x,y,r,color) {
  ctx.save();ctx.shadowBlur=r*2;ctx.shadowColor=color;circle(x,y,r,color);ctx.restore();
}
function tag(s,x,y,w,color=C.mint) {
  round(x,y,w,46,23,C.panel,color); text(s,x+w/2,y+11,20,color,500,'center');
}
function reveal(t,delay,fn) {
  const p=ease((t-delay)/.85); if(p<=0)return;
  ctx.save();ctx.globalAlpha*=p;ctx.translate(0,(1-p)*22);fn();ctx.restore();
}
function base() {
  ctx.fillStyle=C.bg;ctx.fillRect(0,0,W,H);
  const g=ctx.createRadialGradient(1560,390,20,1500,470,1000);
  g.addColorStop(0,'#12272c');g.addColorStop(1,C.bg);ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  // Very fine technical register. Dots are decorative, not measured data.
  ctx.fillStyle='#1b2a2e';
  for(let x=110;x<1830;x+=44)for(let y=148;y<985;y+=44)ctx.fillRect(x,y,1.6,1.6);
}
function brand(section,n) {
  text('darwin',110,50,29,C.white,700);text('/',213,50,29,C.mint,400);text('router',236,50,29,C.white,400);
  if(section)tracked(section,110,126,20,C.mint,3);
  text(String(n).padStart(2,'0')+' / 06',1810,54,21,C.muted,400,'right',MONO);
}
function footer(s) {text(s,110,967,26,C.muted);}
function pathPoint(points,u) {
  const [a,b,c,d]=points,v=1-u;
  return [v*v*v*a[0]+3*v*v*u*b[0]+3*v*u*u*c[0]+u*u*u*d[0],v*v*v*a[1]+3*v*v*u*b[1]+3*v*u*u*c[1]+u*u*u*d[1]];
}
function routePath(points,color,t,speed=0.3,particles=3) {
  ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(...points[0]);ctx.bezierCurveTo(...points[1],...points[2],...points[3]);ctx.stroke();
  for(let i=0;i<particles;i++){
    const u=((t*speed+i/particles)%1+1)%1,p=pathPoint(points,u);
    glow(...p,4.5,color);
  }
}
function smallArrow(x,y,color=C.mint) {line(x-10,y-7,x,y,color,2);line(x-10,y+7,x,y,color,2);}

function intro(t) {
  base();
  const z=1+0.022*smooth(t/6),w=W*z,h=H*z;
  ctx.drawImage(hero,(W-w)/2,(H-h)/2,w,h);
  // A controlled light point carries the transition into the routing sequence.
  ctx.save();ctx.globalAlpha=.45*clamp((t-4)/1.5);line(110,1032,110+1700*ease((t-4)/2),1032,C.mint,3);ctx.restore();
}
function routing(t) {
  base();brand('01 / SEMANTIC ROUTING',1);
  reveal(t,0,()=>text('Every request finds its lane.',110,183,82,C.white,650));
  reveal(t,.15,()=>text('Text is embedded, matched to examples, then dispatched.',110,295,34,C.muted));
  reveal(t,.45,()=>{
    routePath([[440,510],[475,510],[485,510],[525,510]],C.muted,t);
    routePath([[835,510],[870,510],[880,510],[915,510]],C.mint,t);
    routePath([[1245,510],[1330,510],[1305,419],[1380,419]],C.mint,t);
    routePath([[1245,510],[1330,510],[1305,631],[1380,631]],C.blue,t+.5);
    round(110,422,330,176); tracked('REQUEST',138,448,18,C.muted,2);
    text('Text prompt',138,492,35,C.white,600);text('OpenAI-shaped API',138,548,22,C.muted);
    round(525,422,310,176);tracked('SIE EMBEDDINGS',553,448,18,C.mint,2);
    text('Qwen3',553,490,37,C.white,600);text('Embedding-4B',553,541,27,C.muted);
    round(915,391,330,238,22,C.panel2);tracked('EXEMPLAR INDEX',941,419,18,C.mint,2);
    text('cosine kNN',941,460,35,C.white,600);
    for(let i=0;i<7;i++)circle(956+i*39,548,10,i<5?C.mint:C.blue);
    text('7 nearest neighbours',941,588,21,C.muted);
    round(1380,342,430,154,20,C.panel,C.mint);tracked('SIMPLE',1410,369,18,C.mint,2);
    text('Qwen3.5-4B',1410,410,38,C.white,600);text('Superlinked Inference Engine',1410,459,22,C.muted);
    round(1380,554,430,154,20,C.panel,C.blue);tracked('REASONING',1410,582,18,C.blue,2);
    text('qwen-max',1410,623,38,C.white,600);text('Alibaba Cloud Model Studio',1410,672,22,C.muted);
  });
  reveal(t,1.6,()=>{
    line(110,797,1810,797);text('A vector index is the routing policy.',110,835,46,C.white,550);
    text('Image, vision and video requests use separate pre-route rules.',110,906,27,C.muted);
  });
}
function judge(t) {
  base();brand('02 / THE DARWIN LOOP',2);
  reveal(t,0,()=>text('Uncertainty earns a second opinion.',110,183,76,C.white,650));
  reveal(t,.1,()=>text('Compare two real answers. Keep model identities hidden.',110,291,34,C.muted));
  reveal(t,.4,()=>{
    routePath([[508,549],[540,549],[548,549],[583,549]],C.gold,t);
    routePath([[991,549],[1025,549],[1035,549],[1070,549]],C.gold,t);
    round(110,420,398,270);tracked('EXPLORATION GATE',142,454,18,C.gold,2);
    text('Low margin',142,505,40,C.white,600);text('or 10% sample',142,559,37,C.white,400);
    text('Asynchronous review',142,640,24,C.muted);
    round(583,420,408,270);tracked('SHADOW RUN',615,454,18,C.gold,2);
    text('Try the other',615,505,40,C.white,600);text('text route',615,559,40,C.white,600);
    text('The real alternative response',615,640,23,C.muted);
    round(1070,420,740,270,20,C.panel2,C.gold);tracked('BLIND A/B JUDGE',1102,454,18,C.gold,2);
    round(1102,507,310,86,12,C.bg);text('Response A',1257,531,29,C.white,500,'center');
    round(1440,507,338,86,12,C.bg);text('Response B',1609,531,29,C.white,500,'center');
    text('Correctness + completeness',1102,634,30,C.white,500);
  });
  reveal(t,1.7,()=>{
    text('Prefer the cheaper answer when it is adequate.',110,796,49,C.mint,550);
    text('Clear verdicts proceed to guarded writeback.',110,872,32,C.muted);
  });
  footer('The gate opens when the kNN margin is below 0.7, or via random sampling.');
}
function learn(t) {
  base();brand('03 / ONLINE ADAPTATION',3);
  reveal(t,0,()=>text('Learning is an index write.',110,183,86,C.white,650));
  reveal(t,.1,()=>text('A judged query becomes a labelled example for future routing.',110,300,34,C.muted));
  reveal(t,.4,()=>{
    round(110,402,730,446);tracked('ILLUSTRATIVE EXEMPLAR',144,436,18,C.muted,2);
    text('{',146,494,31,C.muted,400,'left',MONO);
    const rows=[['"text"',': "the judged query",'],['"label"',': "simple",'],['"source"',': "judge",'],['"embedding"',': [ ... ]']];
    rows.forEach((r,i)=>{text(r[0],178,547+i*53,27,C.mint,400,'left',MONO);text(r[1],r[0]==='"embedding"'?359:310,547+i*53,27,C.white,400,'left',MONO);});
    text('}',146,770,31,C.muted,400,'left',MONO);
    round(960,402,850,446);tracked('ROUTER MEMORY',998,436,18,C.mint,2);
    tag('+1 labelled exemplar',1480,427,288,C.mint);
    for(let row=0;row<6;row++)for(let col=0;col<13;col++){
      const x=1016+col*59,y=526+row*48;
      const color=col<6+(row%3)?C.mint:C.blue;
      ctx.save();ctx.globalAlpha*=.36+((col*7+row*3)%5)*.105;circle(x,y,5.5,color);ctx.restore();
    }
    const p=smooth((t-1.2)/2.8),nx=lerp(856,1380,p),ny=lerp(672,622,p);
    routePath([[840,625],[900,625],[900,625],[960,625]],C.mint,t,.25,1);
    glow(nx,ny,9,C.mint);circle(nx,ny,20+5*Math.sin(t*3),null,C.mint,1.5);
    text('numpy + JSONL',998,790,24,C.muted,400,'left',MONO);
  });
  reveal(t,1.2,()=>{
    text('No training job. No weight update.',110,895,40,C.white,550);
  });
  footer('Guardrails: deduplication · insert caps · protected bootstrap · verdict-flip hysteresis');
}
function results(t) {
  base();brand('04 / REPORTED HACKATHON RESULTS',4);
  reveal(t,0,()=>text('One day. A better routing policy.',110,183,80,C.white,650));
  reveal(t,.1,()=>text('14 Aug 2026   ·   Cold start 09:54 → end of day 17:45',110,295,30,C.muted));
  reveal(t,.35,()=>{
    round(110,390,825,451);tracked('HOLDOUT ROUTING ACCURACY',147,426,20,C.muted,2);
    text('75%',147,481,105,C.white,650);text('→',405,496,76,C.muted,300);text('85%',530,481,105,C.mint,650);
    text('+10 percentage points',147,618,32,C.mint,500);
    line(147,683,892,683);text('40 fixed queries',147,714,37,C.white,550);text('Evaluation traffic never enters the index.',147,772,27,C.muted);
    round(985,390,825,451);tracked('ESTIMATED COST PER REQUEST',1022,426,20,C.muted,2);
    text('50%',1022,481,111,C.mint,650);text('lower',1300,525,55,C.white,500);
    text('$0.0022 → $0.0011',1022,619,35,C.white,500,'left',MONO);
    line(1022,683,1767,683);text('57 → 91 exemplars',1022,714,37,C.white,550);text('The index grows as judged examples arrive.',1022,772,27,C.muted);
  });
  reveal(t,1.4,()=>text('Adaptation you can inspect, one example at a time.',110,895,39,C.white,500));
  footer('Historical results reported in this repository. Costs use hardcoded token-price estimates.');
}
function inAction(t) {
  base();brand('',5);
  // A compact title gives the real recording most of the canvas.
  text('Watch the learning happen.',110,106,59,C.white,600);
  round(124,189,1672,784,16,C.bg,C.line);
  ctx.drawImage(dashboard,140,205,1640,752);
  text('Recorded 14 Aug 2026 · intermediate dashboard snapshot',140,989,24,C.muted);
}
function outro(t) {
  base();brand('06 / TRAINING-FREE SEMANTIC ROUTING',6);
  reveal(t,0,()=>tag('FIRST PRIZE · SUPERLINKED × QWEN HACKATHON',110,184,665,C.mint));
  ['Route.','Judge.','Improve.'].forEach((s,i)=>reveal(t,i*.27,()=>text(s,110,289+i*132,123,i===2?C.mint:C.white,650)));
  reveal(t,.5,()=>{
    const cx=1408,cy=545,r=246;
    const pts=[[cx,cy-r],[cx+r*.866,cy+r*.5],[cx-r*.866,cy+r*.5]];
    ctx.save();ctx.strokeStyle=C.line;ctx.lineWidth=1;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.stroke();ctx.restore();
    for(let i=0;i<3;i++){
      const a=pts[i],b=pts[(i+1)%3];
      routePath([a,[lerp(a[0],cx,.55),lerp(a[1],cy,.55)],[lerp(b[0],cx,.55),lerp(b[1],cy,.55)],b],[C.mint,C.blue,C.gold][i],t,.18,2);
      circle(...a,37,C.bg,[C.mint,C.blue,C.gold][i],2);text(String(i+1).padStart(2,'0'),a[0],a[1]-13,24,C.white,500,'center',MONO);
    }
    text('ROUTE',cx,cy-r-81,24,C.mint,500,'center');
    text('JUDGE',pts[1][0]+50,pts[1][1]-9,24,C.blue,500);
    text('IMPROVE',pts[2][0]-58,pts[2][1]-9,24,C.gold,500,'right');
    text('darwin',cx,cy-25,43,C.white,650,'center');
  });
  reveal(t,1.1,()=>{
    text('Training-free. Inspectable. Built in one day.',110,757,36,C.muted);
    line(110,843,1810,843);text('github.com/hashkanna/darwin-router',110,884,38,C.white,500);
    text('A router that learns while it routes.',110,954,28,C.mint);
  });
}
const scenes=[intro,routing,judge,learn,results,inAction,outro];
function timeline(t) {
  if(t<6)return;
  const gap=8,x=110,width=1700;
  for(let i=1;i<chapters.length;i++){
    const ch=chapters[i],w=(width-gap*5)/6,l=x+(w+gap)*(i-1);
    line(l,1040,l+w,1040,C.line,3);
    const p=clamp((t-ch.start)/(ch.end-ch.start));if(p)line(l,1040,l+w*p,1040,C.mint,3);
  }
}
function render(t) {
  ctx.globalAlpha=1;ctx.setTransform(1,0,0,1,0,0);
  let index=chapters.findIndex(c=>t>=c.start&&t<c.end);if(index<0)index=6;
  scenes[index](t-chapters[index].start);
  // Short optical dissolve into each next chapter.
  if(index<6 && t>chapters[index].end-.45){
    ctx.save();ctx.globalAlpha=smooth((t-(chapters[index].end-.45))/.45);scenes[index+1](0);ctx.restore();
  }
  timeline(t);
}
function score() {
  // Original, deterministic ambient electronic score; no samples or external music.
  const sr=44100,n=sr*DURATION,pcm=Buffer.alloc(n*4),TAU=2*Math.PI;
  const chords=[[130.8128,155.5635,195.9977,233.0819],[103.8262,130.8128,155.5635,195.9977],[116.5409,146.8324,174.6141,220],[97.9989,130.8128,155.5635,195.9977]];
  const chordAt=t=>chords[Math.floor(t/7.5)%chords.length];
  const tone=(t,f)=>Math.sin(TAU*f*t)+.18*Math.sin(TAU*f*2*t);
  for(let i=0;i<n;i++){
    const t=i/sr,ch=chordAt(t),fade=Math.min(smooth(t/2),smooth((DURATION-t)/3));
    let l=0,r=0;
    for(let k=0;k<4;k++){
      const a=.019*(.8+.2*Math.sin(TAU*.08*t+k));
      l+=a*tone(t,ch[k]*(1-.0006));r+=a*tone(t,ch[k]*(1+.0006));
    }
    const beat=Math.floor(t/.625),bt=t-beat*.625;
    const f=ch[[0,2,1,3,2,1,3,2][beat%8]]*2;
    const bell=.07*Math.exp(-bt*7)*smooth(bt/.015)*Math.sin(TAU*f*bt);
    l+=bell*(beat%2?.4:1);r+=bell*(beat%2?1:.4);
    const kt=t%1.25,pulse=.07*Math.exp(-kt*17)*Math.sin(TAU*(48*kt+1.5*(1-Math.exp(-kt*18))));
    l+=pulse;r+=pulse;
    // Soft editorial transition chimes.
    for(const mark of [6,15,24,33,42,52]){
      const dt=t-mark;if(dt>=0&&dt<1.5){const chime=.035*Math.exp(-dt*4)*smooth(dt/.02)*Math.sin(TAU*783.9909*dt);l+=chime;r+=chime;}
    }
    pcm.writeInt16LE(Math.round(clamp(l*fade,-1,1)*32767),i*4);pcm.writeInt16LE(Math.round(clamp(r*fade,-1,1)*32767),i*4+2);
  }
  const h=Buffer.alloc(44);h.write('RIFF');h.writeUInt32LE(36+pcm.length,4);h.write('WAVEfmt ',8);h.writeUInt32LE(16,16);h.writeUInt16LE(1,20);h.writeUInt16LE(2,22);h.writeUInt32LE(sr,24);h.writeUInt32LE(sr*4,28);h.writeUInt16LE(4,32);h.writeUInt16LE(16,34);h.write('data',36);h.writeUInt32LE(pcm.length,40);
  fs.writeFileSync(path.join(BUILD,'score.wav'),Buffer.concat([h,pcm]));
}
async function main() {
  cmd(['-ss','35','-i',path.join(ASSETS,'demo_video.mp4'),'-vf','crop=1920:880:0:100','-frames:v','1',path.join(BUILD,'dashboard.jpg')]);
  hero=await loadImage(path.join(ASSETS,'darwin-router-hero.png'));
  dashboard=await loadImage(path.join(BUILD,'dashboard.jpg'));
  if(process.argv.includes('--stills')){
    for(const t of [2,10,19,28,37,47,56]){render(t);fs.writeFileSync(path.join(BUILD,`scene-${t}.png`),canvas.toBuffer('image/png'));}
    console.log('Seven review frames written to '+BUILD);return;
  }
  const video=path.join(BUILD,'motion.mp4');
  const enc=spawn('ffmpeg',['-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgba','-s',`${W}x${H}`,'-r',String(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',video],{stdio:['pipe','inherit','inherit']});
  const encoded=once(enc,'close');
  enc.stdin.on('error',e=>{if(e.code!=='EPIPE')throw e;});
  for(let f=0;f<DURATION*FPS;f++){
    render(f/FPS);
    if(!enc.stdin.write(canvas.data()))await once(enc.stdin,'drain');
    if(f%(FPS*5)===0)console.log(`Rendered ${f/FPS}/${DURATION}s`);
  }
  enc.stdin.end();const [code]=await encoded;if(code!==0)throw new Error('Video encoding failed');
  score();
  cmd(['-i',video,'-ss','35','-t','10','-i',path.join(ASSETS,'demo_video.mp4'),'-i',path.join(BUILD,'score.wav'),
    '-filter_complex','[1:v]crop=1920:880:0:100,scale=1640:752,setsar=1,format=yuva420p,fade=t=in:st=0:d=0.3:alpha=1,fade=t=out:st=9.55:d=0.45:alpha=1,setpts=PTS-STARTPTS+42/TB[screen];[0:v][screen]overlay=x=140:y=205:eof_action=pass:enable=between(t\\,42\\,52)[out];[2:a]loudnorm=I=-21:TP=-2:LRA=7[a]',
    '-map','[out]','-map','[a]','-t',String(DURATION),'-c:v','libx264','-preset','slow','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000','-movflags','+faststart',path.join(ASSETS,'darwin-router-overview.mp4')]);
  // The video preview is a compact, genuinely animated excerpt of the route sequence.
  cmd(['-ss','7.1','-t','7.2','-i',path.join(ASSETS,'darwin-router-overview.mp4'),'-filter_complex','[0:v]fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle','-loop','0',path.join(ASSETS,'darwin-router-preview.gif')]);
  console.log('Finished: hero, 60-second overview and animated README preview.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
