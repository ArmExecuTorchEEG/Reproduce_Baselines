/* Hardware-free preview: a simulated board and a synthetic dataset behind the REAL GUI, bridge, controller, protocol,
 * summary and session logic. Loaded by index.html only when the page is opened as ?stream=1&sim=1; the hardware page
 * (?stream=1) never fetches this file, so its asset, handshake and serial checks are untouched.
 *
 * What is simulated: EdgeStreamDataset.load (144 synthetic trials, inputs and 22-channel EEG, generated here, not read
 * from fixtures) and EdgeStream.SerialLink (an in-page board that answers HELLO/INFER with the real wire protocol).
 * Everything else, including the controller's CRC, session and request checks, runs unchanged. Every result shown is
 * synthetic and the page says so in one banner. */
(()=>{'use strict';
if(!window.EdgeSimulatedPreview||!window.EdgeStream||!window.EdgeStreamProtocol||!window.EdgeStreamDataset)return;
const P=window.EdgeStreamProtocol,T=P.T,N=144,SAMPLES=500;
const names=['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'];
function lcg(seed){let s=seed>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};}
const rndInput=lcg(7),rndBoard=lcg(99);
const inputs=Array.from({length:N},()=>{const b=new Uint8Array(11000);for(let i=0;i<b.length;i++)b[i]=Math.floor(rndInput()*256);return b;});
const amp=names.map((_,c)=>2.5+((c*7)%11)*1.1);
const channelCache=new Map();
function channels(i){if(channelCache.has(i))return channelCache.get(i);const r=lcg(1000+i),out={};for(let c=0;c<names.length;c++){const a=amp[c]*(0.8+r()*0.4),f=8+r()*6,ph=r()*6.283,x=new Array(SAMPLES);for(let j=0;j<SAMPLES;j++)x[j]=a*Math.SQRT2*Math.sin(j/250*6.283*f+ph)+(r()-0.5)*a*0.3;out[names[c]]=Object.freeze(x);}const frozen=Object.freeze(out);channelCache.set(i,frozen);return frozen;}
async function makeDataset(){
 const trials=[];for(let i=0;i<N;i++){const run=String(Math.floor(i/24)),cue=1000+i*50;trials.push({index:i,trial_id:`S1/1test/run${run}/cue${cue}`,ground_truth:i%2,source_trial:{subject:1,session:'1test',run,cue_sample:cue,start:cue+250,stop:cue+750,artifact:i%9===5},input_sha256:await window.EdgeStream.hash(inputs[i])});}
 return {schema:1,synthetic:true,model_sha256:P.model,channel_order:names,trials,input:async i=>inputs[i].slice(),channels};
}
function readyPayload(){const p=new Uint8Array(69),v=new DataView(p.buffer);p.set(P.model.match(/../g).map(s=>parseInt(s,16)));v.setFloat32(32,0.04422667250037193,true);v.setInt32(36,-10,true);v.setUint32(40,11000,true);[1,1,22,500].forEach((n,i)=>v.setUint16(44+2*i,n,true));p[52]=1;p.set(new TextEncoder().encode('SIMULATED-BOARD'),53);return p;}
let dataset=null;
/* Same surface as EdgeStream.SerialLink: connect/write/disconnect, onData/onDisconnect, frames, connected. */
class SimulatedBoard{
 constructor(){this.onData=()=>{};this.onDisconnect=()=>{};this.frames=[];this.connected=false;this.port=null;this.timers=new Set();this.parser=new P.Parser(f=>this.frame(f));this.accuracy=0.72;this.latencyMs=140;}
 async connect(){if(this.connected)throw Error('Already connected');this.connected=true;this.port=true;}
 async write(bytes){if(!this.connected)throw Error('Simulated board not connected');this.frames.push({direction:'tx',at:Date.now(),bytes:Array.from(bytes)});this.parser.push(bytes);}
 send(frame){if(!this.connected)return;const bytes=P.encode(frame);this.frames.push({direction:'rx',at:Date.now(),bytes:Array.from(bytes)});this.onData(bytes);}
 later(fn,ms){const t=setTimeout(()=>{this.timers.delete(t);fn();},ms);this.timers.add(t);}
 frame(f){
  if(f.type===T.HELLO){this.later(()=>this.send({...f,type:T.READY,payload:readyPayload()}),60);return;}
  if(f.type!==T.INFER)return;
  const crc=P.crc(f.payload),ack=new Uint8Array(4);new DataView(ack.buffer).setUint32(0,crc,true);this.later(()=>this.send({...f,type:T.ACCEPTED,payload:ack}),20);
  const truth=dataset?.trials[f.trial]?.ground_truth??0,correct=rndBoard()<this.accuracy,cls=correct?truth:1-truth,margin=0.5+rndBoard()*3;
  const p=new Uint8Array(16),v=new DataView(p.buffer);v.setUint32(0,crc,true);v.setFloat32(4,cls?-margin:margin,true);v.setFloat32(8,cls?margin:-margin,true);p[12]=cls;
  this.later(()=>this.send({...f,type:T.RESULT,payload:p}),this.latencyMs);
 }
 async disconnect(){for(const t of this.timers)clearTimeout(t);this.timers.clear();if(!this.connected)return;this.connected=false;this.port=null;this.onDisconnect();}
}
window.EdgeStreamDataset.load=async()=>{dataset=dataset||await makeDataset();return dataset;};
window.EdgeStream.SerialLink=SimulatedBoard;
window.EdgeSimulatedBoard={SimulatedBoard,get dataset(){return dataset;}};
// Session metadata: the summary must not present synthetic results as the deployed model's. Class labels and EEG units
// are reused from the Model 2 contract (the simulated READY carries that hash), the names and source are not.
if(window.EdgeSessionRun){const attach=window.EdgeSessionRun.attach.bind(window.EdgeSessionRun);window.EdgeSessionRun.attach=host=>{const base=window.EdgeModelMetadata?.find?.(P.model)||{};const manager=attach({...host,source:'synthetic-preview',metadata:{...base,modelId:'simulated-board',modelName:'Simulated board (synthetic data)',experimentName:'Hardware-free preview',datasetId:'synthetic-sim',datasetName:'Synthetic dataset generated in the browser'}});label();return manager;};}
function label(){const pill=document.querySelector('.pill.mock'),stat=document.getElementById('boardStat');if(pill)pill.textContent='Simulated board';if(stat)stat.textContent='Simulated board · no hardware';}
// One clear preview label, outside the panels.
const bar=document.createElement('div');bar.id='simBanner';bar.setAttribute('role','note');bar.textContent='HARDWARE-FREE PREVIEW — simulated board, synthetic EEG and results. Not hardware output.';
bar.style.cssText='position:sticky;top:0;z-index:50;padding:8px 16px;background:#3a2a12;color:#ffe3b3;border-bottom:1px solid #f6bd6877;font:600 12px/1.4 "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.3px;text-align:center';
document.body.prepend(bar);label();
const status=document.getElementById('transportStatus');if(status)status.textContent='Simulated board ready · Connect UART starts the simulated handshake (no serial port is opened)';
})();
