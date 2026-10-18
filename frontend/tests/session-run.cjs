/* End Run / New Run lifecycle against the real streaming Controller with a fake in-process board.
 * Synthetic 4-trial dataset and fake transport only: no serial port, no board, no production assets. */
'use strict';
const assert=require('node:assert/strict');
const P=require('../streaming/protocol.js'),{Controller,hash}=require('../streaming/controller.js'),{createRunManager}=require('../summary/session-run.js');
const T=P.T;
function ready(){const p=new Uint8Array(69),v=new DataView(p.buffer);p.set(P.model.match(/../g).map(s=>parseInt(s,16)));v.setFloat32(32,.04422667250037193,true);v.setInt32(36,-10,true);v.setUint32(40,11000,true);[1,1,22,500].forEach((n,i)=>v.setUint16(44+2*i,n,true));p[52]=1;p.set(new TextEncoder().encode('TEST-FAKE-BOARD'),53);return p;}
const sleep=ms=>new Promise(ok=>setTimeout(ok,ms));
async function until(f,label='condition'){for(let i=0;i<2000;i++){if(f())return;await sleep(2);}throw Error('Timed out waiting for '+label);}
class Board{
 constructor({predict=()=>0,hold=false,mode='normal'}={}){this.predict=predict;this.hold=hold;this.mode=mode;this.calls=0;this.release=null;this.onData=()=>{};this.onDisconnect=()=>{};this.parser=new P.Parser(f=>this.frame(f));}
 async write(b){this.parser.push(b);}
 send(f){this.onData(P.encode(f));}
 frame(f){if(f.type===T.HELLO){this.send({...f,type:T.READY,payload:ready()});return;}if(f.type!==T.INFER)return;this.calls++;if(this.mode==='disconnect'){this.onDisconnect();return;}
  const crc=P.crc(f.payload),ack=new Uint8Array(4);new DataView(ack.buffer).setUint32(0,crc,true);this.send({...f,type:T.ACCEPTED,payload:ack});if(this.mode==='missing')return;
  const cls=this.predict(f.trial),p=new Uint8Array(16),v=new DataView(p.buffer);v.setUint32(0,crc,true);v.setFloat32(4,cls?-1:1,true);v.setFloat32(8,cls?1:-1,true);p[12]=cls;
  const respond=()=>{this.release=null;this.send({...f,type:T.RESULT,payload:p});if(this.mode==='duplicate')this.send({...f,type:T.RESULT,payload:p});};
  if(this.hold)this.release=respond;else setTimeout(respond,0);}
}
// Synthetic EEG per trial index: three channels with known amplitudes (Cz grows with the index, C3 fixed 5, C4 fixed 10); trial 1 is artifact-flagged.
const EEG_NAMES=['Cz','C3','C4'];
function fixtureChannels(i){const amp={Cz:1+i,C3:5,C4:10};const out={};for(const n of EEG_NAMES)out[n]=Object.freeze(Array.from({length:500},(_,j)=>amp[n]*Math.SQRT2*Math.sin(j/250*2*Math.PI*10)));return Object.freeze(out);}
async function makeDataset(n=4,subject=1,{withEEG=true}={}){const trials=[],inputs=[];for(let i=0;i<n;i++){const b=new Uint8Array(11000).fill(i+1);inputs.push(b);trials.push({index:i,trial_id:`S${subject}/1test/run0/cue${100+i}`,ground_truth:i%2,source_trial:{subject,session:'1test',run:'0',cue_sample:100+i,artifact:i===1},input_sha256:await hash(b)});}const d={schema:1,model_sha256:P.model,channel_order:EEG_NAMES,trials,input:async i=>inputs[i]};if(withEEG){let calls=0;d.channels=i=>{calls++;return fixtureChannels(i);};d.channelCalls=()=>calls;}return d;}
function visuals(){const v={auto:true,pending:null,count:0,visualize(){v.count++;return new Promise(ok=>{if(v.auto)setTimeout(ok,0);else v.pending=ok;});},cancel(){if(v.pending){const ok=v.pending;v.pending=null;ok();}}};return v;}
function fakeView(){const calls=[];return {calls,render(s,o){calls.push(['render',s.session.status,!!o?.final,s.metrics.total]);},ending(d){calls.push(['ending',d]);},message(t){calls.push(['message',t]);},clear(){calls.push(['clear']);},hide(){calls.push(['hide']);},show(){}};}
let clockMs=1_000_000;const now=()=>clockMs;
function harness(dataset,board,{stopAt=Infinity,timeoutMs=10000,ackMs=5000}={}){
 const visual=visuals(),view=fakeView(),rule={stopAt},resetErrors=[];let controller;
 controller=new Controller({dataset,transport:board,visualize:visual.visualize,cancelVisualization:visual.cancel,timeoutMs,ackMs,onRecord:()=>{if(controller.records.length>=rule.stopAt)controller.pause();}});
 const attach=()=>createRunManager({controller,dataset,firmware:controller.firmware,cancelVisualization:visual.cancel,view,now,reset:async()=>{try{await controller.reset(777);}catch(e){resetErrors.push(e.message);}}});
 return {controller,visual,view,rule,resetErrors,attach};
}
(async()=>{
 const dataset=await makeDataset(4);
 // A. Play, Pause, Resume (statistics preserved), End Run while paused, repeated End Run, New Run, second run.
 {
  const board=new Board({predict:i=>i<2?i%2:1-i%2});const h=harness(dataset,board,{stopAt:2});await h.controller.handshake();const mgr=h.attach();
  assert.equal(mgr.metadata.modelId,'model2-subject1-tinyeegnet-int8-ethosu85','verified metadata resolved from the handshake hash');
  assert.deepEqual(h.view.calls,[],'no summary before the first Play');
  assert.equal(mgr.startedAt,null);clockMs+=5000;await h.controller.play();assert.equal(h.controller.records.length,2);assert.equal(mgr.startedAt,1_005_000,'started at first Play');
  assert.deepEqual(h.view.calls,[['render','paused',false,2]],'Pause switches the panel to a live summary once');assert.equal(mgr.liveSummary.metrics.total,2);assert.equal(mgr.summary,null,'pause never finalizes');
  h.rule.stopAt=4;clockMs+=20000;await h.controller.play();assert.equal(h.controller.records.length,4);assert.equal(mgr.startedAt,1_005_000,'Resume does not restart the clock');
  assert.deepEqual(h.view.calls.slice(1),[['hide'],['render','paused',false,4]],'Resume hides the summary; the next pause shows the updated one');
  // Repeated Pause/Resume keeps statistics and playback state.
  for(let k=0;k<3;k++){h.rule.stopAt=h.controller.records.length+1;const before=h.controller.records.length,played=h.visual.count;await h.controller.play();assert.equal(h.controller.records.length,before+1);assert.equal(h.visual.count,played+1,'queued visualization still runs');assert.deepEqual(h.view.calls.slice(-2),[['hide'],['render','paused',false,before+1]]);}
  assert.equal(h.controller.records.length,7);assert.equal(mgr.liveSummary.metrics.total,7);assert.equal(mgr.history.length,0);
  h.rule.stopAt=4;clockMs+=1000;const summary=await mgr.endRun();assert.equal(mgr.phase,'finalized');
  assert.equal(summary.session.status,'completed');assert.equal(summary.metrics.total,7);assert.equal(summary.metrics.correct,h.controller.records.filter(x=>x.class_id===x.ground_truth).length,'correct count matches the validated records');assert.equal(summary.metrics.unique.correct,2,'indices 0 and 1 are the correct ones in the first cycle');assert.equal(summary.metrics.unique.count,4);assert.equal(summary.metrics.repeated,3);
  assert.equal(summary.metrics.duration.ms,21000,'first Play to End Run, pauses included');assert.equal(summary.metrics.latency.count,7,'browser round trip measured per completed request');assert.equal(mgr.liveSummary,null,'finalized snapshot replaces the live view');
  assert.equal(summary.session.modelHash,P.model);assert.equal(summary.session.firmwareBuild,'TEST-FAKE-BOARD');assert.equal(summary.session.subjectId,1);assert.equal(summary.session.seed,42);
  assert.deepEqual(summary.classes.map(c=>c.name),['left_hand','right_hand']);assert.equal(summary.results.every(r=>typeof r.logits[0]==='number'),true);
  assert.deepEqual(h.view.calls.slice(-2).map(c=>c[0]),['ending','render']);assert.deepEqual(h.view.calls.at(-1),['render','completed',true,7]);
  // EEG overview from the dataset provider with verified µV: pooled over the 4 unique recordings, one channels() call each despite 7 trials.
  const e=summary.eeg;assert.equal(e.available,true);assert.equal(e.units,'µV');assert.equal(e.recordings.used,4);assert.equal(e.highest.name,'C4');assert.equal(e.lowest.name,'Cz');assert.equal(e.artifact.flagged,1);assert.equal(e.sampleCount,3*4*500);
  assert(Math.abs(e.highest.rms-10)<1e-9&&Math.abs(e.channels.find(c=>c.name==='C3').rms-5)<1e-9,'RMS of a sine with amplitude a*sqrt2 is a');assert(Math.abs(e.lowest.rms-Math.sqrt((1+4+9+16)/4))<1e-9,'Cz pooled over indices 0..3');
  assert.equal(dataset.channelCalls(),4,'EEG read once per unique recording, not per frame or per repeat');assert.throws(()=>{summary.eeg.channels.push(1);},TypeError);
  assert.equal(await mgr.endRun(),summary,'repeated End Run returns the same snapshot');await assert.rejects(h.controller.play(),/Run ended/,'Play cannot restart a finalized run');
  assert.equal(await mgr.newRun(),true);assert.equal(mgr.phase,'idle');assert.equal(mgr.summary,null);assert.equal(mgr.liveSummary,null);assert.equal(mgr.startedAt,null);assert.equal(h.controller.records.length,0);assert.equal(h.controller.archives[0].length,7);assert.equal(h.view.calls.at(-1)[0],'clear');
  assert.equal(h.controller.paused,true,'New Run never starts inference');const afterReset=h.view.calls.length;h.controller.pause();assert.equal(h.view.calls.length,afterReset,'paused before the first Play of a new run shows the prediction view, not a summary');
  h.rule.stopAt=1;await h.controller.play();assert.equal(h.controller.records.length,1);const second=await mgr.endRun();assert.equal(second.metrics.total,1);assert.equal(mgr.history.length,2);assert.notEqual(second.session.sessionId,summary.session.sessionId,'new session nonce after reset');
  assert.equal(second.eeg.recordings.used,1,'New Run resets the EEG accumulation');assert.equal(summary.eeg.recordings.used,4,'earlier snapshot unchanged');
  mgr.detach();assert.equal(Object.prototype.hasOwnProperty.call(h.controller,'play'),false,'detach restores the prototype play');
  console.log('PASS: paused live summary vs finalized snapshot; repeated pause/resume keeps statistics and queued visualization; End Run while paused; Play blocked after finalization; New Run archives and clears without starting inference.');
 }
 // A2. Pause during an outstanding request: the paused summary updates exactly once when the valid response arrives; its visualization stays queued.
 {
  const board=new Board({hold:true,predict:()=>1});const h=harness(dataset,board);await h.controller.handshake();const mgr=h.attach();
  const run=h.controller.play();await until(()=>board.release,'request in flight');h.controller.pause();
  assert.deepEqual(h.view.calls.at(-1),['render','paused',false,0],'paused summary before the response');const n=h.view.calls.length;
  board.release();await run;
  assert.equal(h.controller.records.length,1);assert.deepEqual(h.view.calls.slice(n),[['render','paused',false,1]],'one update with the completed result');assert.equal(mgr.liveSummary.metrics.total,1);assert.equal(h.visual.count,1,'queued visualization ran rather than being dropped');assert.equal(mgr.summary,null);
  // Resume before the response arrives: the prediction view returns and the result lands without reopening the summary.
  const b2=new Board({hold:true});const g=harness(dataset,b2);await g.controller.handshake();const m2=g.attach();g.rule.stopAt=Infinity;
  const run2=g.controller.play();await until(()=>b2.release,'request in flight');g.controller.pause();assert.equal(g.view.calls.at(-1)[0],'render');
  g.rule.stopAt=1;g.controller.play().catch(()=>{}); // returns the running loop's promise; do not await it while the request is held
  assert.deepEqual(g.view.calls.at(-1),['hide'],'Resume hides the paused summary');assert.equal(g.controller.paused,false);
  b2.release();await run2;await until(()=>g.controller.task===null,'loop settled');
  assert.equal(g.controller.records.length,1);assert.deepEqual(g.view.calls.at(-1),['render','paused',false,1],'the stop rule paused again after the result; one render for that pause');
  console.log('PASS: pause during an outstanding request includes the final valid response once; resume before the response keeps the prediction view.');
 }
 // A3. Restart (Reset) during a pause clears the live summary and returns to the prediction view; errors show an interrupted live summary.
 {
  const h=harness(dataset,new Board(),{stopAt:2});await h.controller.handshake();const mgr=h.attach();await h.controller.play();assert.equal(mgr.liveSummary.metrics.total,2);
  await h.controller.reset(11);assert.equal(mgr.liveSummary,null);assert.equal(mgr.phase,'idle');assert.equal(h.view.calls.at(-1)[0],'clear');assert.equal(h.controller.archives[0].length,2);
  const d=harness(dataset,new Board({mode:'disconnect'}));await d.controller.handshake();const md=d.attach();await d.controller.play();
  assert.equal(d.controller.failed!==null,true);assert.deepEqual(d.view.calls.at(-1),['render','interrupted',false,0],'errors show Run interrupted, not Run complete');assert.equal(md.summary,null,'error does not finalize by itself');
  const fin=await md.endRun();assert.equal(fin.session.status,'interrupted');assert.equal(md.history.length,1);
  console.log('PASS: Reset during pause restores the prediction view; a UART stop shows an interrupted live summary until End Run finalizes it.');
 }
 // B. End Run during an outstanding hardware request: the final response is retained exactly once, no new trial is sent.
 {
  const board=new Board({hold:true,predict:()=>1});const h=harness(dataset,board);await h.controller.handshake();const mgr=h.attach();
  const run=h.controller.play();await until(()=>board.release,'request in flight');assert.equal(h.controller.pending.request,1);
  const ending=mgr.endRun();assert.equal(mgr.phase,'ending');assert.equal(mgr.endRun(),ending,'second click joins the same finalization');
  await assert.rejects(h.controller.play(),/End Run in progress/);assert.equal(await mgr.newRun(),false,'New Run waits for the outstanding request');assert.equal(mgr.summary,null);
  await sleep(20);assert.equal(mgr.phase,'ending','finalization waits for the request');
  board.release();const summary=await ending;await run;
  assert.equal(summary.session.status,'completed');assert.equal(summary.metrics.total,1);assert.equal(board.calls,1,'no new trial after End Run');assert.equal(h.controller.pending,null);
  assert.equal(summary.eeg.available,true);assert.equal(summary.eeg.recordings.used,1,'final in-flight result feeds the EEG overview even though its animation was skipped');
  console.log('PASS: End Run with an outstanding request keeps the final result exactly once and schedules nothing new.');
 }
 // C. End Run while the visualization is paused (animation never advances) does not hang.
 {
  const board=new Board();const h=harness(dataset,board);h.visual.auto=false;await h.controller.handshake();const mgr=h.attach();
  const run=h.controller.play();await until(()=>h.visual.pending,'blocked visualization');assert.equal(h.controller.records.length,1);
  const summary=await Promise.race([mgr.endRun(),sleep(1500).then(()=>{throw Error('End Run hung on a paused visualization');})]);await run;
  assert.equal(summary.metrics.total,1);assert.equal(board.calls,1);
  console.log('PASS: End Run while visualization is paused finalizes without waiting for the animation.');
 }
 // D. Timeout during finalization: interrupted status, unresolved request excluded; New Run is blocked and explained; End Run again is safe.
 {
  const board=new Board({mode:'missing'});const h=harness(dataset,board,{timeoutMs:30,ackMs:200});await h.controller.handshake();const mgr=h.attach();
  const run=h.controller.play();await until(()=>h.controller.pending?.accepted,'accepted request');const summary=await mgr.endRun();await run;
  assert.equal(summary.session.status,'interrupted');assert.equal(summary.metrics.total,0);assert.equal(summary.metrics.accuracy,null);assert.equal(summary.session.interruption.unresolvedRequest,1);assert.match(summary.session.interruption.message,/timeout/);
  assert.equal(summary.eeg.available,false);assert.match(summary.eeg.reason,/no completed recordings/,'unresolved request contributes no EEG');
  assert.equal(await mgr.newRun(),false);assert.equal(h.resetErrors.length,1);assert.match(h.resetErrors[0],/unresolved/);assert.deepEqual(h.view.calls.at(-1),['message','New Run blocked']);assert.equal(mgr.phase,'finalized','a blocked reset keeps the ended run ended');
  const again=await mgr.endRun();assert.equal(again,summary,'End Run again re-shows the same interrupted snapshot');assert.equal(mgr.history.length,1);await assert.rejects(h.controller.play(),/Run ended/);
  console.log('PASS: result timeout during finalization is reported as interrupted; unresolved request is excluded; New Run explains the required reconnect.');
 }
 // E. Disconnect while finalizing an outstanding request.
 {
  const board=new Board({hold:true});const h=harness(dataset,board);await h.controller.handshake();const mgr=h.attach();
  const run=h.controller.play();await until(()=>board.release,'request in flight');const ending=mgr.endRun();board.onDisconnect();const summary=await ending;await run;
  assert.equal(summary.session.status,'interrupted');assert.match(summary.session.interruption.message,/disconnected/);assert.equal(summary.metrics.total,0);
  console.log('PASS: UART disconnect during finalization yields an interrupted summary with no fabricated result.');
 }
 // F. Duplicate hardware responses count once; results retained after an idle disconnect are still summarized.
 {
  const board=new Board({mode:'duplicate',predict:()=>0});const h=harness(dataset,board,{stopAt:1});await h.controller.handshake();const mgr=h.attach();
  await h.controller.play();assert.equal(h.controller.records.length,1);board.onDisconnect();const summary=await mgr.endRun();
  assert.equal(summary.metrics.total,1);assert.equal(summary.metrics.duplicatesIgnored,0,'controller already dropped the duplicate');assert.equal(summary.session.status,'completed');assert.match(summary.session.note,/disconnected/);
  assert.equal(await mgr.newRun(),true);assert.deepEqual(h.view.calls.at(-1),['message','New run ready · reconnect required']);
  console.log('PASS: duplicate responses are counted once; idle disconnect is a note, not an interruption; New Run explains reconnection.');
 }
 // G. Empty session and a Reset issued during End Run (generation guard).
 {
  const h=harness(dataset,new Board());await h.controller.handshake();const mgr=h.attach();const empty=await mgr.endRun();
  assert.equal(empty.session.status,'empty');assert.equal(empty.metrics.total,0);assert.equal(empty.metrics.accuracy,null);assert.equal(empty.metrics.duration,null);assert.equal(empty.eeg.available,false);
  // EEG provider variants: no channels() on the dataset, unverified units, and an explicit host provider with a different channel set.
  const plain=await makeDataset(4,1,{withEEG:false});const p1=harness(plain,new Board(),{stopAt:1});await p1.controller.handshake();const m1=p1.attach();await p1.controller.play();const s1=await m1.endRun();
  assert.equal(s1.metrics.total,1,'classification summary opens without EEG');assert.equal(s1.eeg.available,false);assert.match(s1.eeg.reason,/no EEG source/);
  const p2=harness(dataset,new Board(),{stopAt:1});await p2.controller.handshake();const m2u=createRunManager({controller:p2.controller,dataset,firmware:p2.controller.firmware,metadata:{...m1.metadata,eeg:undefined},view:p2.view,now});await p2.controller.play();const s2=await m2u.endRun();
  assert.equal(s2.eeg.available,false);assert.match(s2.eeg.reason,/units not verified/);assert.deepEqual(s2.classes.map(c=>c.name),['left_hand','right_hand']);
  const p3=harness(dataset,new Board(),{stopAt:2});await p3.controller.handshake();const m3=createRunManager({controller:p3.controller,dataset,firmware:p3.controller.firmware,view:p3.view,now,eeg:{units:'µV',unitsVerified:true,samples:r=>({P9:[1,-1,1,-1],P10:[2,-2,2,-2].map(x=>x*(r.trial+1))}),artifact:()=>null}});await p3.controller.play();const s3=await m3.endRun();
  assert.deepEqual(s3.eeg.channels.map(c=>c.name),['P10','P9']);assert.equal(s3.eeg.recordings.used,2);assert.deepEqual(s3.eeg.artifact,{known:false,flagged:0});
  const board=new Board({hold:true});const g=harness(dataset,board);await g.controller.handshake();const m2=g.attach();
  const run=g.controller.play();await until(()=>board.release,'request in flight');const ending=m2.endRun();const resetting=g.controller.reset(9);board.release();await resetting;await run;
  assert.equal(await ending,null,'End Run yields to the Reset that superseded it');assert.equal(m2.phase,'idle');assert.equal(m2.summary,null);assert.equal(g.controller.records.length,0);assert.equal(g.controller.archives[0].length,1);
  console.log('PASS: empty session shows zero trials and unavailable accuracy; Reset during End Run wins without a stale summary.');
 }
 console.log('PASS: session run lifecycle with the real controller and a fake board only; no physical UART used.');
})().catch(e=>{console.error(e);process.exit(1);});
