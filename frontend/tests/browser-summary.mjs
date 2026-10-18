/* Session summary in a real (isolated, headless) Chrome: the hardware-free preview (real GUI + simulated board, end to
 * end), streaming mode with a fake serial port injected into the page, and the untouched legacy page. No physical port. */
import {page} from './browser-cdp.mjs';import assert from 'node:assert/strict';import fs from 'node:fs/promises';import path from 'node:path';
const base=process.env.EDGEBCI_BASE||'http://127.0.0.1:8000',outputDir=path.resolve(process.env.EDGEBCI_QA_DIR||'/tmp/edgebci-summary-qa');await fs.mkdir(outputDir,{recursive:true});
const sleep=ms=>new Promise(ok=>setTimeout(ok,ms));
const errorsOf=tab=>tab.events.filter(e=>e.method==='Runtime.exceptionThrown'||e.method==='Network.loadingFailed'||(e.method==='Log.entryAdded'&&e.params.entry.level==='error'));
async function until(tab,expr,label=expr,tries=300){for(let i=0;i<tries;i++){const x=await tab.evaluate(expr);if(x)return x;await sleep(60);}throw Error('Timed out: '+label);}
const PANELS='(()=>{const r={};for(const s of [".cursor-panel",".brain-panel",".drawer",".top-grid",".controls"]){const b=D.querySelector(s).getBoundingClientRect();r[s]=[b.x,b.y,b.width,b.height];}return r;})()';
/* Summary layout/content assertions shared by the preview and the live page. D = document expression. */
async function inspectSummary(tab,D,width,{expectResume,expectNewRun,pill}){
 const s=await tab.evaluate(`(()=>{const doc=${D},box=doc.getElementById("sessionSummary"),wrap=doc.querySelector(".ss-matrix-wrap"),card=doc.querySelector(".ss-body"),btn=doc.querySelector(".ss-actions .ss-btn"),meta=doc.querySelector(".ss-eeg-meta"),corner=doc.querySelector(".ss-corner"),rms=doc.querySelector(".ss-eeg-label"),h3=[...doc.querySelectorAll(".ss-h3")].map(e=>e.getBoundingClientRect().top),line=doc.querySelector(".ss-line"),m=doc.querySelector(".ss-matrix-panel"),e=doc.querySelector(".ss-eeg"),a=doc.querySelector(".ss-actions"),foot=doc.querySelector(".ss-foot");const bb=btn?btn.getBoundingClientRect():null,pb=box.getBoundingClientRect();return {hidden:box.hidden,pill:doc.querySelector(".ss-pill").textContent,text:card.textContent,overflow:doc.documentElement.scrollWidth>doc.documentElement.clientWidth,clip:box.scrollWidth>box.clientWidth+1,matrixScroll:wrap?wrap.scrollWidth>wrap.clientWidth+1:false,resume:doc.getElementById("ssResume")?.textContent??null,newRun:doc.getElementById("ssNewRun")?.textContent??null,close:!!doc.getElementById("ssClose"),font:getComputedStyle(box).fontFamily,inPanel:box.parentElement.classList.contains("cursor-panel"),btn:bb&&{cx:bb.left+bb.width/2,w:bb.width,h:bb.height,textAlign:getComputedStyle(btn).textAlign},panelCx:pb.left+pb.width/2,h3Tops:h3,cornerTop:corner?corner.getBoundingClientRect().top:null,rmsTop:rms?rms.getBoundingClientRect().top:null,gapTop:line&&m?Math.min(m.getBoundingClientRect().top,e.getBoundingClientRect().top)-line.getBoundingClientRect().bottom:null,gapBottom:a&&m?a.getBoundingClientRect().top-Math.max(m.getBoundingClientRect().bottom,e.getBoundingClientRect().bottom):null,footInEEG:foot?foot.closest(".ss-eeg")!==null&&foot.scrollWidth<=foot.clientWidth+1:null,diag:doc.querySelectorAll(".ss-diag").length,off:doc.querySelectorAll(".ss-off").length};})()`);
 assert.equal(s.hidden,false);assert.equal(s.inPanel,true,'summary lives inside the cursor panel');assert.match(s.font,/Helvetica Neue/);assert.equal(s.close,false);
 assert.equal(s.overflow,false,'no horizontal overflow at '+width);assert.equal(s.clip,false,'summary not clipped at '+width);assert.equal(s.matrixScroll,false,'matrix fits at '+width);
 assert.equal(s.pill,pill);assert.equal(s.resume,expectResume?'Resume':null);assert.equal(s.newRun,expectNewRun?'New Run':null);
 assert.doesNotMatch(s.text,/ynthetic|validated hardware results|Model hash|Session ID|Elapsed|Completed trials|Back to live view|Includes repeats|Press Play|✦|■| · /);
 assert.match(s.text,/Confusion Matrix.*Actual ↓.*Predicted →.*Top EEG Channels.*RMS \(µV\)/);assert.equal(s.diag,2);assert.equal(s.off,2);
 if(s.footInEEG!==null)assert.equal(s.footInEEG,true,'EEG supporting text sits under Top EEG Channels without clipping');
 if(s.btn){assert(Math.abs(s.btn.cx-s.panelCx)<2,'action centered');assert(s.btn.w>=172&&s.btn.h>=40,'larger action button');assert.equal(s.btn.textAlign,'center');}
 if(width>=1000){if(s.h3Tops.length===2)assert(Math.abs(s.h3Tops[0]-s.h3Tops[1])<1,'headings share one baseline');if(s.cornerTop!==null&&s.rmsTop!==null)assert(Math.abs(s.cornerTop-s.rmsTop)<1,'Actual ↓ aligns with RMS (µV)');if(s.gapTop!==null)assert(s.gapTop>=24,'space above the lower section');if(s.gapBottom!==null)assert(s.gapBottom>=16&&Math.abs(s.gapTop-s.gapBottom)<=40,'space above and below the lower section is balanced: '+JSON.stringify([s.gapTop,s.gapBottom]));}
 return s;
}
/* Drives Connect → Play → Pause → Resume → End Run → New Run on a page whose controller is reachable as W (window expression). */
async function driveSession(tab,W,D,width,{label,connectExpr}){
 const panelsBefore=await tab.evaluate(PANELS.replaceAll('D.',D+'.'));
 assert.equal(await tab.evaluate(`${D}.getElementById('sessionSummary').hidden`),true,'prediction view before Play');
 // Run controls: no diamond anywhere, End Run icon centred with its label.
 const ctl=await tab.evaluate(`(()=>{const d=${D},b=d.getElementById('endRunBtn'),i=b.querySelector('.run-icon-stop'),br=b.getBoundingClientRect(),ir=i.getBoundingClientRect(),r=d.createRange();r.selectNodeContents(b);const cr=r.getBoundingClientRect();return {iconDy:Math.abs((ir.top+ir.height/2)-(br.top+br.height/2)),contentDx:Math.abs((cr.left+cr.width/2)-(br.left+br.width/2)),endText:b.textContent,newText:d.getElementById('newRunBtn').textContent,anyDiamond:d.body.innerHTML.includes('✦')};})()`);
 assert.equal(ctl.endText,'End Run');assert.equal(ctl.newText,'New Run');assert.equal(ctl.anyDiamond,false,'no diamond in the page');assert(ctl.iconDy<=1.5,'End Run square vertically centred with its label');assert(ctl.contentDx<=2,'icon and label centred together');
 await tab.evaluate(`(()=>{const c=${W}.EdgeStreamingController,old=c.onRecord;${W}.stopAt=5;c.onRecord=r=>{old(r);if(c.records.length>=${W}.stopAt)c.pause();};})()`);
 await tab.evaluate(`${D}.getElementById('playBtn').click()`);await until(tab,`${W}.EdgeStreamingController.records.length===5 && ${W}.EdgeStreamingController.paused`,label+': five results then pause');
 await until(tab,`!${D}.getElementById("sessionSummary").hidden`,label+': paused summary');
 let s=await inspectSummary(tab,D,width,{expectResume:true,expectNewRun:false,pill:'Run paused'});
 const live=await tab.evaluate(`(()=>{const m=${W}.EdgeSessionRun.current.liveSummary,c=${W}.EdgeStreamingController;return {total:m.metrics.total,records:c.records.length,final:${W}.EdgeSessionRun.current.summary,phase:${W}.EdgeSessionRun.current.phase,results:${W}.ArmEEGDemo.getState().playback.results.length,playLabel:${D}.getElementById('playBtn').textContent};})()`);
 assert.equal(live.total,5);assert.equal(live.total,live.records);assert.equal(live.final,null,'pause does not finalize');assert.equal(live.phase,'idle');assert.equal(live.results,5,'playback queue intact under the summary');assert.equal(live.playLabel,'▶ Resume');
 // Desktop rows are fixed, so nothing may move at all. On phones the rows are auto-sized and the prediction header's own
 // responsive wrapping depends on its content ("Waiting" vs a prediction), so there the view switch itself is checked:
 // the summary and the prediction view that replaces it on Resume must share the same panel boxes.
 const strict=width>=1000,panelsPaused=await tab.evaluate(PANELS.replaceAll('D.',D+'.'));
 if(strict)assert.deepEqual(panelsPaused,panelsBefore,'no layout jump when the right panel switches');
 await tab.screenshot(`${outputDir}/${label}-paused-${width}.png`);
 // Resume from the panel: shares the top Play flow; statistics continue.
 await tab.evaluate(`${W}.stopAt=7`);await tab.evaluate(`${D}.getElementById('ssResume').click()`);await until(tab,`${D}.getElementById("sessionSummary").hidden && !${W}.EdgeStreamingController.paused`,label+': resumed');
 assert.deepEqual(await tab.evaluate(PANELS.replaceAll('D.',D+'.')),panelsPaused,'switching back to the prediction view moves nothing');
 assert.equal(await tab.evaluate(`${D}.getElementById('playBtn').textContent`),'Ⅱ Pause');
 await until(tab,`${W}.EdgeStreamingController.records.length===7 && ${W}.EdgeStreamingController.paused && !${D}.getElementById("sessionSummary").hidden`,label+': second pause');
 assert.equal(await tab.evaluate(`${W}.EdgeSessionRun.current.liveSummary.metrics.total`),7);assert.equal(await tab.evaluate(`${W}.ArmEEGDemo.getState().playback.results.length`),7,'queued visualizations retained across pause/resume');
 // End Run while paused with a queued visualization.
 await tab.evaluate(`${D}.getElementById('endRunBtn').click()`);await until(tab,`${W}.EdgeSessionRun.current.phase==="finalized"`,label+': finalized');
 s=await inspectSummary(tab,D,width,{expectResume:false,expectNewRun:true,pill:'Run complete'});
 const fin=await tab.evaluate(`(()=>{const m=${W}.EdgeSessionRun.current.summary,c=${W}.EdgeStreamingController;return {total:m.metrics.total,correct:m.metrics.correct,records:c.records.length,recCorrect:c.records.filter(r=>r.class_id===r.ground_truth).length,unique:m.metrics.unique.count,stateUnique:c.state().unique,hash:m.session.modelHash,source:m.session.source,model:m.session.modelName,eeg:{available:m.eeg.available,used:m.eeg.recordings.used,units:m.eeg.units,n:m.eeg.channels.length},order:c.dataset.channel_order,ranks:${D}.querySelectorAll(".ss-rank").length,endDisabled:${D}.getElementById("endRunBtn").disabled,playLabel:${D}.getElementById('playBtn').textContent};})()`);
 assert.equal(fin.total,7);assert.equal(fin.total,fin.records);assert.equal(fin.correct,fin.recCorrect,'summary matches controller.records, not the animated counter');assert.equal(fin.unique,fin.stateUnique);assert.equal(fin.hash.length,64);assert.equal(fin.endDisabled,true);assert.equal(fin.playLabel,'▶ Play');
 assert.equal(fin.eeg.available,true);assert.equal(fin.eeg.used,fin.unique);assert.equal(fin.eeg.units,'µV');assert.equal(fin.eeg.n,fin.order.length);assert.equal(fin.ranks,3);
 if(strict)assert.deepEqual(await tab.evaluate(PANELS.replaceAll('D.',D+'.')),panelsBefore);
 await tab.screenshot(`${outputDir}/${label}-complete-${width}.png`);
 await tab.evaluate(`${D}.getElementById('playBtn').click()`);await sleep(200);assert.equal(await tab.evaluate(`${W}.EdgeStreamingController.paused`),true,'Play after End Run does not restart');
 // New Run from the panel: reset flow, prediction view restored, nothing starts.
 await tab.evaluate(`${D}.getElementById('ssNewRun').click()`);await until(tab,`${W}.EdgeStreamingController.records.length===0 && ${W}.EdgeSessionRun.current.phase==="idle"`,label+': new run');
 assert.equal(await tab.evaluate(`${D}.getElementById('sessionSummary').hidden`),true);assert.equal(await tab.evaluate(`${W}.EdgeStreamingController.archives[0].length`),7);assert.equal(await tab.evaluate(`${W}.EdgeStreamingController.paused`),true,'New Run does not start inference');
 return fin;
}
// 1. Hardware-free preview: the real GUI as ?stream=1&sim=1 inside summary/preview.html, driven end to end.
for(const [width,height] of [[1512,950],[390,844]]){
 const tab=await page(base+'/summary/preview.html',width,height,'document.readyState==="complete" && window.previewReady===true && !!window.previewWindow && !!window.previewWindow.EdgeSimulatedBoard && !!window.previewWindow.EdgeSessionRun');
 const W='window.previewWindow',D='window.previewDocument';
 const sim=await tab.evaluate(`(()=>{const d=${D};return {banners:d.querySelectorAll('#simBanner').length,banner:d.getElementById('simBanner').textContent,pill:d.querySelector('.pill.mock').textContent,flag:${W}.EdgeSimulatedPreview,link:${W}.EdgeStream.SerialLink.name,endDisabled:d.getElementById('endRunBtn').disabled};})()`);
 assert.equal(sim.banners,1,'one preview label');assert.match(sim.banner,/HARDWARE-FREE PREVIEW.*Not hardware output/);assert.equal(sim.pill,'Simulated board');assert.equal(sim.flag,true);assert.equal(sim.link,'SimulatedBoard');assert.equal(sim.endDisabled,true);
 await tab.evaluate(`${D}.getElementById('connectBtn').click()`);await until(tab,`${W}.EdgeStreamingController?.ready===true`,'simulated handshake',500);
 assert.equal(await tab.evaluate(`${W}.EdgeStreamingController.firmware.build`),'SIMULATED-BOARD');assert.equal(await tab.evaluate(`${W}.EdgeStreamingController.dataset.synthetic`),true);
 const fin=await driveSession(tab,W,D,width,{label:'preview'});
 assert.equal(fin.source,'synthetic-preview');assert.match(fin.model,/Simulated board/);
 if(width>=1000){
  // Pause during an outstanding (slow) request: the paused summary updates once with the valid response.
  await tab.evaluate(`${W}.EdgeStreamingController.transport.latencyMs=2500;${W}.stopAt=Infinity`);await tab.evaluate(`${D}.getElementById('playBtn').click()`);await until(tab,`!!${W}.EdgeStreamingController.pending`,'request in flight');
  await tab.evaluate(`${D}.getElementById('playBtn').click()`);await until(tab,`${W}.EdgeStreamingController.paused && !${D}.getElementById("sessionSummary").hidden`,'paused during request');
  assert.equal(await tab.evaluate(`${W}.EdgeSessionRun.current.liveSummary.metrics.total`),0);assert.equal(await tab.evaluate(`!!${D}.getElementById('ssResume')`),true);
  await until(tab,`${W}.EdgeStreamingController.records.length===1 && ${W}.EdgeSessionRun.current.liveSummary.metrics.total===1`,'paused summary updated with the final response',200);
  assert.equal(await tab.evaluate(`${W}.ArmEEGDemo.getState().playback.results.length`),1,'its visualization is queued, not lost');
  await tab.evaluate(`${D}.getElementById('endRunBtn').click()`);await until(tab,`${W}.EdgeSessionRun.current.phase==="finalized"`,'finalized after slow request');
  assert.equal(await tab.evaluate(`${W}.EdgeSessionRun.current.summary.metrics.total`),1);assert.equal(await tab.evaluate(`${W}.EdgeStreamingController.pending`),null);
 }
 assert.deepEqual(errorsOf(tab),[]);await tab.close();
}
console.log('PASS: hardware-free preview drives the real GUI end to end (Connect, Play, Pause summary, Resume, End Run, New Run, pause during a slow request) at desktop and phone widths; one synthetic label; controls without diamonds and with a centred End Run icon.');
// 2. Streaming mode: hardware page never loads the simulation; fake serial board injected into the page.
const tab=await page(base+'/?stream=1');
await until(tab,'!!window.EdgeStream && !!window.EdgeSessionRun && !!ArmEEGDemo.setPlaying');
const iso=await tab.evaluate('({flag:window.EdgeSimulatedPreview,banner:!!document.getElementById("simBanner"),script:!!document.querySelector("script[src*=\\"preview-sim\\"]"),link:EdgeStream.SerialLink.name,sim:typeof window.EdgeSimulatedBoard})');
assert.deepEqual(iso,{flag:false,banner:false,script:false,link:'SerialLink',sim:'undefined'},'hardware mode loads no simulation');
assert.equal(await tab.evaluate("document.getElementById('endRunBtn').hidden"),false);assert.equal(await tab.evaluate("document.getElementById('endRunBtn').disabled"),true,'End Run disabled before a verified session');
assert.equal(await tab.evaluate("document.getElementById('sessionSummary').parentElement.className"),'panel cursor-panel');
await tab.evaluate(`(()=>{let outgoing;window.fakeBoard={calls:0,hold:false,release:null};const P=EdgeStreamProtocol,T=P.T;const send=f=>outgoing.enqueue(P.encode(f));
const parser=new P.Parser(f=>{if(f.type===T.HELLO){const p=new Uint8Array(69),v=new DataView(p.buffer);p.set(P.model.match(/../g).map(x=>parseInt(x,16)));v.setFloat32(32,0.04422667250037193,true);v.setInt32(36,-10,true);v.setUint32(40,11000,true);[1,1,22,500].forEach((n,i)=>v.setUint16(44+i*2,n,true));p[52]=1;p.set(new TextEncoder().encode('TEST-FAKE-ONLY'),53);send({...f,type:T.READY,payload:p});}
if(f.type===T.INFER){fakeBoard.calls++;const crc=P.crc(f.payload),ack=new Uint8Array(4);new DataView(ack.buffer).setUint32(0,crc,true);send({...f,type:T.ACCEPTED,payload:ack});const p=new Uint8Array(16),v=new DataView(p.buffer),cls=f.trial%3===0?1:0;v.setUint32(0,crc,true);v.setFloat32(4,cls?-1:1,true);v.setFloat32(8,cls?1:-1,true);p[12]=cls;const respond=()=>{fakeBoard.release=null;send({...f,type:T.RESULT,payload:p});};if(fakeBoard.hold)fakeBoard.release=respond;else setTimeout(respond,80);}});
const fake={readable:new ReadableStream({start(c){outgoing=c;}}),writable:new WritableStream({write(b){parser.push(b);}}),open:async()=>{},close:async()=>{}};Object.defineProperty(navigator,'serial',{configurable:true,value:{requestPort:async()=>fake}});})()`);
await tab.evaluate("document.getElementById('connectBtn').click()");
const connected=await Promise.race([until(tab,'window.EdgeStreamingController?.ready','handshake').then(()=>true),sleep(4000).then(()=>false)]);
if(!connected){
 const status=await tab.evaluate("document.getElementById('transportStatus').textContent");
 assert.match(status,/missing|asset|metadata/i,'expected only the known missing-asset stop: '+status);
 assert.equal(await tab.evaluate("document.getElementById('endRunBtn').disabled"),true);
 await tab.evaluate("document.getElementById('endRunBtn').click()");await tab.screenshot(outputDir+'/stream-no-assets.png');
 const unrelated=errorsOf(tab).filter(e=>!/streaming\/data|metadata\.json/.test(JSON.stringify(e)));assert.deepEqual(unrelated,[],'only the missing-asset fetch may fail');await tab.close();
 console.log('BLOCKED: streaming assets (frontend/streaming/data) are absent locally, so the live pause/end-run flow could not be exercised on the hardware page; isolation from the simulation and the no-session guard were checked.');
}else{
 const fin=await driveSession(tab,'window','document',1512,{label:'stream'});
 assert.equal(fin.source,'live-hardware');assert.match(fin.model,/Model 2/);
 // Pause during an outstanding request with a held board: paused summary shows, then updates once; End Run then completes.
 await tab.evaluate('fakeBoard.hold=true;window.stopAt=Infinity');await tab.evaluate("document.getElementById('playBtn').click()");await until(tab,'!!fakeBoard.release','held request');
 await tab.evaluate("document.getElementById('playBtn').click()");await until(tab,'EdgeStreamingController.paused && !document.getElementById("sessionSummary").hidden','paused during request');
 assert.equal(await tab.evaluate('EdgeSessionRun.current.liveSummary.metrics.total'),0);assert.equal(await tab.evaluate('!!document.getElementById("ssResume")'),true,'Resume offered while paused with a request in flight');
 await tab.evaluate('fakeBoard.release()');await until(tab,'EdgeStreamingController.records.length===1 && EdgeSessionRun.current.liveSummary.metrics.total===1','paused summary updated with the final response');
 assert.equal(await tab.evaluate('ArmEEGDemo.getState().playback.results.length'),1,'its visualization is queued, not lost');
 await tab.evaluate("document.getElementById('endRunBtn').click()");await until(tab,'EdgeSessionRun.current.phase==="finalized"','finalized after held request');
 const held=await tab.evaluate('({total:EdgeSessionRun.current.summary.metrics.total,calls:fakeBoard.calls,pending:EdgeStreamingController.pending,eegUsed:EdgeSessionRun.current.summary.eeg.recordings.used,pill:document.querySelector(".ss-pill").textContent})');assert.equal(held.total,1);assert.equal(held.calls,8,'no new request after End Run');assert.equal(held.pending,null);assert.equal(held.eegUsed,1);assert.equal(held.pill,'Run complete');
 assert.deepEqual(errorsOf(tab),[]);await tab.close();
 console.log('PASS: live streaming right-panel views with a fake serial port: paused summary, Resume, End Run complete, Play blocked, New Run, pause during an outstanding request updated once; no simulation loaded; no errors.');
}
// 3. Legacy four-trial page keeps its controls hidden and its replay result.
const legacy=await page(base+'/?v=eeg-1');
assert.equal(await legacy.evaluate("document.getElementById('endRunBtn').hidden"),true);assert.equal(await legacy.evaluate("document.getElementById('newRunBtn').hidden"),true);assert.equal(await legacy.evaluate("document.getElementById('sessionSummary').hidden"),true);assert.equal(await legacy.evaluate('!!document.getElementById("simBanner")'),false);
await legacy.evaluate("document.getElementById('replayBtn').click()");await until(legacy,'ArmEEGDemo.getState().playback.complete && !ArmEEGDemo.getState().playing','legacy replay');
assert.equal(await legacy.evaluate("document.getElementById('accuracyValue').textContent"),'75%');assert.deepEqual(errorsOf(legacy),[]);await legacy.screenshot(outputDir+'/legacy-replay.png');await legacy.close();
console.log('PASS: legacy four-trial replay unchanged (75%, 3/4) with run controls hidden.');
