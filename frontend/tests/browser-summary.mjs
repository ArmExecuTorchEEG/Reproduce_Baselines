/* Session summary in a real (isolated, headless) Chrome: synthetic preview inside the full GUI, streaming mode with a
 * fake in-page serial board (pause/resume/end-run views in the right panel), and the untouched legacy page. No physical port. */
import {page} from './browser-cdp.mjs';import assert from 'node:assert/strict';import fs from 'node:fs/promises';import path from 'node:path';
const base=process.env.EDGEBCI_BASE||'http://127.0.0.1:8000',outputDir=path.resolve(process.env.EDGEBCI_QA_DIR||'/tmp/edgebci-summary-qa');await fs.mkdir(outputDir,{recursive:true});
const sleep=ms=>new Promise(ok=>setTimeout(ok,ms));
const errorsOf=tab=>tab.events.filter(e=>e.method==='Runtime.exceptionThrown'||e.method==='Network.loadingFailed'||(e.method==='Log.entryAdded'&&e.params.entry.level==='error'));
async function until(tab,expr,label=expr){for(let i=0;i<300;i++){const x=await tab.evaluate(expr);if(x)return x;await sleep(60);}throw Error('Timed out: '+label);}
const PANELS='(()=>{const r={};for(const s of [".cursor-panel",".brain-panel",".drawer",".top-grid",".controls"]){const b=D.querySelector(s).getBoundingClientRect();r[s]=[b.x,b.y,b.width,b.height];}return r;})()';
// 1. Synthetic preview: the real GUI (mock mode) in a frame, right panel switched between prediction view and synthetic summaries.
for(const [width,height,suffix] of [[1512,950,'desktop'],[390,844,'phone']]){
 const tab=await page(base+'/summary/preview.html',width,height,'document.readyState==="complete" && window.previewReady===true && !!window.previewDocument');
 const D='window.previewDocument';
 await tab.evaluate('previewShow("completed","playing")');await sleep(100);
 assert.equal(await tab.evaluate(`${D}.getElementById("sessionSummary").hidden`),true,'prediction view while playing');
 const before=await tab.evaluate(PANELS.replaceAll('D.',D+'.'));
 for(const key of ['completed','interrupted','empty','multisubject']){
  for(const mode of ['paused','complete']){
   await tab.evaluate(`previewShow(${JSON.stringify(key)},${JSON.stringify(mode)})`);await sleep(150);
   const s=await tab.evaluate(`(()=>{const doc=${D},box=doc.getElementById("sessionSummary"),wrap=doc.querySelector(".ss-matrix-wrap"),card=doc.querySelector(".ss-body");return {hidden:box.hidden,status:previewSummary.session.status,total:previewSummary.metrics.total,acc:previewSummary.metrics.accuracy,conf:previewSummary.metrics.confusion,pill:doc.querySelector(".ss-pill").textContent,overflow:doc.documentElement.scrollWidth>doc.documentElement.clientWidth,clip:box.scrollWidth>box.clientWidth+1,matrixScroll:wrap?wrap.scrollWidth>wrap.clientWidth+1:false,text:card.textContent,eeg:doc.querySelector(".ss-eeg").textContent,eegAvailable:previewSummary.eeg.available,ranks:doc.querySelectorAll(".ss-rank").length,matrix:doc.querySelector(".ss-matrix")?.textContent,diag:doc.querySelectorAll(".ss-diag").length,off:doc.querySelectorAll(".ss-off").length,newRun:!!doc.getElementById("ssNewRun"),close:!!doc.getElementById("ssClose"),font:getComputedStyle(box).fontFamily,inPanel:box.parentElement.classList.contains("cursor-panel")};})()`);
   assert.equal(s.hidden,false);assert.equal(s.inPanel,true,'summary lives inside the cursor panel');assert.match(s.font,/Helvetica Neue/);assert.equal(s.close,false,'no close control');assert.equal(s.newRun,mode==='complete','New Run only on the finalized view');
   assert.equal(s.overflow,false,'no horizontal overflow at '+width);assert.equal(s.clip,false,'summary not clipped at '+width);assert.equal(s.matrixScroll,false,'matrix fits at '+width);
   assert.doesNotMatch(s.text,/ynthetic|validated hardware results|Model hash|Session ID|Elapsed|Completed trials|Back to live view/);
   assert.match(s.eeg,/Top EEG Channels/);assert.match(s.eeg,/RMS \(µV\)/);assert.match(s.eeg,/Amplitude ranking does not indicate neural activation/);assert.match(s.matrix,/Actual ↓/);assert.match(s.matrix,/Predicted →/);assert.equal(s.diag,2);assert.equal(s.off,2);assert.match(s.text,/Trials/);assert.match(s.text,/Duration/);
   if(mode==='paused')assert.equal(s.pill,key==='interrupted'?'Run interrupted':'Run paused');else assert.equal(s.pill,key==='interrupted'?'Run interrupted':'Run complete');
   if(key==='empty'){assert.equal(s.total,0);assert.equal(s.acc,null);assert.equal(s.eegAvailable,false);assert.match(s.eeg,/EEG amplitude unavailable/);assert.equal(s.ranks,0);assert.match(s.matrix,/Unavailable/);}
   else{assert.equal(s.eegAvailable,true);assert.equal(s.ranks,3);assert.match(s.eeg,/µV/);}
   if(key==='completed'){assert.deepEqual(s.conf,[[17,3],[7,13]]);assert.equal(s.acc,0.75);assert.match(s.matrix,/Left hand85% correct/);assert.match(s.matrix,/Right hand65% correct/);assert.match(s.matrix,/17Correct left/);assert.match(s.matrix,/3Incorrect left/);assert.match(s.matrix,/7Incorrect right/);assert.match(s.matrix,/13Correct right/);assert.match(s.text,/30 unique recordings · 10 repeat evaluations/);assert.match(s.text,/Includes repeats\./);}
   if(key==='multisubject'){
    assert.deepEqual(s.conf,[[35,12],[15,34]]);
    assert.deepEqual(await tab.evaluate(`Array.from(${D}.querySelectorAll(".ss-matrix thead th:not(.ss-corner)"),e=>e.textContent)`),['Left hand','Right hand']);
    assert.deepEqual(await tab.evaluate(`Array.from(${D}.querySelectorAll(".ss-matrix tbody th span"),e=>e.textContent)`),['Left hand','Right hand']);
    assert.deepEqual(await tab.evaluate(`Array.from(${D}.querySelectorAll(".ss-matrix td"),e=>e.textContent)`),['34Correct left','15Incorrect left','12Incorrect right','35Correct right']);
    assert.equal(await tab.evaluate('previewSummary.eeg.channels.length'),8);
    const eegBox=await tab.evaluate(`(()=>{const c=getComputedStyle(${D}.querySelector(".ss-eeg"));return {border:c.borderTopWidth,shadow:c.boxShadow,radius:c.borderTopLeftRadius};})()`);assert.deepEqual(eegBox,{border:'0px',shadow:'none',radius:'0px'});
    if(width>=1000){const [h1,h2]=await tab.evaluate(`[${D}.querySelector(".ss-matrix-panel .ss-h3").getBoundingClientRect().top,${D}.querySelector(".ss-eeg .ss-h3").getBoundingClientRect().top]`);assert(Math.abs(h1-h2)<1,'headings share one baseline on desktop');}
    else{const [mBottom,eTop]=await tab.evaluate(`[${D}.querySelector(".ss-matrix-panel").getBoundingClientRect().bottom,${D}.querySelector(".ss-eeg").getBoundingClientRect().top]`);assert(eTop>=mBottom,'EEG below the matrix on narrow screens');}
   }
   if(key==='interrupted')assert.match(s.eeg,/without EEG/);
   assert.deepEqual(await tab.evaluate(PANELS.replaceAll('D.',D+'.')),before,'switching views never moves or resizes the panels');
   await tab.screenshot(`${outputDir}/preview-${key}-${mode}-${suffix}.png`);
  }
 }
 await tab.evaluate('previewShow("completed","playing")');await sleep(80);assert.equal(await tab.evaluate(`${D}.getElementById("sessionSummary").hidden`),true);
 assert.match(await tab.evaluate("document.getElementById('syntheticLabel').textContent"),/SYNTHETIC PREVIEW DATA — not hardware results/);
 assert.deepEqual(errorsOf(tab),[]);await tab.close();
}
console.log('PASS: synthetic preview inside the full GUI: paused and complete views for four scenarios at desktop and phone widths, panels fixed, no overflow, no errors.');
// 2. Streaming mode with a fake serial board injected into the page (same approach as tests/streaming/browser-stream.mjs).
const tab=await page(base+'/?stream=1');
await until(tab,'!!window.EdgeStream && !!window.EdgeSessionRun && !!ArmEEGDemo.setPlaying');
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
 console.log('BLOCKED: streaming assets (frontend/streaming/data) are absent locally, so the live pause/end-run flow could not be exercised in the browser; only the no-session guard was checked.');
}else{
 const D='document';const panelsBefore=await tab.evaluate(PANELS.replaceAll('D.',D+'.'));
 assert.equal(await tab.evaluate("document.getElementById('sessionSummary').hidden"),true,'prediction view before Play');
 await tab.evaluate('(()=>{const c=EdgeStreamingController,old=c.onRecord;window.stopAt=5;c.onRecord=r=>{old(r);if(c.records.length>=window.stopAt)c.pause();};})()');
 await tab.evaluate("document.getElementById('playBtn').click()");await until(tab,'EdgeStreamingController.records.length===5 && EdgeStreamingController.paused','five results then pause');
 await until(tab,'!document.getElementById("sessionSummary").hidden','paused summary');
 let s=await tab.evaluate('(()=>{const m=EdgeSessionRun.current.liveSummary;return {pill:document.querySelector(".ss-pill").textContent,total:m.metrics.total,records:EdgeStreamingController.records.length,final:EdgeSessionRun.current.summary,phase:EdgeSessionRun.current.phase,results:ArmEEGDemo.getState().playback.results.length,hint:document.querySelector(".ss-hint")?.textContent,newRun:!!document.getElementById("ssNewRun")};})()');
 assert.equal(s.pill,'Run paused');assert.equal(s.total,5);assert.equal(s.total,s.records);assert.equal(s.final,null,'pause does not finalize');assert.equal(s.phase,'idle');assert.equal(s.results,5,'playback queue intact under the summary');assert.match(s.hint,/Press Play to resume/);assert.equal(s.newRun,false);
 assert.deepEqual(await tab.evaluate(PANELS.replaceAll('D.',D+'.')),panelsBefore,'no layout jump when the right panel switches');
 await tab.screenshot(outputDir+'/stream-paused.png');
 // Resume: prediction view returns and playback continues; pause again shows the updated live summary.
 await tab.evaluate('window.stopAt=7');await tab.evaluate("document.getElementById('playBtn').click()");await until(tab,'document.getElementById("sessionSummary").hidden && !EdgeStreamingController.paused','resumed');
 await until(tab,'EdgeStreamingController.records.length===7 && EdgeStreamingController.paused && !document.getElementById("sessionSummary").hidden','second pause');
 assert.equal(await tab.evaluate('EdgeSessionRun.current.liveSummary.metrics.total'),7);assert.equal(await tab.evaluate('ArmEEGDemo.getState().playback.results.length'),7,'queued visualizations retained across pause/resume');
 // End Run while paused with a queued visualization: finalizes without waiting for the animation.
 await tab.evaluate("document.getElementById('endRunBtn').click()");await until(tab,'EdgeSessionRun.current.phase==="finalized"','finalized');
 s=await tab.evaluate('(()=>{const m=EdgeSessionRun.current.summary,c=EdgeStreamingController;return {pill:document.querySelector(".ss-pill").textContent,total:m.metrics.total,correct:m.metrics.correct,records:c.records.length,recCorrect:c.records.filter(r=>r.class_id===r.ground_truth).length,unique:m.metrics.unique.count,stateUnique:c.state().unique,hash:m.session.modelHash,newRun:!!document.getElementById("ssNewRun"),eeg:{available:m.eeg.available,used:m.eeg.recordings.used,units:m.eeg.units,n:m.eeg.channels.length},order:c.dataset.channel_order,ranks:document.querySelectorAll(".ss-rank").length,text:document.querySelector(".ss-body").textContent,endDisabled:document.getElementById("endRunBtn").disabled};})()');
 assert.equal(s.pill,'Run complete');assert.equal(s.total,7);assert.equal(s.total,s.records);assert.equal(s.correct,s.recCorrect,'summary matches controller.records, not the animated counter');assert.equal(s.unique,s.stateUnique);assert.equal(s.hash.length,64);assert.equal(s.newRun,true);assert.equal(s.endDisabled,true);
 assert.equal(s.eeg.available,true);assert.equal(s.eeg.used,s.unique);assert.equal(s.eeg.units,'µV');assert.equal(s.eeg.n,s.order.length);assert.equal(s.ranks,3);
 assert.match(s.text,/Confusion Matrix.*Correct left.*Incorrect left.*Incorrect right.*Correct right.*Top EEG Channels.*RMS \(µV\).*Prerecorded EEG/);assert.doesNotMatch(s.text,/Model 2|Model hash|Session ID|round trip|ynthetic|Back to live view/);
 assert.deepEqual(await tab.evaluate(PANELS.replaceAll('D.',D+'.')),panelsBefore);
 await tab.screenshot(outputDir+'/stream-complete.png');
 await tab.evaluate("document.getElementById('playBtn').click()");await sleep(200);assert.equal(await tab.evaluate('EdgeStreamingController.paused'),true,'Play after End Run does not restart');assert.match(await tab.evaluate("document.getElementById('transportStatus').textContent"),/Run ended/);
 // New Run from the panel: reset flow, prediction view restored, nothing starts.
 await tab.evaluate("document.getElementById('ssNewRun').click()");await until(tab,'EdgeStreamingController.records.length===0 && EdgeSessionRun.current.phase==="idle"','new run');
 assert.equal(await tab.evaluate("document.getElementById('sessionSummary').hidden"),true);assert.equal(await tab.evaluate('EdgeStreamingController.archives[0].length'),7);assert.equal(await tab.evaluate('EdgeStreamingController.paused'),true,'New Run does not start inference');
 // Pause during an outstanding request: paused summary shows, then updates once when the valid response arrives; End Run then completes.
 await tab.evaluate('fakeBoard.hold=true;window.stopAt=Infinity');await tab.evaluate("document.getElementById('playBtn').click()");await until(tab,'!!fakeBoard.release','held request');
 await tab.evaluate("document.getElementById('playBtn').click()");await until(tab,'EdgeStreamingController.paused && !document.getElementById("sessionSummary").hidden','paused during request');
 assert.equal(await tab.evaluate('EdgeSessionRun.current.liveSummary.metrics.total'),0);assert.equal(await tab.evaluate('document.querySelector(".ss-pill").textContent'),'Run paused');
 await tab.evaluate('fakeBoard.release()');await until(tab,'EdgeStreamingController.records.length===1 && EdgeSessionRun.current.liveSummary.metrics.total===1','paused summary updated with the final response');
 assert.equal(await tab.evaluate('ArmEEGDemo.getState().playback.results.length'),1,'its visualization is queued, not lost');assert.equal(await tab.evaluate('EdgeSessionRun.current.phase'),'idle');
 await tab.evaluate("document.getElementById('endRunBtn').click()");await until(tab,'EdgeSessionRun.current.phase==="finalized"','finalized after held request');
 const held=await tab.evaluate('({total:EdgeSessionRun.current.summary.metrics.total,calls:fakeBoard.calls,pending:EdgeStreamingController.pending,eegUsed:EdgeSessionRun.current.summary.eeg.recordings.used,pill:document.querySelector(".ss-pill").textContent})');assert.equal(held.total,1);assert.equal(held.calls,8,'no new request after End Run');assert.equal(held.pending,null);assert.equal(held.eegUsed,1);assert.equal(held.pill,'Run complete');
 assert.deepEqual(errorsOf(tab),[]);await tab.close();
 console.log('PASS: live streaming right-panel views: paused summary (no finalize, playback intact, no layout jump), resume, End Run complete, Play blocked, New Run, pause during an outstanding request updated once; no errors.');
}
// 3. Legacy four-trial page keeps its controls hidden and its replay result.
const legacy=await page(base+'/?v=eeg-1');
assert.equal(await legacy.evaluate("document.getElementById('endRunBtn').hidden"),true);assert.equal(await legacy.evaluate("document.getElementById('newRunBtn').hidden"),true);assert.equal(await legacy.evaluate("document.getElementById('sessionSummary').hidden"),true);
await legacy.evaluate("document.getElementById('replayBtn').click()");await until(legacy,'ArmEEGDemo.getState().playback.complete && !ArmEEGDemo.getState().playing','legacy replay');
assert.equal(await legacy.evaluate("document.getElementById('accuracyValue').textContent"),'75%');assert.deepEqual(errorsOf(legacy),[]);await legacy.screenshot(outputDir+'/legacy-replay.png');await legacy.close();
console.log('PASS: legacy four-trial replay unchanged (75%, 3/4) with run controls hidden.');
