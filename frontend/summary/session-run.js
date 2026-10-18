/* End Run / New Run lifecycle and the right-panel summary for the continuous streaming session. No second inference
 * loop: this wraps one existing controller instance (frontend/streaming/controller.js or a compatible one) and only
 * calls its pause()/reset() plus the host's cancelVisualization(). Summaries are built from controller.records.
 *
 * Views of the right panel (summary-view.js renders into #sessionSummary inside the cursor panel):
 *   playing              summary hidden, prediction/cursor view visible.
 *   Run paused           controller paused (or stopped by an error -> "Run interrupted") after the first Play: a LIVE
 *                        summary rebuilt from the current records; nothing is finalized or frozen. A result that
 *                        completes after Pause updates it once; its visualization stays queued for Resume.
 *   Run complete         End Run finalized a frozen snapshot (status completed / empty) -> "Run complete";
 *                        an unresolved request at End Run -> "Run interrupted". Kept in history.
 *   New Run / Reset      controller.reset through the host flow; clears live and final state, hides the summary.
 *
 * Host contract (bridge.js passes this to EdgeSessionRun.attach after a verified handshake):
 *   controller           Controller with records, pending, task, failed, ready, seed, session, firmware,
 *                        pause(), play(), reset(seed), onState, onRecord, visualize, cancelVisualization.
 *   dataset              loaded dataset ({trials:[{trial_id, source_trial:{subject,session,...}}], model_sha256, channels(i)}).
 *   firmware             {model_sha256, build} from the READY handshake (defaults to controller.firmware).
 *   cancelVisualization  the same function given to the controller; resolves a blocked visualization.
 *   reset                host reset flow (must call controller.reset); defaults to controller.reset(seed).
 *   metadata             optional verified EdgeModelMetadata entry; resolved from firmware/dataset when omitted.
 *   eeg                  optional EEG provider {units, unitsVerified, samples(record), artifact(record)}.
 * On attach the manager wraps controller.play/reset/visualize/onState/onRecord on that instance only; detach() restores them.
 */
(function(root){'use strict';
const S=root.EdgeSessionSummary||(typeof require==='function'?require('./session-summary.js'):null);
const M=root.EdgeModelMetadata||(typeof require==='function'?require('./model-metadata.js'):null);
const E=root.EdgeEEGOverview||(typeof require==='function'?require('./eeg-overview.js'):null);
/* EEG provider default: the dataset's channels(index) plus the verified units from the model metadata entry. */
function defaultEEGProvider(dataset,meta){
 if(!dataset||typeof dataset.channels!=='function')return null;
 const units=meta?.eeg?.units??null;
 return {units,unitsVerified:!!units&&meta.eeg.unitsVerified!==false,samples:r=>{try{return dataset.channels(r.trial);}catch{return null;}},artifact:r=>typeof r.artifact==='boolean'?r.artifact:null};
}
const NO_EEG=Object.freeze({available:false,reason:'no EEG source for this session',label:'Prerecorded EEG · channel RMS',units:null,recordings:Object.freeze({total:0,withEEG:0,missing:0,used:0}),artifact:Object.freeze({known:false,flagged:0}),channels:Object.freeze([]),highest:null,lowest:null,excludedChannels:Object.freeze([]),sampleCount:0});
function explainReset(error){const text=error?.message||String(error);return /unresolved/i.test(text)?text+'. Disconnect UART, then Connect UART to start a new session; the captured results stay available for export.':text+'. Reconnect the streaming firmware before starting a new run.';}
function createRunManager({controller,dataset=null,firmware=null,metadata,eeg,cancelVisualization=null,reset=null,view=null,now=()=>Date.now(),clock=null,source='live-hardware',onPhase=()=>{}}){
 if(!controller)throw Error('Session run manager needs a controller');
 const meta=metadata!==undefined?metadata:M.resolve({firmware:firmware??controller.firmware??null,dataset});
 const provider=eeg!==undefined?eeg:defaultEEGProvider(dataset,meta);
 const newAccumulator=()=>E.createAccumulator({units:provider?.units??null,unitsVerified:!!provider?.unitsVerified});
 let eegAcc=newAccumulator(); // One pass per unique completed recording, at result time; nothing per animation frame.
 function accumulateEEG(r){if(!provider)return;const id=S.sourceIdentity(r,{datasetId:meta?.datasetId??null,dataset});if(id===null||eegAcc.has(id))return;let channels=null,artifact=null;try{channels=provider.samples(r);artifact=typeof provider.artifact==='function'?provider.artifact(r):null;}catch{channels=null;}eegAcc.add(id,channels,{artifact});}
 const tick=clock||(()=>typeof performance!=='undefined'&&performance.now?performance.now():Date.now());
 const cancel=()=>{if(typeof cancelVisualization==='function')cancelVisualization();else if(typeof controller.cancelVisualization==='function')controller.cancelVisualization();};
 let phase='idle',generation=0,startedAt=null,summary=null,liveSummary=null,liveKey=null,ending=null;const history=[];
 const pendingStart=new Map(),roundTrips=new Map();
 const original={onState:controller.onState,onRecord:controller.onRecord,visualize:controller.visualize,play:controller.play,reset:controller.reset,ownPlay:Object.prototype.hasOwnProperty.call(controller,'play'),ownReset:Object.prototype.hasOwnProperty.call(controller,'reset')};
 const key=(session,request)=>String(session)+':'+String(request);
 const setPhase=(p,detail)=>{phase=p;onPhase(p,detail);};
 const clear=()=>{startedAt=null;summary=null;liveSummary=null;liveKey=null;ending=null;pendingStart.clear();roundTrips.clear();eegAcc=newAccumulator();};
 /* Build a summary from the current records. status: 'paused' | 'interrupted' (live) or completed/empty/interrupted (final). */
 function snapshot(status){
  const endedAt=now(),pending=controller.pending??null,failed=controller.failed??null,records=controller.records||[],fw=firmware??controller.firmware??null;
  for(const r of records)accumulateEEG(r); // Covers records that arrived before attach; already-counted recordings are skipped.
  const results=S.adaptStreamingRecords(records,{datasetId:meta?.datasetId??null,dataset,roundTrips});
  const subjects=new Set((dataset?.trials??[]).map(t=>t?.source_trial?.subject).filter(x=>x!==undefined&&x!==null));
  const session={sessionId:controller.session!==undefined&&controller.session!==null?String(controller.session):null,modelId:meta?.modelId??null,modelName:meta?.modelName??null,experimentName:meta?.experimentName??null,datasetId:meta?.datasetId??null,datasetName:meta?.datasetName??null,
   subjectId:subjects.size===1?[...subjects][0]:(meta?.subjects?.length===1?meta.subjects[0]:null),modelHash:fw?.model_sha256??null,modelHashVerified:!!fw?.model_sha256,firmwareBuild:fw?.build??null,seed:controller.seed??null,source,status,
   interruption:status==='interrupted'?{message:failed?.message??'Hardware request unresolved',unresolvedRequest:pending?.request??null,trialId:pending?.trial?.trial_id??null}:null,
   note:status!=='interrupted'&&failed?'Connection stopped: '+failed.message:null,startedAt,endedAt,classLabels:meta?.classLabels,classNamesVerified:!!meta};
  return S.build(session,results,{eeg:provider?eegAcc.result():NO_EEG});
 }
 /* Right-panel view while the run is not finalized: hidden when playing or before the first Play; live summary when paused or stopped. */
 function syncLive(state){
  if(phase!=='idle')return;
  const started=startedAt!==null||(controller.records||[]).length>0;
  if(!started||(!state.paused&&!state.error)){if(liveKey!==null){liveKey=null;liveSummary=null;view?.hide?.();}return;}
  const status=state.error?'interrupted':'paused',k=status+':'+(controller.records||[]).length+':'+(state.error||'')+':'+(controller.pending?.request??'');
  if(k===liveKey)return;
  liveKey=k;liveSummary=snapshot(status);view?.render?.(liveSummary,{final:false});
 }
 controller.onState=s=>{if(s&&!s.paused&&!s.error&&startedAt===null)startedAt=now();const p=s?.pending;if(p&&p.request!==undefined&&p.request!==null){const k=key(controller.session,p.request);if(!pendingStart.has(k))pendingStart.set(k,tick());}if(typeof original.onState==='function')original.onState.call(controller,s);if(s)syncLive(s);};
 controller.onRecord=r=>{const k=key(r.session,r.request);if(pendingStart.has(k)){roundTrips.set(k,tick()-pendingStart.get(k));pendingStart.delete(k);}accumulateEEG(r);if(typeof original.onRecord==='function')original.onRecord.call(controller,r);};
 controller.visualize=(record,trial)=>phase==='idle'&&typeof original.visualize==='function'?original.visualize.call(controller,record,trial):Promise.resolve();
 controller.play=(...args)=>phase==='idle'?original.play.apply(controller,args):Promise.reject(Error(phase==='ending'?'End Run in progress · finishing the outstanding request':phase==='resetting'?'New Run in progress':'Run ended · use New Run to start another session'));
 controller.reset=async(...args)=>{generation++;setPhase('resetting');try{await original.reset.apply(controller,args);clear();view?.clear?.();setPhase('idle');}catch(e){setPhase(summary?'finalized':'idle');view?.message?.('New Run blocked',explainReset(e));throw e;}}; // A blocked reset keeps the ended run ended.
 function finalize(){
  const pending=controller.pending??null,records=controller.records||[];
  summary=snapshot(pending?'interrupted':(records.length?'completed':'empty'));history.push(summary);liveSummary=null;liveKey=null;setPhase('finalized');view?.render?.(summary,{final:true});return summary;
 }
 function endRun(){
  if(phase==='finalized'){view?.render?.(summary,{final:true});return Promise.resolve(summary);}
  if(phase==='ending')return ending;
  if(phase==='resetting')return Promise.resolve(null);
  const gen=generation,pending=controller.pending??null;setPhase('ending',{request:pending?.request??null});
  view?.ending?.({request:pending?.request??null,trialId:pending?.trial?.trial_id??null});
  controller.pause();cancel();
  ending=(async()=>{const task=controller.task;if(task){try{await task;}catch{}}if(gen!==generation||phase!=='ending')return null;return finalize();})();
  return ending;
 }
 async function newRun(){
  if(phase==='ending'){view?.ending?.({message:'End Run is still waiting for the outstanding request. New Run becomes available once it resolves.'});return false;}
  if(phase==='resetting')return false;
  try{if(typeof reset==='function')await reset();else await controller.reset(controller.seed);}catch{/* explained by the reset wrapper */}
  const cleared=phase==='idle'&&summary===null&&liveSummary===null;
  if(cleared&&!controller.ready)view?.message?.('New run ready · reconnect required','The previous results were archived, but the UART link is closed. Use Connect UART to verify the streaming firmware before Play. Inference does not start automatically.');
  return cleared;
 }
 function detach(){controller.onState=original.onState;controller.onRecord=original.onRecord;controller.visualize=original.visualize;if(original.ownPlay)controller.play=original.play;else delete controller.play;if(original.ownReset)controller.reset=original.reset;else delete controller.reset;}
 return {endRun,newRun,detach,get phase(){return phase;},get summary(){return summary;},get liveSummary(){return liveSummary;},get history(){return history.slice();},get metadata(){return meta;},get eegProvider(){return provider;},get startedAt(){return startedAt;},roundTrips};
}
const api={current:null,view:null,createRunManager,defaultEEGProvider,explainReset,
 attach(host){api.current?.detach();api.view?.clear?.();api.current=createRunManager({...host,view:api.view,onPhase:p=>api.syncButtons(p)});api.syncButtons('idle');return api.current;},
 get history(){return api.current?api.current.history:[];},
 syncButtons(phase){if(typeof document==='undefined')return;const end=document.getElementById('endRunBtn'),fresh=document.getElementById('newRunBtn');if(!end||!fresh)return;const attached=!!api.current;
  end.disabled=!attached||phase!=='idle';end.textContent=phase==='ending'?'… Ending run':'■ End Run';end.title=!attached?'Connect streaming firmware first':phase==='finalized'?'Run ended · use New Run to start another session':phase==='ending'?'Finishing the outstanding request':'Stop scheduling trials, finish the outstanding request and show the final summary';
  fresh.disabled=!attached||phase==='ending'||phase==='resetting';fresh.title=attached?'Archive this run, clear the summary and start a new seeded session (does not start inference)':'Connect streaming firmware first';},
 install(){
  if(typeof document==='undefined')return null;const $=id=>document.getElementById(id);const end=$('endRunBtn'),fresh=$('newRunBtn'),container=$('sessionSummary');
  if(!end||!fresh||!container||!root.EdgeSummaryView)return null;
  if(!(root.window||root).EdgeStreamingMode)return null; // Legacy four-trial replay keeps its original controls.
  api.view=root.EdgeSummaryView.create(container,{onNewRun:()=>api.current?.newRun()});
  end.hidden=false;fresh.hidden=false;api.syncButtons('idle');
  end.addEventListener('click',()=>{if(!api.current){api.view.message('No streaming session','Connect UART and verify the streaming firmware before ending a run.');return;}api.current.endRun();});
  fresh.addEventListener('click',()=>{api.current?.newRun();});
  return api.view;
 }};
root.EdgeSessionRun=api;
if(typeof module!=='undefined')module.exports=api;
if(typeof document!=='undefined')api.install(); // Browser only; a no-op outside streaming mode or without the run buttons.
})(globalThis);
