/* SYNTHETIC DEMONSTRATION DATA for summary/preview.html only.
 * Results and EEG samples are generated from fixed pseudo-random seeds; no EEG recording, board, UART or model is
 * involved. Never loaded by index.html. */
(function(root){'use strict';
function lcg(seed){let s=seed>>>0;return()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return s/4294967296;};}
const CH22=['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'];
const CH8=['C3','Cz','C4','FC1','FC2','CP1','CP2','Pz'];
function results({seed,subjects,recordings,trials,accuracy,classes,session='1test',dataset='synthetic-preview'}){
 const rnd=lcg(seed),out=[];
 for(let n=0;n<trials;n++){const subject=subjects[Math.floor(rnd()*subjects.length)],rec=n<recordings?n:Math.floor(rnd()*recordings),truth=classes[rec%classes.length],correct=rnd()<accuracy,pred=correct?truth:classes[(classes.indexOf(truth)+1)%classes.length];
  const margin=0.6+rnd()*3,logits=classes.map(c=>c===pred?margin:-margin);
  out.push({trialId:`synthetic-session:${n+1}`,sourceTrialId:`${dataset}|S${subject}/${session}/run${rec%6}/cue${1000+rec*37}`,subjectId:subject,groundTruth:truth,predictedClass:pred,logits,roundTripMs:1050+Math.round(rnd()*180)});}
 return out;
}
/* One synthetic EEG window per unique recording: sinusoids plus noise with a fixed per-channel amplitude. */
function eeg(rows,{seed,names,samples=250,missingEvery=0,artifactEvery=0}){
 const rnd=lcg(seed),amp=Object.fromEntries(names.map((n,i)=>[n,2+((i*7)%11)*1.1+rnd()*2])),seen=new Set(),out=[];
 for(const r of rows){if(seen.has(r.sourceTrialId))continue;seen.add(r.sourceTrialId);const k=seen.size;
  if(missingEvery&&k%missingEvery===0){out.push({sourceTrialId:r.sourceTrialId,channels:null,artifact:false});continue;}
  const channels={};for(const n of names){const a=amp[n]*(0.8+rnd()*0.4),f=8+rnd()*6,ph=rnd()*6.28,x=new Array(samples);for(let j=0;j<samples;j++)x[j]=a*Math.SQRT2*Math.sin(j/250*6.283*f+ph)+(rnd()-0.5)*a*0.3;channels[n]=x;}
  out.push({sourceTrialId:r.sourceTrialId,channels,artifact:artifactEvery?k%artifactEvery===0:false});}
 return out;
}
/* Deterministic outcomes: per actual class {correct, wrong}; recordings repeat after `recordings` unique ones. */
function explicit({perClass,recordings,subject=1,session='1test',dataset='synthetic-preview'}){
 const classes=Object.keys(perClass).map(Number),queues={};for(const c of classes){const q=[],{correct,wrong}=perClass[c];for(let i=0;i<correct+wrong;i++)q.push(wrong&&i%Math.ceil((correct+wrong)/wrong)===Math.ceil((correct+wrong)/wrong)-1&&q.filter(x=>!x).length<wrong?false:true);while(q.filter(x=>!x).length<wrong)q[q.lastIndexOf(true)]=false;queues[c]=q;}
 const total=Object.values(queues).reduce((s,q)=>s+q.length,0),out=[];
 for(let n=0;n<total;n++){const rec=n<recordings?n:n-recordings,truth=classes[rec%classes.length],q=queues[truth];if(!q.length){n--;continue;}const ok=q.shift(),pred=ok?truth:classes[(classes.indexOf(truth)+1)%classes.length];
  out.push({trialId:`synthetic-session:${out.length+1}`,sourceTrialId:`${dataset}|S${subject}/${session}/run${rec%6}/cue${1000+rec*37}`,subjectId:subject,groundTruth:truth,predictedClass:pred,logits:classes.map(c=>c===pred?1.4:-1.4),roundTripMs:1100});}
 return out;
}
const base={source:'synthetic-preview',modelName:'Synthetic preview model (not a deployed model)',modelId:'synthetic-preview',experimentName:'Layout review with generated numbers',datasetName:'Synthetic dataset · generated in the browser',datasetId:'synthetic-preview',firmwareBuild:null,modelHash:null,seed:12345};
const classesLR={0:{name:'left_hand',display:'Left hand',short:'Left'},1:{name:'right_hand',display:'Right hand',short:'Right'}};
// 40 completed trials over 30 unique recordings (10 repeats): actual left 17 correct / 3 wrong, actual right 13 correct / 7 wrong → 75% overall.
const completed=explicit({perClass:{0:{correct:17,wrong:3},1:{correct:13,wrong:7}},recordings:30});
const interrupted=results({seed:11,subjects:[1],recordings:11,trials:11,accuracy:.64,classes:[0,1]});
const multi=results({seed:23,subjects:[1,3,7],recordings:60,trials:96,accuracy:.68,classes:[0,1]});
root.EdgeSummaryPreviewData={
 eegUnits:{units:'µV',unitsVerified:true},
 scenarios:{
  completed:{label:'Completed session',session:{...base,sessionId:'synthetic-0001',subjectId:1,classLabels:classesLR,startedAt:Date.UTC(2026,9,18,10,0,0),endedAt:Date.UTC(2026,9,18,10,7,42)},results:completed,eeg:eeg(completed,{seed:3,names:CH22,artifactEvery:9})},
  interrupted:{label:'Interrupted session',session:{...base,sessionId:'synthetic-0002',subjectId:1,classLabels:classesLR,status:'interrupted',interruption:{message:'Hardware result timeout; request unresolved',unresolvedRequest:12,trialId:'S1/1test/run3/cue1407'},startedAt:Date.UTC(2026,9,18,11,0,0),endedAt:Date.UTC(2026,9,18,11,2,5)},results:interrupted,eeg:eeg(interrupted,{seed:5,names:CH22,missingEvery:5,artifactEvery:4})},
  empty:{label:'Empty session',session:{...base,sessionId:'synthetic-0003',classLabels:classesLR,startedAt:null,endedAt:Date.UTC(2026,9,18,12,0,0)},results:[],eeg:[]},
  multisubject:{label:'Multi-subject model (reversed class map, 8 channels)',session:{...base,sessionId:'synthetic-0004',modelName:'Synthetic multi-subject preview (not a deployed model)',experimentName:'Three synthetic subjects · class map reversed on purpose',subjectId:null,classLabels:{0:{name:'right_hand',display:'Right hand',short:'Right',order:1},1:{name:'left_hand',display:'Left hand',short:'Left',order:0}},startedAt:Date.UTC(2026,9,18,13,0,0),endedAt:Date.UTC(2026,9,18,13,25,13)},results:multi,eeg:eeg(multi,{seed:9,names:CH8})}
 }
};
})(globalThis);
