/* Offline protocol/controller fixtures only. No physical UART or model inference. */
'use strict';
const assert=require('node:assert/strict'),crypto=require('node:crypto'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.resolve(__dirname,'../..');
const original=require(root+'/frontend/streaming/protocol.js');
const {profile,protocol}=require(root+'/frontend/multisubject/profile.js');
const P=protocol(original);global.EdgeStreamProtocol=P;
const base=require(root+'/frontend/streaming/controller.js');
const {Controller}=require(root+'/frontend/multisubject/controller.js');
const {validateMetadata}=require(root+'/frontend/multisubject/dataset.js');
const bytes=Uint8Array.from({length:5632},(_,i)=>i%256),input_sha256=crypto.createHash('sha256').update(bytes).digest('hex');
const metadata={schema:2,model_sha256:profile.model_sha256,input_shape:[1,1,22,256],input_quant_scale:profile.input_quant_scale,input_zero_point:-12,sample_rate_hz:125,samples_per_channel:256,channel_order:[...profile.channel_order],assets:{'inputs.bin':'a'.repeat(64),'eeg.f32':'b'.repeat(64)},trials:Array.from({length:1080},(_,index)=>{const subject=1+Math.floor(index/120),session_trial=25+index%120,artifact=index<94;return {index,subject,trial_id:`S${subject}/1test/run0/cue${session_trial*1000}`,ground_truth:index%2,artifact,source_trial:{subject,session:'1test',run:0,cue_sample:session_trial*1000,session_trial,artifact},input_sha256};})};
const dataset={...metadata,input:async()=>bytes.slice()};
function ready(f,bad=false){const payload=new Uint8Array(69),v=new DataView(payload.buffer);payload.set(profile.model_sha256.match(/../g).map(x=>parseInt(x,16)));if(bad)payload[0]^=1;v.setFloat32(32,profile.input_quant_scale,true);v.setInt32(36,-12,true);v.setUint32(40,5632,true);profile.input_shape.forEach((n,i)=>v.setUint16(44+i*2,n,true));payload[52]=1;payload.set(new TextEncoder().encode('SIMULATION-ONLY'),53);return {...f,type:P.T.READY,payload};}
class SimulatedSerial {
 constructor(){this.sent=[];this.requests=[];this.gate=null;this.badModel=false;this.corruptCRC=false;this.connected=true;this.onData=()=>{};this.onDisconnect=()=>{};}
 send(f){const b=P.encode(f);for(let i=0;i<b.length;i+=7)this.onData(b.subarray(i,i+7));}
 async write(b){this.sent.push(b);if(b.length===1)return;const f=P.decode(b.subarray(0,-1));if(f.type===P.T.HELLO){this.send(ready(f,this.badModel));return;}
  assert.equal(f.type,P.T.INFER);assert.equal(f.payload.length,5632);assert(f.trial<1080);assert.deepEqual(f.payload,bytes);this.requests.push(f);if(this.gate)await this.gate;
  const checksum=P.crc(f.payload),ack=new Uint8Array(4);new DataView(ack.buffer).setUint32(0,this.corruptCRC?checksum^1:checksum,true);this.send({...f,type:P.T.ACCEPTED,payload:ack});
  const result=new Uint8Array(16),v=new DataView(result.buffer);v.setUint32(0,checksum,true);v.setFloat32(4,-1,true);v.setFloat32(8,2,true);result[12]=1;this.send({...f,type:P.T.RESULT,payload:result});
 }
}
const pause=()=>new Promise(r=>setImmediate(r));
async function until(f){for(let i=0;i<300;i++){if(f())return;await pause();}throw Error('Test condition timed out');}
(async()=>{
 validateMetadata(metadata);
 for(const change of [m=>m.trials[0].source_trial.session_trial=24,m=>m.trials[0].subject=9,m=>m.trials[0].ground_truth=3,m=>m.trials[0].artifact=false,m=>m.trials[0].trial_id='wrong',m=>m.input_quant_scale=0.1,m=>m.model_sha256=original.model]){const m=structuredClone(metadata);change(m);assert.throws(()=>validateMetadata(m));}
 assert.equal(original.model,'644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298','Original Model 2 profile is unchanged');
 const hello={session:1n,request:0,trial:0xffffffff};assert.equal(P.ready(ready(hello)).id,profile.id);assert.throws(()=>original.ready(ready(hello)),/mismatch/);assert.throws(()=>P.ready(ready(hello,true)),/mismatch/);
 const wire=P.encode({type:P.T.INFER,session:1n,request:1,trial:1079,payload:bytes});for(let split=0;split<wire.length;split+=73){const accepted=[];const parser=new P.Parser(f=>accepted.push(f));parser.push(wire.subarray(0,split));parser.push(wire.subarray(split));assert.equal(accepted.length,1);assert.deepEqual(accepted[0].payload,bytes);}
 let c,t=new SimulatedSerial(),clock=0;c=new Controller({dataset,transport:t,clock:()=>++clock,visualize:async()=>{},onRecord:()=>{if(c.records.length===1081)c.pause();}});
 await c.handshake();await c.play();assert.equal(c.failed,null);assert.equal(c.records.length,1081);assert.equal(c.state().unique,1080);assert.equal(c.completedRecords.filter(r=>r.repeated).length,1);assert.equal(new Set(c.completedRecords.map(r=>r.subject_id)).size,9);assert.equal(new Set(c.records.slice(0,1080).map(r=>r.trial_id)).size,1080);
 for(const r of c.completedRecords){assert.equal(r.ground_truth,metadata.trials[r.trial].ground_truth);assert.deepEqual(r.logits,[-1,2]);assert.equal(r.class_id,1);assert.equal(r.model_id,profile.id);assert.equal(r.input_crc32,P.crc(bytes));assert.equal(r.timing.hardware_inference_ms,null);assert(r.timing.host_round_trip_ms>=r.timing.host_ack_to_result_ms);assert(Object.isFrozen(r));assert(Object.isFrozen(r.logits));}
 // One in-flight response survives Pause; Resume continues the existing sequence.
 let release;t.gate=new Promise(r=>release=r);const running=c.play();await until(()=>c.pending);const before=c.records.length,position=c.sequence.position;c.pause();release();await running;assert.equal(c.records.length,before+1);assert.equal(c.sequence.position,position);assert.equal(c.pending,null);t.gate=null;
 // End Run waits for exactly one in-flight result, cancels visualization, then blocks Play.
 let done;t.gate=new Promise(r=>done=r);c.play();await until(()=>c.pending);const ending=c.endRun();done();const summary=await ending;assert.equal(summary.state.ended,true);assert.equal(c.pending,null);await assert.rejects(c.play(),/Run ended/);const old=c.records.length;await c.reset(991);assert.equal(c.records.length,0);assert.equal(c.completedArchives[0].length,old);assert.equal(c.sequence.position,0);assert.equal(c.state().ended,false);
 const mismatchTransport=new SimulatedSerial();mismatchTransport.badModel=true;const mismatch=new Controller({dataset,transport:mismatchTransport});await assert.rejects(mismatch.handshake(),/mismatch/);assert.equal(mismatch.records.length,0);assert.equal(mismatchTransport.requests.length,0);
 const invalid=new SimulatedSerial();invalid.corruptCRC=true;const bad=new Controller({dataset,transport:invalid});await bad.handshake();await bad.play();assert.match(bad.failed.message,/checksum mismatch/);assert.equal(bad.records.length,0);assert.equal(invalid.requests.length,1);
 const noResponse=new Controller({dataset,transport:{write:async()=>{}},ackMs:5});await assert.rejects(noResponse.handshake(),/timeout/);assert.equal(noResponse.records.length,0);
 // Pure selection has no subject/trial count cap and is reproducible across cycles.
 const a=new base.Sequence(metadata.trials.map(t=>t.trial_id),112),b=new base.Sequence(metadata.trials.map(t=>t.trial_id),112);for(let i=0;i<1100;i++)assert.deepEqual(await a.next(),await b.next());
 console.log('PASS: multi-subject profile/data eligibility, full 1080-trial cycle plus reshuffle, fragmented all-byte serialization, strict handshake/CRC, actual-response-only records, identity/ground truth, pause/in-flight, End Run, Reset archives, unique/repeat and explicit timing. Simulated serial only.');
})().catch(e=>{console.error(e);process.exitCode=1;});
