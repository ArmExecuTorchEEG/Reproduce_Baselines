'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),{spawn,execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),P=require(root+'/frontend/streaming/protocol.js'),{Controller,Sequence,hash}=require(root+'/frontend/streaming/controller.js');
const binary=process.env.EDGE_STREAM_NATIVE||'/private/tmp/edgebci-stream/parser-tests';
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function until(f){for(let i=0;i<1000;i++){if(f())return;await pause(2);}throw Error('Timed out');}
class PipeSerial{
 constructor(){this.child=spawn(binary,['--serve']);this.child.stdout.on('data',b=>{for(let i=0;i<b.length;i+=7)this.onData(new Uint8Array(b.subarray(i,i+7)));});this.child.on('exit',()=>this.onDisconnect());this.onData=()=>{};this.onDisconnect=()=>{};this.sent=[];this.gate=null;}
 async write(bytes){this.sent.push(bytes);if(this.gate&&bytes.length>1000)await this.gate;await new Promise((ok,no)=>this.child.stdin.write(bytes,e=>e?no(e):ok()));}
 close(){this.onDisconnect=()=>{};this.child.stdin.end();}
}
(async()=>{
 assert.equal(P.crc(new TextEncoder().encode('123456789')),0xcbf43926);
 const payload=Uint8Array.from({length:11000},(_,i)=>i%256),wire=P.encode({type:P.T.INFER,session:1n,request:1,trial:0,payload});
 for(let i=0;i<wire.length;i+=137){let count=0;const parser=new P.Parser(f=>{assert.deepEqual(f.payload,payload);count++;});parser.push(wire.subarray(0,i));parser.push(wire.subarray(i));assert.equal(count,1);}
 let now=0,count=0,errors=0;const parser=new P.Parser(()=>count++,()=>errors++,()=>now);parser.push(wire.subarray(0,100));now=1000;parser.tick();parser.push(wire.subarray(100));assert.equal(count,0);parser.push(wire);assert.equal(count,1);assert.equal(errors,1);
 const metadata=JSON.parse(fs.readFileSync(root+'/frontend/streaming/data/metadata.json')),inputs=fs.readFileSync(root+'/frontend/streaming/data/inputs.bin');
 const dataset={...metadata,input:async i=>new Uint8Array(inputs.subarray(i*11000,(i+1)*11000))};
 const seq=new Sequence(dataset.trials.map(t=>t.trial_id),42),saved=fs.readFileSync(root+'/artifacts/model2-session2-phase1/selection-seed42-300.jsonl','utf8').trim().split('\n').map(JSON.parse);
 for(let i=0;i<300;i++)assert.equal((await seq.next()).id,saved[i].trial_id,'JS/Python selection order');
 const transport=new PipeSerial();let controller;controller=new Controller({dataset,transport,visualize:async()=>{},onRecord:()=>{if(controller.records.length===160)controller.pause();}});await controller.handshake();await controller.play();assert.equal(controller.records.length,160);assert.equal(controller.state().unique,144);assert.equal(controller.state().total,160);assert.equal(controller.failed,null);for(const r of controller.records){const b=await dataset.input(r.trial);const a=new Int8Array(b.buffer);assert.deepEqual(r.logits,[a[0],a[10999]],'Explicit native MOCK forward returns input endpoints, not a real model prediction');}
 // Pause during an upload: the result is retained, no next request is scheduled.
 let release;transport.gate=new Promise(ok=>release=ok);const running=controller.play();await until(()=>controller.pending!==null);controller.pause();const before=controller.records.length;release();await running;assert.equal(controller.records.length,before+1);assert.equal(controller.pending,null);
 const old=controller.records.length;await controller.reset(1234);assert.equal(controller.records.length,0);assert.equal(controller.archives[0].length,old);assert.equal(controller.sequence.position,0);
 // Corrupt or missing replies never turn into fabricated predictions.
 transport.close();
 const silent={write:async()=>{}};const missing=new Controller({dataset,transport:silent,ackMs:10});await assert.rejects(missing.handshake(),/timeout/);assert.equal(missing.records.length,0);
 const badDataset={...dataset,input:async()=>new Uint8Array(11000)};const t2=new PipeSerial();const corrupt=new Controller({dataset:badDataset,transport:t2});await corrupt.handshake();await corrupt.play();assert.match(corrupt.failed.message,/hash mismatch/);assert.equal(corrupt.records.length,0);t2.close();
 console.log('PASS: JS/C++ binary pipe interoperability; fragmentation, receive timeout, all-byte payloads, Python/JS seeded selection, 160 real-input mock executions over reshuffle, pause/in-flight retention, reset/archive, missing board and integrity failure. No physical serial port used.');
})().catch(e=>{console.error(e);process.exit(1);});
