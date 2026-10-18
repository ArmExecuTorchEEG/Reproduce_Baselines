/* Real wire bytes through the portable C++ server; explicit test-only toy inference. */
'use strict';
const assert=require('node:assert/strict'),path=require('node:path'),{spawn}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),original=require(root+'/frontend/streaming/protocol.js');
const {profile,protocol}=require(root+'/frontend/multisubject/profile.js');global.EdgeStreamProtocol=protocol(original);
const P=global.EdgeStreamProtocol,{hash}=require(root+'/frontend/streaming/controller.js'),{Controller}=require(root+'/frontend/multisubject/controller.js');
(async()=>{
 const bytes=Uint8Array.from({length:5632},(_,i)=>i%256),sha=await hash(bytes);
 const dataset={trials:Array.from({length:1080},(_,index)=>({index,trial_id:'native-fixture/'+index,ground_truth:index%2,input_sha256:sha,source_trial:{subject:1+Math.floor(index/120),artifact:false,session_trial:25+index%120}})),input:async()=>bytes.slice()};
 const child=spawn(process.env.EDGE_MULTI_NATIVE||'/private/tmp/edgebci-multisubject-native-test',['--serve']);
 let processError=null;child.on('error',e=>{processError=e;});const stderr=[];child.stderr.on('data',b=>stderr.push(b));
 const transport={onData:()=>{},onDisconnect:()=>{},write:b=>new Promise((resolve,reject)=>child.stdin.write(b,e=>e?reject(e):resolve()))};
 child.stdout.on('data',b=>{for(let i=0;i<b.length;i+=11)transport.onData(new Uint8Array(b.subarray(i,i+11)));});
 let c;c=new Controller({dataset,transport,visualize:async()=>{},onRecord:()=>{if(c.records.length===1081)c.pause();}});
 try{await c.handshake();await c.play();assert.equal(c.failed,null);assert.equal(c.records.length,1081);assert.equal(c.state().unique,1080);assert.equal(c.completedRecords.filter(r=>r.repeated).length,1);for(const r of c.records){assert.deepEqual(r.logits,[0,-1]);assert.equal(r.class_id,0);assert.equal(r.input_crc32,P.crc(bytes));}assert.equal(processError,null);}finally{child.stdin.end();}
 await new Promise((resolve,reject)=>child.on('close',code=>code===0?resolve():reject(Error('Native mock server exited '+code+': '+Buffer.concat(stderr)))));
 console.log('PASS: unchanged JS controller and new profile interoperate with actual bounded C++ parser/server for1081 requests, all1080trial IDs+reshuffle, fragmented responses and exact test-only endpoint logits. No model inference/physical serial used.');
})().catch(e=>{console.error(e);process.exitCode=1;});
