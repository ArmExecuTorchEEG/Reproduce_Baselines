/* Reads genuine prepared assets; no serial port, prediction generation or board access. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const {load}=require(root+'/frontend/multisubject/dataset.js');
const hash=async b=>crypto.createHash('sha256').update(b).digest('hex');
const fetch=async file=>{try{const b=fs.readFileSync(path.join(root,'frontend',file));return {ok:true,json:async()=>JSON.parse(b),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};}catch{return {ok:false};}};
(async()=>{
 const dataset=await load('multisubject/data/',{fetch,hash});let flags=0;
 for(const trial of dataset.trials){assert.equal(await hash(await dataset.input(trial.index)),trial.input_sha256);const channels=dataset.channels(trial.index);assert.equal(Object.keys(channels).length,22);for(const a of Object.values(channels)){assert.equal(a.length,256);assert(a.every(Number.isFinite));assert(Math.max(...a)>Math.min(...a));assert(Math.sqrt(a.reduce((sum,v)=>sum+v*v,0)/a.length)>0);}assert.equal(dataset.channels(trial.index),channels);flags+=Number(trial.artifact);}
 assert.equal(flags,94);await assert.rejects(dataset.input(1080),/Unknown/);assert.throws(()=>dataset.channels(-1),/Unknown/);
 const corruptFetch=async f=>{const r=await fetch(f);if(f.endsWith('inputs.bin')){const b=new Uint8Array(await r.arrayBuffer());b[0]^=1;return {...r,arrayBuffer:async()=>b.buffer};}return r;};await assert.rejects(load('multisubject/data/',{fetch:corruptFetch,hash}),/integrity/);
 console.log('PASS: all 1080 genuine prepared INT8 hashes, corresponding 22x256 finite nonconstant EEG windows, 94 flags, strict pool/identity and corrupted-asset rejection. No inference or serial port used.');
})().catch(e=>{console.error(e);process.exitCode=1;});
