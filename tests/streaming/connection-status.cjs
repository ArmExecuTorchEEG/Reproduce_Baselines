/* No Web Serial or hardware access: exercise connection UI with mocked transport. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
async function scenario(failure){
 const nodes=new Map();const element=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',handlers:{},addEventListener(n,f){this.handlers[n]=f},querySelector(){return {textContent:''}}});return nodes.get(id)};
 let connected=false,finished=false,begins=0,handshakes=0;
 const context={window:{EdgeStreamingMode:true},document:{getElementById:element,querySelector:()=>({textContent:''})},ArmEEGDemo:{setPlaying(){}},ArmEEGHardware:{begin(){begins++;finished=false;connected=true},disconnected(){connected=false;finished=true},connected(){connected=true}},EdgeStreamDataset:{load:async()=>({})},EdgeStream:{SerialLink:class{async connect(){}},Controller:class{constructor(){this.ready=false;this.records=[];this.paused=true}async handshake(){handshakes++;if(failure)throw Error('Streaming firmware handshake timeout');this.ready=true}fail(e){this.failed=e}play(){this.paused=false;return Promise.resolve()}pause(){this.paused=true}}}};
 vm.createContext(context);vm.runInContext(fs.readFileSync('frontend/streaming/bridge.js','utf8'),context);
 assert.equal(connected,false);await element('connectBtn').onclick();assert.equal(handshakes,1);
 if(failure){assert.equal(connected,false);element('playBtn').handlers.click({stopImmediatePropagation(){}});assert.match(element('transportStatus').textContent,/handshake timeout/);assert.equal(handshakes,1);}else{assert.equal(connected,true);assert.equal(finished,false,'READY must clear pre-handshake completion state');assert.equal(begins,3);assert.match(element('transportStatus').textContent,/Streaming firmware verified/);element('playBtn').handlers.click({stopImmediatePropagation(){}});assert.equal(context.window.EdgeStreamingController.paused,false);}
}
(async()=>{await scenario(false);await scenario(true);console.log('PASS: READY clears pre-handshake completion; Play enabled only after verification; error retained without retry. No physical port used.');})().catch(e=>{console.error(e);process.exitCode=1});
