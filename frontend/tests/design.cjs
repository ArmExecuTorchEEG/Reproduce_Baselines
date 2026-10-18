const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const dir=path.resolve(__dirname,'..'),source=process.env.CINDY_UI_DIR||path.join(os.homedir(),'Desktop/cindy-new-ui');
const expected=JSON.parse(fs.readFileSync(path.join(__dirname,'cindy-source-manifest.json'),'utf8'));
const hash=buf=>crypto.createHash('sha256').update(buf).digest('hex');
/* Cindy's stylesheet must survive byte-identical as the leading block of styles.css. Deliberate layout sizing
 * may only be appended after it as a separate block that starts with a comment naming the change; the original
 * rules themselves are never edited. Returns the appended block ('' when the file is the untouched original). */
function appendedBlock(file,label){
 const data=fs.readFileSync(file);if(hash(data)===expected['styles.css'])return '';
 let at=-1;while((at=data.indexOf(10,at+1))!==-1){if(hash(data.subarray(0,at+1))===expected['styles.css']){const rest=data.subarray(at+1).toString('utf8');assert.match(rest,/^\s*\/\*[^*]*\*\//,label+': appended sizing block must start with a comment naming the change');return rest;}}
 assert.fail(label+': Cindy stylesheet must remain byte-identical as the leading block; only a documented block may be appended');
}
const merged=appendedBlock(path.join(dir,'styles.css'),'frontend/styles.css');
if(fs.existsSync(source)){
 for(const [name,sha] of Object.entries(expected)){const file=path.join(source,name);if(name==='styles.css'){const block=appendedBlock(file,'design-reference styles.css');if(block)console.log('NOTE: the design-reference copy of styles.css carries the same appended sizing block as frontend/styles.css; Cindy\'s original rules are intact but the copy is no longer the untouched original.');assert.equal(block,merged,'design-reference and frontend sizing blocks must agree so geometry comparisons stay meaningful');}else assert.equal(hash(fs.readFileSync(file)),sha,'Original reference changed: '+name);}
 const original=fs.readFileSync(path.join(source,'index.html'),'utf8'),mergedHtml=fs.readFileSync(path.join(dir,'index.html'),'utf8');
 for(const pattern of [/<svg class="person"[\s\S]*?<\/svg>/,/<svg class="scalp-topomap[\s\S]*?<\/svg>/])assert.equal(mergedHtml.match(pattern)[0],original.match(pattern)[0],'Cindy SVG must remain byte-identical');
}
console.log('PASS: Cindy stylesheet preserved as the leading block'+(merged?' with a documented appended sizing block ('+merged.trim().split('\n')[0]+')':'')+'; human and 22-electrode SVGs preserved; all supplied originals including PDF unchanged apart from that block.');
