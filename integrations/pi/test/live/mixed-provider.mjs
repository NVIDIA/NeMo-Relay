// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { extension, withRpc } from './rpc.mjs';

const [pi] = process.argv.slice(2);
assert.ok(pi, 'supply the Pi 1.6.0 executable');
const received = [];
const server = createServer((req, res) => {
  received.push({ path:req.url, headers:Object.keys(req.headers) });
  res.end('{}');
});
server.listen(0,'127.0.0.1');
await once(server,'listening');
try {
  await withRpc(pi, async (root, agent) => {
    const base = 'http://127.0.0.1:' + server.address().port;
    const mixed = join(root,'mixed.mjs');
    await writeFile(join(agent,'settings.json'),JSON.stringify({defaultProvider:'mixed',defaultModel:'chat'}));
    await writeFile(mixed, `export default pi => {
      const cost={input:0,output:0,cacheRead:0,cacheWrite:0};
      const common={baseUrl:${JSON.stringify(base)},input:['text'],cost,contextWindow:64000};
      async function dispatch(model, _context, options) {
        await fetch(model.baseUrl+'/'+model.id,{method:'POST',headers:options.headers});
        return {};
      }
      pi.registerProvider('mixed',{apiKey:'fixture',models:[
        {...common,id:'chat',name:'Chat',api:'openai-completions',maxTokens:4096,reasoning:false},
        {...common,id:'image',name:'Image',api:'fixture-images',type:'image',output:['image']},
        {...common,id:'classifier',name:'Classifier',api:'fixture-classifier',type:'classifier'},
      ],images:{'fixture-images':{generateImages:dispatch}},classifiers:{'fixture-classifier':{classify:dispatch}}});
      pi.registerCommand('mixed-probe',{description:'exercise non-chat operations',handler:async (_,ctx)=>{
        const registry=ctx.modelRegistry;
        if (!registry.getRegisteredNativeProvider('mixed')) throw new Error('chat wrapper was never registered');
        await registry.generateImages(registry.findOfType('image','mixed','image'),{messages:[]});
        await registry.classify(registry.findOfType('classifier','mixed','classifier'),{input:'fixture',labels:['a','b']});
      }});
    };`);
    return {extensions:[mixed,extension], flags:['--model','mixed/chat'],
      env:{ NEMO_RELAY_PI_REDIRECT:'force', NEMO_RELAY_PROXY_CREDENTIAL:'fixture-only' }};
  }, async ({command}) => {
    const result = await command('prompt',{message:'/mixed-probe'});
    assert.ok(result.success, JSON.stringify(result));
    assert.deepEqual(received.map(r=>r.path), ['/image','/classifier']);
    assert.ok(received.every(r=>r.headers.every(h=>!h.startsWith('x-nemo-relay-'))), JSON.stringify(received));
  });
} finally { server.closeAllConnections(); await new Promise(r=>server.close(r)); }
