"""Stage an unchanged PTE and development-only EEG vectors in the official app.

Use the pinned official export environment. Does not build, install or run FVP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official',type=Path,required=True)
    p.add_argument('--export-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    original=args.export_run
    report=json.loads((original/'export_report.json').read_text())
    assert report['versions']['executorch']=='1.5.1'
    assert report['target']=='ethos-u85-256' and report['system_config']=='Ethos_U85_SRAM_MRAM'
    assert report['memory_mode']=='Shared_Sram'
    args.out.mkdir(parents=True,exist_ok=False)
    old={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for root in ('artifacts','results') for f in Path(root).rglob('*') if f.is_file() and not f.is_relative_to(args.out)}
    save(args.out/'preserved_hashes.json',old)
    project=args.out/'project'
    shutil.copytree(args.official,project,ignore=shutil.ignore_patterns('.git','.venv','out','tmp'))
    torch.set_num_threads(4)
    d=torch.load(original/'export_data.pt',weights_only=True)
    manifest=json.loads((original/'data_manifest.json').read_text())
    labels=d['development_y']
    indices=[int(i) for label in (0,1) for i in torch.where(labels==label)[0][:2]]
    assert len(indices)==4
    x=d['development_x'][indices]
    q=report['input_quantization']
    inputs=torch.clamp(torch.round(x/q['scale'])+q['zero_point'],-128,127).to(torch.int8)
    dequant=(inputs.float()-q['zero_point'])*q['scale']
    # References correspond to EXACT bytes supplied to FVP, including input rounding.
    from torchao.quantization.pt2e import quantize_pt2e  # register saved Q/DQ operators
    module=torch.export.load(original/'quantized_qdq.pt2').module()
    with torch.no_grad(): expected=torch.cat([module(v[None]) for v in dequant])
    assert torch.isfinite(expected).all()
    np.savez(args.out/'validation_vectors.npz',inputs=inputs.numpy(),expected_logits=expected.numpy(),labels=labels[indices].numpy())
    refs={'model_sha256':hashlib.sha256((original/'tinyeegnet_int8.pte').read_bytes()).hexdigest(),
          'model_source':str(original/'tinyeegnet_int8.pte'),'export_versions':report['versions'],
          'official_hackathon_commit':subprocess.check_output(['git','-C',str(args.official),'rev-parse','HEAD'],text=True).strip(),
          'reference_scope':'host PT2E Q/DQ applied to reconstructed int8 input, NOT FVP',
          'indices':indices,'trials':[manifest['development_trials'][i] for i in indices],
          'labels':labels[indices].tolist(),'expected_logits':expected.tolist(),
          'expected_predictions':expected.argmax(1).tolist(),'input_quantization':q,
          'status':'prepared; firmware unbuilt and FVP unexecuted'}
    assert all(t['subject']==1 and t['session']=='0train' for t in refs['trials']) if 'subject1' in original.name else all(t['subject']==8 for t in refs['trials'])
    save(args.out/'reference.json',refs)
    payload=(original/'tinyeegnet_int8.pte').read_bytes()
    layer=project/'ai_layer'
    (layer/'model_pte.c').write_text('#include "model_pte.h"\n__attribute__((aligned(16))) const unsigned char model_pte[] = {\n'+','.join(str(v) for v in payload)+'\n};\nconst unsigned long model_pte_size = sizeof(model_pte);\n')
    (layer/'model.pte').write_bytes(payload)
    ly=layer/'ai_layer.clayer.yml'
    ly.write_text(ly.read_text().replace('TinyCNN int8 image classifier','TinyEEGNet fixed EEG classifier').replace('    - component: Machine Learning:ExecuTorch Operators:Quantized quantize\n',''))
    header='#pragma once\n#include <cstdint>\nconstexpr int kEEGTrials=4;\nconstexpr int kEEGSamples=11000;\nalignas(16) static const int8_t eeg_inputs[4][11000]={\n'
    header+=',\n'.join('{'+','.join(str(int(v)) for v in row.flatten())+'}' for row in inputs)+'\n};\n'
    (project/'src/eeg_vectors.h').write_text(header)
    src=project/'src/app_main.cpp'
    text=src.read_text()
    text=text.replace('#include "model_pte.h"','#include "model_pte.h"\n#include "eeg_vectors.h"\n#include <cstring>\n#include <cmath>')
    a=text.index('  // TinyCNN takes'); b=text.index('  printf("Test_result: PASS',a)
    text=text[:a]+'''  alignas(16) static int8_t input_data[kEEGSamples];
  std::array<SizesType,4> sizes{1,1,22,500};
  std::array<DimOrderType,4> dim_order{0,1,2,3};
  TensorImpl input_impl(ScalarType::Char, sizes.size(), sizes.data(), input_data, dim_order.data());
  Tensor input(&input_impl);
  for (int trial=0; trial<kEEGTrials; ++trial) {
    std::memcpy(input_data,eeg_inputs[trial],sizeof(input_data));
    auto outputs=module.forward(input);
    if (!outputs.ok()) {
      printf("EEG_ERROR trial=%d err=%u\\n",trial,static_cast<unsigned>(outputs.error()));
      return 1;
    }
    if (outputs->size()!=1 || !outputs->front().isTensor()) return 2;
    Tensor t=outputs->front().toTensor();
    if (t.scalar_type()!=ScalarType::Float || t.dim()!=2 || t.size(0)!=1 || t.size(1)!=2) return 3;
    const float* logits=t.const_data_ptr<float>();
    if (!std::isfinite(logits[0]) || !std::isfinite(logits[1])) return 4;
    const int predicted=logits[1]>logits[0] ? 1:0;
    printf("EEG_RESULT %d %.9g %.9g %d\\n",trial,static_cast<double>(logits[0]),static_cast<double>(logits[1]),predicted);
    fflush(stdout);
  }

''' +text[b:]
    text=text.replace('Test_result: PASS','Execution_result: PASS (compare host reference separately)')
    src.write_text(text)
    solution=project/'cmsis-executorch.csolution.yml'
    text=solution.read_text().replace('model: FVP_Corstone_SSE-320','model: ${workspaceFolder}/.vscode/fvp.sh').replace('input-shape: 1x16x16x3','input-shape: 1x1x22x500')
    solution.write_text(text)
    # macOS Bash 3.2 treats empty arrays as unbound under set -u.
    shim=project/'.vscode/fvp.sh'
    shim.write_text(shim.read_text().replace('"${NAME[@]}" "${PORTS[@]}"', '${NAME[@]+"${NAME[@]}"} ${PORTS[@]+"${PORTS[@]}"}'))
    # Use the official backend, allocator, device startup, driver and memory map unchanged.
    shutil.copyfile(project/'.vscode/launch.json.mac',project/'.vscode/launch.json')
    save(args.out/'stage_verification.json',{'source_pte_unchanged':hashlib.sha256((original/'tinyeegnet_int8.pte').read_bytes()).hexdigest()==refs['model_sha256'],
          'staged_pte_identical':(layer/'model.pte').read_bytes()==payload,
          'prior_files_unchanged':all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in old.items())})
    print(f'Staged {project}: four development vectors, model unchanged; NOT built or executed.')

if __name__=='__main__': main()
