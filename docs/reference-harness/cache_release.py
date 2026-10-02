"""Keep the exact baseline kernel-tuning inputs constant across Python observers."""
import ast,shutil,stat
from common import *

ORIGINAL=NV/'cache-blessed'

def normalized_model(path):
    tree=ast.parse(path.read_text())
    tree.body=[node for node in tree.body if not (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='_diag_checkpoint') and not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_DIAG_SYNC_STAGES' for t in node.targets))]
    return ast.dump(tree,include_attributes=False)

def compatible(candidate):
    module('cache_baseline_static',BASE/'launcher.py').static()
    module('cache_target_static',candidate/'launcher.py').static()
    base=read(BASE/'MANIFEST.json');target=read(candidate/'MANIFEST.json')
    assert set(base['files'])<=set(target['files']),'Missing sealed baseline source'
    assert sha(candidate/'environment.json')==sha(BASE/'environment.json')
    for filename in ['PROFILES.json','graph-prefix.json']:
        b=read(BASE/filename);c=read(candidate/filename)
        profile=c['profiles']['graph-prefix'] if filename=='PROFILES.json' else c
        argv=profile['argv']
        if '--profiler-config' in argv:
            i=argv.index('--profiler-config');del argv[i:i+2]
        assert b==c,('Kernel input configuration differs',filename)
    allowed={'overlay-integrity.json','PROFILES.json','graph-prefix.json','overlay/vllm/stability_telemetry.py','overlay/vllm/model_executor/models/qwen3_next.py'}
    if target.get('qualification_variable')=='profile':
        allowed.update({'overlay/vllm/dense_timing.py','overlay/vllm/v1/worker/gpu_model_runner.py','overlay/vllm/v1/spec_decode/llm_base_proposer.py'})
        assert target['state']!='SEALED_PRODUCTION','Never promote profiling instrumentation'
    for rel,entry in target['files'].items():
        assert sha(candidate/rel)==entry['sha256']
        if rel not in allowed:assert rel in base['files'] and entry==base['files'][rel],('Native/kernel-affecting source differs',rel)
    assert normalized_model(candidate/'overlay/vllm/model_executor/models/qwen3_next.py')==normalized_model(BASE/'overlay/vllm/model_executor/models/qwen3_next.py'),'Model math differs outside diagnostic boundaries'
    return {'state':'PASS','scope':'Identical image/model/environment/shape/configuration; torch compilation mode remains 0. Native/JIT kernel definitions unchanged; allowed differences are CPU telemetry, diagnostic fences, or separately non-promotable profiler observers. CUDA graphs recaptured for each fresh process.','source_manifest':sha(BASE/'MANIFEST.json'),'target_manifest':sha(candidate/'MANIFEST.json')}

def prepare(candidate):
    if candidate==BASE:return ORIGINAL
    proof=compatible(candidate);dest=candidate/'cache-blessed'
    if dest.exists():
        receipt=read(dest/'cache-provenance.json');assert receipt['target_manifest']==sha(candidate/'MANIFEST.json')
        assert read(dest/'bless.json')['manifest_sha']==sha(candidate/'MANIFEST.json')
        for rel,digest in receipt['files'].items():assert sha(dest/'tree'/rel)==digest
        return dest
    original=read(ORIGINAL/'bless.json');assert original['manifest_sha']==sha(BASE/'MANIFEST.json')
    profiles=read(candidate/'PROFILES.json');assert original['image_id']==profiles['image']
    staging=candidate/'cache-blessed.pending';staging.mkdir();tree=staging/'tree';tree.mkdir();files={}
    for source in sorted((ORIGINAL/'tree').rglob('*')):
        if not stat.S_ISREG(source.lstat().st_mode):continue
        rel=source.relative_to(ORIGINAL/'tree');target=tree/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        files[str(rel)]=sha(source);assert sha(target)==files[str(rel)]
    blessing=dict(original,manifest_sha=sha(candidate/'MANIFEST.json'),cache_files=len(files),cache_bytes=sum((tree/rel).stat().st_size for rel in files),source_qualification='exact baseline cached kernel artifacts copied after compatibility proof')
    save(staging/'bless.json',blessing)
    save(staging/'cache-provenance.json',dict(proof,source_directory=str(ORIGINAL),source_bless_sha256=sha(ORIGINAL/'bless.json'),files=files,epoch=time.time()))
    staging.rename(dest);return dest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);a=p.parse_args();print(prepare(a.candidate))
