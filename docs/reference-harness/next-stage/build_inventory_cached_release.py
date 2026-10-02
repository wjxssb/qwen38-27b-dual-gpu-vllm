"""After renderer qualification, add just the sealed model identity cache."""
from common import *
import shutil,ast,difflib
parent=NV/'candidate-v45-renderer-handshake-overlap-20260923';dest=NV/'candidate-v46-renderer-inventory-cache-20260923'
qualification=read(Q/'recovery/evidence/renderer-campaign-r2/receipt.json')
assert qualification['state']=='PASS_MEASURED_RESTORED_NOT_PROMOTED'
assert qualification['candidate']==str(parent) and qualification['numerics']=='PASS_EXACT_TOKEN_HASH_AND_SELECTED_RAW_LOGITS'
assert len(qualification['faults']['candidate'])==3
assert not dest.exists();m=read(parent/'MANIFEST.json');dest.mkdir()
for rel,e in m['files'].items():
 assert sha(parent/rel)==e['sha256'];p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/rel,p)
helper=module('seal_inventory',Q/'recovery/source-candidate/inventory_cache.py')
profiles=read(parent/'PROFILES.json');inventory=read(NV/'NVIDIA_MODEL_INVENTORY.json');p2p=read('/run/p2p-stable/verified')
inputs={'inventory_sha256':sha(NV/'NVIDIA_MODEL_INVENTORY.json'),'profiles_sha256':sha(parent/'PROFILES.json'),'environment_sha256':sha(parent/'environment.json'),'model_revision':profiles['model_revision'],'image':profiles['image'],'driver':p2p['driver'],'kernel':p2p['kernel'],'p2p_manifest':p2p['manifest_sha256'],'gpus':p2p['gpus']}
assert inventory['revision']==profiles['model_revision']
started=time.time();receipt=helper.seal(profiles['model_host_path'],inventory,inputs);receipt.update(sealed_epoch=time.time(),seal_elapsed_seconds=time.time()-started)
save(dest/'model-inventory-cache.json',receipt);shutil.copy2(Q/'recovery/source-candidate/inventory_cache.py',dest/'inventory_cache.py')
p=dest/'launcher.py';before=p.read_text();s=before.replace('def verify_inventory(profiles):','def _full_verify_inventory(profiles):',1)
needle='def reject_processes(processes):'
hook='''def verify_inventory(profiles):
 # static() has already verified the helper and receipt against this release.
 import importlib.util
 spec=importlib.util.spec_from_file_location('sealed_model_inventory_cache',ROOT/'inventory_cache.py')
 cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
 p2p=read('/run/p2p-stable/verified')
 inputs={'inventory_sha256':sha(STORAGE_ROOT/'NVIDIA_MODEL_INVENTORY.json'),'profiles_sha256':sha(ROOT/'PROFILES.json'),'environment_sha256':sha(ROOT/'environment.json'),'model_revision':profiles['model_revision'],'image':profiles['image'],'driver':p2p['driver'],'kernel':p2p['kernel'],'p2p_manifest':p2p['manifest_sha256'],'gpus':p2p['gpus']}
 started=time.monotonic()
 hit=cache.matches(profiles['model_host_path'],read(ROOT/'model-inventory-cache.json'),inputs)
 result=inputs['inventory_sha256'] if hit else _full_verify_inventory(profiles)
 print('MODEL_INVENTORY_VERIFY='+json.dumps({'mode':'SEALED_STAT_IDENTITY' if hit else 'FULL_SHA_FALLBACK','elapsed_seconds':time.monotonic()-started,'inventory_sha256':result}),flush=True)
 return result

'''
assert s.count(needle)==1;s=s.replace(needle,hook+needle);ast.parse(s);p.write_text(s)
for rel in ['inventory_cache.py','model-inventory-cache.json']:m['files'][rel]={}
m.update(state='SEALED_CANDIDATE_NOT_PROMOTED',parent_manifest_sha256=sha(parent/'MANIFEST.json'),qualification_variable='renderer-overlap-inventory-cache')
m['files']={rel:{'bytes':(dest/rel).stat().st_size,'sha256':sha(dest/rel)} for rel in m['files']};save(dest/'MANIFEST.json',m)
l=module('cache_release_seal',dest/'launcher.py');l.static();assert l.verify_inventory(profiles)==inputs['inventory_sha256']
out=Q/'recovery/evidence/inventory-cache-assembly';out.mkdir();(out/'source.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='before/launcher.py',tofile='after/launcher.py')))
save(out/'receipt.json',{'state':'SEALED_CANDIDATE_NOT_PROMOTED','parent':str(parent),'parent_manifest':sha(parent/'MANIFEST.json'),'candidate':str(dest),'manifest':sha(dest/'MANIFEST.json'),'variable':'Replace repeated full model SHA with presealed dev/inode/size/mtime_ns/ctime_ns identity verification; changed input or file falls back to unchanged full verifier.','seal_elapsed_seconds':receipt['seal_elapsed_seconds'],'files':{rel:sha(dest/rel) for rel in ['launcher.py','inventory_cache.py','model-inventory-cache.json']},'patch_sha256':sha(out/'source.patch')})
print(dest)
