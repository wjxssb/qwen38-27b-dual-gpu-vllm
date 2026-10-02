from common import *
import shutil
parent=NV/'production-dense-grouped-20260922';dest=NV/'candidate-v44-xid-original-profiler-20260923';assert not dest.exists();dest.mkdir();m=read(parent/'MANIFEST.json')
for rel,e in m['files'].items():
 assert sha(parent/rel)==e['sha256'];p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/rel,p)
old=read(NV/'candidate-v35-profile-hoststages-r2-20260922/PROFILES.json')['profiles']['graph-prefix']['argv'];i=old.index('--profiler-config');arg=old[i+1]
for rel in ['PROFILES.json','graph-prefix.json']:
 d=read(dest/rel);profile=d['profiles']['graph-prefix'] if rel=='PROFILES.json' else d;assert '--profiler-config' not in profile['argv'];profile['argv']+=['--profiler-config',arg];save(dest/rel,d)
m.update(state='SEALED_DIAGNOSTIC_ONLY',parent_manifest_sha256=sha(parent/'MANIFEST.json'),qualification_variable='profile',diagnostic_only=True)
m['files']={rel:{'bytes':(dest/rel).stat().st_size,'sha256':sha(dest/rel)} for rel in m['files']};save(dest/'MANIFEST.json',m)
module('xid_diagnostic_seal',dest/'launcher.py').static()
save(Q/'xid/evidence/full-profile-assembly.json',{'state':'SEALED_DIAGNOSTIC_ONLY','parent':str(parent),'parent_manifest':sha(parent/'MANIFEST.json'),'candidate':str(dest),'manifest':sha(dest/'MANIFEST.json'),'changed_files':['PROFILES.json','graph-prefix.json'],'profiler_configuration':json.loads(arg),'differences_from_historical_incident':'Current accepted grouped math/fault-check payload; no old extra host-stage/CUDA-event diagnostic instrumentation. Profiler options exactly match historical original configuration.','active_budget':'One fresh 35K profiling session, one preceding CUPTI-disabled warmup. Stop on any new Xid/error. No retries.'})
print(dest)
