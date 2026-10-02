"""Seal instrumentation separately from the accepted production payload."""
from common import *
import shutil, ast

parent=NV/'production-dense-grouped-20260922'
dest=NV/'candidate-v42-startup-spawn-observer-20260923'
assert not dest.exists()
m=read(parent/'MANIFEST.json');dest.mkdir()
for rel,e in m['files'].items():
    assert sha(parent/rel)==e['sha256']
    p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/rel,p)
rel='overlay/startup_observer.py';p=dest/rel
shutil.copy2(Q/'recovery/source-candidate/startup_observer_r3.py',p)
rows=read(dest/'overlay-integrity.json')
rows.append({'path':rel,'target':'/usr/local/lib/python3.12/dist-packages/qwen_startup_observer.py'})
pth='overlay/qwen_startup_observer.pth'
(dest/pth).write_text('import qwen_startup_observer\n')
rows.append({'path':pth,'target':'/usr/local/lib/python3.12/dist-packages/qwen_startup_observer.pth'})
for row in rows:row.update(bytes=(dest/row['path']).stat().st_size,sha256=sha(dest/row['path']))
save(dest/'overlay-integrity.json',rows)
# Host markers wrap existing functions only; no cache, admission or math change.
p=dest/'launcher.py';s=p.read_text();marker="if __name__=='__main__':"
assert s.count(marker)==1
hook='''
if __name__ != '__main__' and os.environ.get('QWEN_STARTUP_OBSERVER_DIR'):
 import importlib.util as _iu
 _sp=_iu.spec_from_file_location('startup_host_observer',ROOT/'overlay/startup_observer.py')
 _ob=_iu.module_from_spec(_sp);_sp.loader.exec_module(_ob)
 for _n in ['static','verify_inventory','preflight','planned_create_command','verify_p2p']:
  globals()[_n]=_ob.timed(globals()[_n],'host.'+_n)
'''
s=s.replace(marker,hook+'\n'+marker);ast.parse(s);p.write_text(s)
m['files'][rel]={}
m['files'][pth]={}
m.update(state='SEALED_DIAGNOSTIC_ONLY',parent_manifest_sha256=sha(parent/'MANIFEST.json'),qualification_variable='startup-observer',diagnostic_only=True)
m['files']={rel:{'bytes':(dest/rel).stat().st_size,'sha256':sha(dest/rel)} for rel in m['files']}
save(dest/'MANIFEST.json',m)
module('startup_sealed',dest/'launcher.py').static()
save(Q/'recovery/evidence'/(dest.name+'-assembly.json'),{'state':'SEALED_DIAGNOSTIC_ONLY','parent':str(parent),'candidate':str(dest),'parent_manifest':sha(parent/'MANIFEST.json'),'manifest':sha(dest/'MANIFEST.json'),'changes':{rel:{'before':e['sha256'],'after':m['files'][rel]['sha256']} for rel,e in read(parent/'MANIFEST.json')['files'].items() if e!=m['files'][rel]},'new_files':[rel for rel in m['files'] if rel not in read(parent/'MANIFEST.json')['files']]})
print(dest)
