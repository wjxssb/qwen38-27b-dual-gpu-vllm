"""Hash the closed task and durable pack after reviewed canonical commit."""
from common import *
import stat

def stream_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    assert read(Q/'evidence/promotion-commit.json')['state']=='PROMOTED'
    assert read(Q/'evidence/independent-review-closure.json')['state']=='PASS'
    assert read(Q/'WORKSTATE.json')['state']=='COMPLETE'
    files={}
    for path in sorted(Q.rglob('*')):
        if '__pycache__' in path.parts or path.name.endswith('.tmp'):continue
        if path.name in ['FINAL-SEAL.json','FINAL-SEAL.sha256']:continue
        if not stat.S_ISREG(path.lstat().st_mode):continue
        files[str(path.relative_to(Q))]={'sha256':stream_sha(path),'bytes':path.stat().st_size}
    pack=PROD/'docs/qwen38-optimization'
    manifest={'state':'FINAL_SEALED','epoch':time.time(),'runtime':read(PROD/'dense-production-release.json'),'files':files,'durable_pack_indexes':{str(pack/name):stream_sha(pack/name) for name in ['SOURCE-MANIFEST.json','SOURCE-MANIFEST.sha256','EVIDENCE-INDEX.json']},'scope':'Task regular files excluding self, temporary files and Python bytecode. Runtime payload and pack artifacts have their own verified manifests. This does not freeze live runtime logs or later service state.'}
    save(Q/'FINAL-SEAL.json',manifest)
    (Q/'FINAL-SEAL.sha256').write_text(sha(Q/'FINAL-SEAL.json')+'  FINAL-SEAL.json\n')
    print(json.dumps({'state':'FINAL_SEALED','files':len(files),'sha256':sha(Q/'FINAL-SEAL.json')}))

if __name__=='__main__':main()
