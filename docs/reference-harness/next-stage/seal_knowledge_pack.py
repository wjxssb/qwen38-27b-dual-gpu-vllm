"""Durable source/evidence inventory. Run only outside timed GPU trials.

Index hashes exclude the two self-referential index files. Their hashes are
sealed separately; final report/review may be added by a later explicit rerun.
"""
from common import *
import shutil,datetime
P=PROD/'docs/qwen38-optimization'
OLD=Path('/home/frank/dense-opt-repair-20260922/qualification')

def main():
    external={}
    def add(path,role,kind):
        p=Path(path);assert p.is_file(),p
        external[str(p)]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'role':role,'created_at':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'created_at_semantics':'filesystem mtime, not asserted original authorship time','type':kind}
    def copy(path,rel,role,kind='evidence'):
        p=Path(path);dest=P/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        add(p,role,kind);external[str(p)]['durable_copy']=str(dest);external[str(p)]['durable_sha256']=sha(dest)
    for row in read(P/'SOURCE-MANIFEST.foundation.json')['files']:
        assert sha(row['path'])==row['sha256']
        assert sha(row['durable_copy'])==row['durable_sha256']
        external[row['path']]=row
    for name in ['FINAL-SEAL.json','FINAL-SEAL.sha256']:
        copy(OLD/name,'evidence/qualification/'+name,'original complete qualification evidence seal')
    for release in [NV/'production-dense-grouped-20260922',*sorted(NV.glob('candidate-v4[0-6]*20260923')),NV/'production-dense-recovery-20260923']:
        if not release.exists():continue
        m=read(release/'MANIFEST.json');kind='production' if release.name.startswith('production-') else 'candidate'
        copy(release/'MANIFEST.json','sources/releases/'+release.name+'/MANIFEST.json','immutable release identity',kind)
        for rel in ['cache-blessed/bless.json','cache-blessed/cache-provenance.json']:
            if (release/rel).exists():copy(release/rel,'sources/releases/'+release.name+'/'+rel,'cache seed identity and compatibility proof',kind)
        for rel,entry in m['files'].items():
            assert sha(release/rel)==entry['sha256']
            if (release/rel).suffix in ['.py','.json','.sh','.md','.txt','.mjs','.pth']:
                copy(release/rel,'sources/releases/'+release.name+'/'+rel,'sealed runtime source/config',kind)
            else:add(release/rel,'sealed runtime binary/data; retained original artifact',kind)
    for p in sorted((Q/'scripts').glob('*.py')):
        copy(p,'reference-harness/next-stage/'+p.name,'next-stage harness; reference absolute paths require rebinding','candidate')
    for root,rel in [(Q/'recovery/source-candidate','sources/recovery-diagnostics'),(Q/'xid/source-candidate','sources/xid-diagnostics')]:
        for p in sorted(root.iterdir()):
            if not p.is_file():continue
            # Preserve diagnostic source. Large third-party libraries remain indexed.
            if p.suffix in ['.py','.cpp','.h','.json']:copy(p,rel+'/'+p.name,'diagnostic-only source, never production','candidate')
            else:add(p,'diagnostic compiled library/input; original artifact','candidate')
    for root in [Q/'recovery/evidence',Q/'xid/evidence']:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix not in ['.json','.md','.patch','.txt','.log']:continue
            add(p,'next-stage experiment evidence (including rejected results)','evidence')
    crash=Q/'xid/evidence/graph-trace-host-crash/analysis.json'
    if crash.exists():
        for path,entry in read(crash)['retained_sources'].items():
            add(path,'offline Graph trace host crash evidence; original core retained','evidence')
            assert external[path]['sha256']==entry['sha256']
    for p in sorted((Q/'evidence').glob('*/receipt.json')):add(p,'fault/qualification receipt','evidence')
    for name in ['final-control-identity.json','production-copy-qualification/receipt.json','production-copy-qualification/final-live/loaded-source-identity.json','production-copy-qualification/final-live/worker-runtime-inspection.json','production-copy-qualification/final-live/cache-compatibility.json','final-production-worker-loss/receipt.json','independent-review.json','independent-review.md','independent-review-closure.json','promotion-commit.json']:
        p=Q/'evidence'/name
        if p.exists():copy(p,'evidence/next-stage/'+name,'final qualification/review/promotion evidence')
    for name in ['FINAL-REPORT.md','REQUEST.txt']:
        p=Q/name
        if p.exists():copy(p,'evidence/next-stage/'+name,'next-stage report/request')
    for p in (Q/'recovery').glob('*.json'):add(p,'recovery derived results','evidence')
    for p in (P/'evidence/history').iterdir():
        if p.is_file():add(p,'durable historical source snapshot','evidence')
    for p in [NV/'drift-closeout-20260914/CLOSEOUT.md',NV/'NVIDIA_MODEL_INVENTORY.json',Path('/home/frank/nvidia-runtime-history/FINAL_ARCHIVE_INDEX.json')]:
        add(p,'model/history provenance; not an executable migration instruction','evidence')
    identity_path=Q/'evidence/final-control-identity.json'
    if identity_path.exists():
        identity=read(identity_path);binding=Path(identity['generated_binding'])
        for p in binding.iterdir():
            if p.is_file():copy(p,'sources/final-binding/'+p.name,'installed production control binding','production')
    exclusions={'SOURCE-MANIFEST.json','EVIDENCE-INDEX.json','SOURCE-MANIFEST.sha256'}
    internal=[]
    for p in sorted(P.rglob('*')):
        if not p.is_file() or str(p.relative_to(P)) in exclusions:continue
        internal.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'role':'durable knowledge artifact','created_at':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'created_at_semantics':'filesystem mtime','type':'evidence' if 'evidence' in p.parts else 'candidate' if 'reference-harness' in p.parts else 'knowledge'})
    save(P/'SOURCE-MANIFEST.json',{'schema':1,'state':'SEALED_FILES','generated_epoch':time.time(),'scope':'All durable pack files except self-referential indexes, plus important verified external sources. Retained historical evidence is also covered by the copied original FINAL-SEAL. Externally indexed data are not claimed portable without their files.','exclusions':sorted(exclusions),'files':list(external.values()),'durable_files':internal})
    save(P/'EVIDENCE-INDEX.json',{'schema':1,'source_manifest':str(P/'SOURCE-MANIFEST.json'),'source_manifest_sha256':sha(P/'SOURCE-MANIFEST.json'),'roles':[{'role':r['role'],'path':r.get('durable_copy',r['path']),'original_path':r['path'],'sha256':r['sha256'],'type':r['type']} for r in external.values()],'durable_artifact_count':len(internal)})
    (P/'SOURCE-MANIFEST.sha256').write_text(sha(P/'SOURCE-MANIFEST.json')+'  SOURCE-MANIFEST.json\n'+sha(P/'EVIDENCE-INDEX.json')+'  EVIDENCE-INDEX.json\n')
    print(json.dumps({'external':len(external),'durable':len(internal),'source_manifest_sha256':sha(P/'SOURCE-MANIFEST.json')}))

if __name__=='__main__':main()
