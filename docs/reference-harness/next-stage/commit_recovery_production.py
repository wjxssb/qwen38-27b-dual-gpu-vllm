"""Commit reviewed, already qualified active production; no deployment changes."""
from common import *
import lifecycle
from fault_test import smoke
from final_runtime_identity import collect

def main():
    stage=read(Q/'evidence/production-copy-qualification/receipt.json')
    assert stage['state']=='QUALIFIED_ACTIVE_PENDING_INDEPENDENT_REVIEW'
    release=Path(stage['production']);fallback=Path(stage['fallback'])
    review_path=Q/'evidence/independent-review.json';review=read(review_path)
    closure_path=Q/'evidence/independent-review-closure.json';closure=read(closure_path)
    assert review['verdict'] in ['PASS','PASS_WITH_DISCLOSED_LIMITATIONS']
    assert closure['state']=='PASS' and closure['unresolved_findings']==0
    assert closure['review_sha256']==sha(review_path)
    assert lifecycle.selected()==release and lifecycle.accepted()==fallback
    assert sha(release/'MANIFEST.json')==stage['production_manifest']
    module('commit_static',release/'launcher.py').static()
    owner,_,_=assert_idle(release)
    prior=read(Q/'evidence/production-copy-qualification/previous-canonical-ledgers.json')
    for path,record in prior.items():assert sha(path)==record['sha256'],'Canonical ownership changed during review'
    record={'state':'COMMITTING_REVIEWED_RELEASE','runtime':str(release),'manifest':sha(release/'MANIFEST.json'),'previous_runtime':str(fallback),'owner':owner,'stage_receipt_sha256':sha(Q/'evidence/production-copy-qualification/receipt.json'),'review_sha256':sha(review_path),'closure_sha256':sha(closure_path),'started_epoch':time.time()}
    out=Q/'evidence/promotion-commit.json';save(out,record)
    record['completion']=smoke()
    record['final_identity']=collect(release,Q/'evidence/final-control-identity.json')
    try:
        save(Q/'canonical.json',{'state':'PROMOTED','candidate':str(release),'manifest':sha(release/'MANIFEST.json'),'receipt':str(out),'epoch':time.time()})
        save(PROD/'dense-production-release.json',{'state':'PROMOTED','runtime':str(release),'manifest_sha256':sha(release/'MANIFEST.json'),'bindings':str(lifecycle.bindings(release)),'receipt':str(out),'previous_runtime':str(fallback),'epoch':time.time()})
    except BaseException:
        for path,old in prior.items():
            current=read(path)
            assert current==old['content'] or current.get('candidate',current.get('runtime'))==str(release),'Refusing to overwrite foreign ledger'
            save(path,old['content'])
        record['state']='COMMIT_FAILED_CANONICAL_LEDGER_RESTORED';save(out,record);raise
    record.update(state='PROMOTED',finished_epoch=time.time());save(out,record)
    save(Q/'PROMOTION-LEDGER.json',{'state':'PROMOTED','canonical':str(release),'manifest':record['manifest'],'fallback':str(fallback),'receipt':str(out),'independent_review':str(review_path),'epoch':time.time()})
    print(json.dumps({'state':'PROMOTED','runtime':str(release),'manifest':record['manifest']}))

if __name__=='__main__':main()
