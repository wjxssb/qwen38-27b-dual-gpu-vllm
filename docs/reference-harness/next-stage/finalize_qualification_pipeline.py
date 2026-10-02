"""Sequential qualification continuation; stops before independent review/commit."""
from common import *
import sys

def main():
    output=Q/'evidence/final-qualification-pipeline.json'
    assert not output.exists()
    record={'state':'WAITING_FOR_RESTORED_QUALIFICATION','steps':[],'started_epoch':time.time(),'canonical_commit':'NEVER_AUTOMATIC; independent review required'};save(output,record)
    def run(name):
        log=Q/'evidence'/(name.removesuffix('.py')+'-final-pipeline.log')
        row={'script':name,'source_sha256':sha(Q/'scripts'/name),'state':'RUNNING','started_epoch':time.time(),'log':str(log)}
        record['steps'].append(row);save(output,record)
        with log.open('x') as stream:
            process=subprocess.Popen([sys.executable,'-B',name],cwd=Q/'scripts',stdout=stream,stderr=subprocess.STDOUT)
            row.update(pid=process.pid,start_ticks=ticks(process.pid));save(output,record);code=process.wait()
        row.update(state='PASS_EXIT' if code==0 else 'FAILED_EXIT',returncode=code,finished_epoch=time.time());save(output,record)
        if code:raise RuntimeError(f'{name} failed; no next action')
    try:
        while True:
            state=read(Q/'evidence/continuation-ledger.json')['state']
            if state=='STOPPED_FOR_REVIEW':raise RuntimeError('Earlier continuation stopped; no GPU action')
            if state=='QUALIFICATION_FINISHED_RESTORED_PENDING_ROOT_REVIEW':break
            time.sleep(3)
        record['state']='DERIVING_COMPLETED_RESULTS';save(output,record)
        run('render_recovery_results.py');run('recovery_native_timelines.py')
        record['state']='QUALIFYING_PROVISIONAL_PRODUCTION_COPY';save(output,record)
        run('stage_recovery_production.py')
        record['state']='PREPARING_REVIEW_PACKAGE';save(output,record)
        for name in ['render_recovery_results.py','recovery_native_timelines.py','knowledge_next_stage.py','render_next_stage_report.py','seal_knowledge_pack.py']:
            # The two derivation scripts run again after the final production fault.
            if any(x['script']==name for x in record['steps']):
                old=Q/'evidence'/(name.removesuffix('.py')+'-final-pipeline.log')
                old.rename(old.with_name(old.stem+'-before-production.log'))
            run(name)
        record['state']='TECHNICAL_WORK_COMPLETE_PENDING_ROOT_CHECK_AND_INDEPENDENT_REVIEW'
    except BaseException as error:record.update(state='STOPPED_FOR_ROOT_REVIEW',error=repr(error));raise
    finally:record['finished_epoch']=time.time();save(output,record)

if __name__=='__main__':main()
