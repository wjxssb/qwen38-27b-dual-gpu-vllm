"""Observe systemd accepting a stop even when the old container already exited."""
from common import *

def select(records,boot,begin,end):
    matches=[]
    for row in records:
        if row.get('_BOOT_ID')!=boot.replace('-','') or row.get('_UID')!=str(os.getuid()) or row.get('_COMM')!='systemd' or row.get('USER_UNIT')!='qwen27b.service':continue
        if not row.get('MESSAGE','').startswith('Stopping qwen27b.service - '):continue
        stamp=int(row['__REALTIME_TIMESTAMP'])/1e6
        if begin<=stamp<=end:matches.append(row)
    if len(matches)>1:raise RuntimeError('Ambiguous systemd stop timeline')
    return matches[0] if matches else None

def observe(receipt,out):
    begin=receipt['timeline']['worker_failure']
    end=min(receipt['timeline'].get('old_supervisor_gone',time.time()),receipt['timeline'].get('new_supervisor_started',time.time()))
    raw=cmd(['journalctl','--user','-b','-u','qwen27b.service','--since','@'+str(begin),'-o','json','--no-pager'])
    row=select([json.loads(line) for line in raw.splitlines()],receipt['old_identity']['boot_id'],begin,end)
    if row:
        save(out/'systemd-stop-job.json',row)
        receipt['timeline'].setdefault('stop_issued',int(row['__REALTIME_TIMESTAMP'])/1e6)
        receipt['stop_timing_source']='systemd manager accepted stop job for the old service, before old supervisor exit; independent of container stop dispatch'

