"""Sequential continuation; stop for review at the first unaccepted result."""
from common import *
import sys
out=Q/'evidence/continuation-ledger.json';record={'state':'WAITING_FOR_RENDERER','steps':[],'started_epoch':time.time()};save(out,record)
def run(name):
 path=Q/'evidence'/(name.removesuffix('.py')+'-controller.log')
 row={'script':name,'source_sha256':sha(Q/'scripts'/name),'state':'RUNNING','started_epoch':time.time()};record['steps'].append(row);save(out,record)
 with path.open('x') as f:
  p=subprocess.Popen([sys.executable,'-B',name],cwd=Q/'scripts',stdout=f,stderr=subprocess.STDOUT)
  row.update(pid=p.pid,start_ticks=ticks(p.pid));save(out,record);code=p.wait()
 row.update(state='PASS_EXIT' if code==0 else 'FAILED_EXIT',returncode=code,finished_epoch=time.time(),log=str(path));save(out,record)
 if code:raise RuntimeError(f'{name} exited {code}; review required, no next GPU action')
try:
 receipt=Q/'recovery/evidence/renderer-campaign-r2/receipt.json'
 while read(receipt)['state']=='RUNNING':time.sleep(3)
 d=read(receipt);assert d['state']=='PASS_MEASURED_RESTORED_NOT_PROMOTED',d['state']
 gain=d['summary']['baseline']['reload_median_s']-d['summary']['candidate']['reload_median_s'];assert gain>=8,('Insufficient gain for further combination; review first',gain)
 record.update(state='INVENTORY_CACHE',renderer_reload_gain_seconds=gain);save(out,record)
 run('build_inventory_cached_release.py');run('inventory_campaign.py')
 assert read(Q/'recovery/evidence/inventory-cache-campaign/receipt.json')['state']=='PASS_MEASURED_RESTORED_NOT_PROMOTED'
 record['state']='MINIMAL_XID_MATRIX';save(out,record);run('minimal_matrix_r3.py')
 minimal=read(Q/'xid/evidence/minimal-matrix-r3/receipt.json')
 if minimal['state']=='COMPLETED_RESTORED' and not any(r['new_xid'] for r in minimal['rows']):
  record['state']='ONE_FULL_PROFILE';save(out,record);run('full_profile_probe.py')
 else:record['full_profile_skipped']='Minimal diagnostic stopped; no further active reproduction without review'
 record['state']='QUALIFICATION_FINISHED_RESTORED_PENDING_ROOT_REVIEW'
except BaseException as error:record.update(state='STOPPED_FOR_REVIEW',error=repr(error));raise
finally:record['finished_epoch']=time.time();save(out,record)
