"""One owned long-prefill request for the final in-flight peer-loss test."""
import re,socket,threading
from http.client import HTTPConnection
from common import *

class InflightRequest:
    def __init__(self,candidate,label):
        self.candidate=candidate;self.label=label;self.lock=threading.Lock();self.done=threading.Event();self.sent=threading.Event();self.connection=None;self.thread=None
        self.result={'state':'NOT_STARTED','label':label,'prompt_tokens':67584,'requested_tokens':256,'tokens_delivered':0}
    def __enter__(self):return self
    def __exit__(self,*unused):
        if self.connection is not None and self.thread is not None and self.thread.is_alive():
            sock=self.connection.sock
            if sock is not None:
                try:sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
        if self.thread:self.thread.join(timeout=5)
    def update(self,**fields):
        with self.lock:self.result.update(fields)
    def snapshot(self):
        with self.lock:return dict(self.result)
    def start(self):
        assert self.thread is None
        fixture=read(Q/'fixtures/tokens-067584.json')
        payload={'model':'qwen38-27b-dense','prompt':fixture['prompt_token_ids'],'max_tokens':256,'min_tokens':256,'ignore_eos':True,'temperature':0,'seed':0,'add_special_tokens':False,'return_token_ids':True,'stream':True,'cache_salt':'fault-'+self.label+'-'+str(time.time_ns())}
        raw=json.dumps(payload).encode();self.update(state='SENDING',started_epoch=time.time(),request_sha256=hashlib.sha256(raw).hexdigest())
        def worker():
            self.connection=HTTPConnection('127.0.0.1',18080,timeout=120)
            try:
                self.connection.request('POST','/v1/completions',raw,{'Content-Type':'application/json','X-Request-Id':self.label})
                self.update(state='SENT',sent_epoch=time.time());self.sent.set()
                response=self.connection.getresponse();self.update(http_status=response.status)
                if response.status!=200:
                    self.update(state='INTERRUPTED',error=response.read(2048).decode(errors='replace'));return
                count=0;finished=False
                while True:
                    line=response.readline()
                    if not line:break
                    if not line.startswith(b'data:'):continue
                    data=line[5:].strip()
                    if data==b'[DONE]':finished=True;continue
                    obj=json.loads(data)
                    if obj.get('error') or obj.get('object')=='error' or obj.get('type') in ('error','response.failed'):
                        self.update(error=json.dumps(obj,ensure_ascii=False)[:2048])
                    for choice in obj.get('choices',[]):count+=len(choice.get('token_ids') or [])
                    self.update(tokens_delivered=count)
                self.update(state='COMPLETED' if finished and count==256 else 'INTERRUPTED',received_done=finished)
            except Exception as error:self.update(state='INTERRUPTED',error=repr(error))
            finally:
                self.connection.close();self.update(finished_epoch=time.time());self.done.set()
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start()
    def await_executing(self):
        assert self.sent.wait(5),'Owned request was not sent'
        a,_=identity(self.candidate);root=self.candidate/'runs'/a['instance_id']/'logs/stability-telemetry'
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if self.done.is_set():raise RuntimeError('Request ended before injection: '+str(self.snapshot()))
            st=status();assert st['queued_requests']==0 and st['active_requests']<=1,'Another client entered the controlled fault window'
            started=self.snapshot()['sent_epoch']*1e9;seen=[]
            for rank in [0,1]:
                path=root/f'worker-rank-{rank}.jsonl'
                if not path.exists():break
                with path.open('rb') as stream:
                    stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-131072));lines=stream.read().splitlines()
                for line in reversed(lines):
                    try:row=json.loads(line)
                    except ValueError:continue
                    if row.get('wall_time_ns',0)<started:break
                    if row.get('event')=='decoder_sync_boundary' and row.get('decoder',{}).get('status')=='pass':seen.append(rank);break
            if seen==[0,1] and st['active_requests']==1:
                self.update(both_ranks_prefill_observed_epoch=time.time(),instance_id=a['instance_id']);return
            time.sleep(.1)
        raise TimeoutError('Both workers did not enter this owned prefill before injection deadline')
    def verify_interrupted(self,failure_epoch):
        assert self.done.is_set(),'Original failed request still waiting'
        row=self.snapshot();assert row['state']=='INTERRUPTED' and row['tokens_delivered']<256,row
        assert row['finished_epoch']-failure_epoch<90,'Client waited for the old 90-second stall interval'
        return row
