import ast,unittest,types
from pathlib import Path
P=Path('/home/frank/nvidia-dense-runtime/candidate-v45-renderer-handshake-overlap-20260923/overlay/vllm/v1/engine/utils.py')
N=types.SimpleNamespace
class HandshakeTests(unittest.TestCase):
 def setup_case(self,messages,n=1,callback=None):
  tree=ast.parse(P.read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='wait_for_engine_startup');f.returns=None
  for a in f.args.args:a.annotation=None
  # Function-local variable annotations are not evaluated.
  self.sent=[];self.received=[]
  class Socket:
   def recv_multipart(inner):
    item=messages.pop(0);self.received.append(item);return item
   def send_multipart(inner,item,**kw):self.sent.append(item)
  sock=Socket()
  class Poller:
   def register(inner,*a):pass
   def poll(inner,*a):return [(sock,1)]
  engines=[N(identity=i.to_bytes(2,'little'),local=True,state='NEW') for i in range(n)]
  ns={'zmq':N(Poller=Poller,POLLIN=1),'CoreEngineProcManager':type('Unused',(),{}),'CoreEngineState':N(NEW='NEW',CONNECTED='CONNECTED',READY='READY'),'STARTUP_POLL_PERIOD_MS':1,'msgspec':N(msgpack=N(decode=lambda x:x,encode=lambda x:x)),'EngineHandshakeMetadata':lambda **kw:kw,'logger':N(debug=lambda *a:None)}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(P),'exec'),ns)
  config=N(data_parallel_size_local=n,data_parallel_hybrid_lb=False,data_parallel_external_lb=False)
  launch=N(engine_manager=None,coordinator=None,watched_frontend_processes=[],addresses='fixed')
  return lambda:ns['wait_for_engine_startup'](sock,engines,config,False,None,launch,callback),engines
 def message(self,rank,status):return (rank.to_bytes(2,'little'),{'status':status,'local':True,'headless':False})
 def test_callback_after_all_init_replies_before_ready(self):
  seen=[]
  def callback():seen.append(([e.state for e in engines],len(self.sent),len(self.received)))
  run,engines=self.setup_case([self.message(0,'HELLO'),self.message(1,'HELLO'),self.message(0,'READY'),self.message(1,'READY')],2,callback)
  run();self.assertEqual(seen,[(['CONNECTED','CONNECTED'],2,2)]);self.assertEqual([e.state for e in engines],['READY','READY'])
 def test_callback_failure_never_admits_ready(self):
  def callback():raise RuntimeError('preparation failed')
  run,engines=self.setup_case([self.message(0,'HELLO'),self.message(0,'READY')],callback=callback)
  with self.assertRaisesRegex(RuntimeError,'preparation failed'):run()
  self.assertEqual(engines[0].state,'CONNECTED');self.assertEqual(len(self.received),1)
 def test_no_callback_retains_original_handshake(self):
  run,e=self.setup_case([self.message(0,'HELLO'),self.message(0,'READY')]);run();self.assertEqual(e[0].state,'READY');self.assertEqual(len(self.sent),1)
 def test_unexpected_peer_never_runs_callback(self):
  seen=[];run,e=self.setup_case([self.message(5,'HELLO')],callback=lambda:seen.append(1))
  with self.assertRaisesRegex(RuntimeError,'unexpected data parallel rank'):run()
  self.assertFalse(seen)
 def test_ready_before_hello_rejected(self):
  run,e=self.setup_case([self.message(0,'READY')])
  with self.assertRaisesRegex(RuntimeError,'Unexpected READY'):run()
if __name__=='__main__':unittest.main()
