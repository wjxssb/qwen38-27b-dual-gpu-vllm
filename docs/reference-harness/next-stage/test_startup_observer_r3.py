import unittest, tempfile, os, importlib.util, asyncio
from pathlib import Path

class ObserverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();os.environ['QWEN_STARTUP_OBSERVER_DIR']=cls.tmp.name
        p=Path(__file__).parents[1]/'recovery/source-candidate/startup_observer_r3.py'
        spec=importlib.util.spec_from_file_location('observer_test',p)
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)
    def test_return_and_exception_preserved(self):
        obj=object();self.assertIs(self.m.timed(lambda:obj,'value')(),obj)
        err=ValueError('original')
        def fail():raise err
        try:self.m.timed(fail,'failure')()
        except ValueError as actual:self.assertIs(actual,err)
        else:self.fail('exception swallowed')
    def test_generator_is_consumed_once(self):
        counts=[]
        def values():
            for n in range(3):counts.append(n);yield n
        self.assertEqual(list(self.m.timed(values,'generator')()),[0,1,2]);self.assertEqual(counts,[0,1,2])
    def test_async_result(self):
        async def value():return 7
        self.assertEqual(asyncio.run(self.m.timed(value,'async')()),7)
    def test_descriptor_binding(self):
        import types
        class C:
            @classmethod
            def c(cls):return cls
            @staticmethod
            def s(x):return x
        module=types.SimpleNamespace(C=C,__name__='fixture')
        self.m.WATCH['fixture']=['C.c','C.s'];self.m.instrument_module(module)
        self.assertIs(C.c(),C);self.assertEqual(C.s(4),4)
    def test_pipe_wrapper_preserves_payload_and_ownership(self):
        import types,threading
        mod=types.SimpleNamespace()
        self.m.instrument_spawn_open(mod)
        r,w=os.pipe();value=b'payload'*4096;got=[]
        def reader():
            while chunk:=os.read(r,4096):got.append(chunk)
        t=threading.Thread(target=reader);t.start()
        with mod.open(w,'wb',closefd=False) as f:self.assertEqual(f.write(value),len(value))
        os.fstat(w);os.close(w);t.join(3);os.close(r)
        self.assertFalse(t.is_alive());self.assertEqual(b''.join(got),value)

    def test_balanced_monotonic_evidence(self):
        import json
        self.m.timed(lambda:None,'balanced')()
        rows=[json.loads(s) for p in Path(self.tmp.name).glob('*.jsonl') for s in p.read_text().splitlines()]
        events=[x for x in rows if x['label']=='balanced']
        self.assertEqual([x['kind'] for x in events],['begin','end'])
        self.assertEqual(events[0]['span'],events[1]['span']);self.assertLess(events[0]['monotonic_ns'],events[1]['monotonic_ns'])

if __name__=='__main__':unittest.main()
