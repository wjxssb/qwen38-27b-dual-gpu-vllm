import ast,unittest,contextlib,types
from pathlib import Path
ROOT=Path('/home/frank/qwen38-next-stage-20260923')
BASE=ROOT/'recovery/source-candidate/image-vllm-renderers-base.py'
NEW=Path('/home/frank/nvidia-dense-runtime/candidate-v43-renderer-overlap-20260923/overlay/vllm/renderers/base.py')

def renderer(path):
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseRenderer')
 names={'warmup','_warmup_multimodal_processors','prewarm_multimodal_for_startup'}
 funcs=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in names]
 for f in funcs:
  f.decorator_list=[];f.returns=None
  for arg in f.args.args:arg.annotation=None
  # Replace only external exception import so real CPU warmup control flow runs.
  f.body=[n for n in f.body if not isinstance(n,ast.ImportFrom)]
 mod=ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[]));ns={'set_default_torch_num_threads':lambda n:contextlib.nullcontext(),'time':__import__('time'),'ChatTemplateResolutionError':LookupError,'logger':types.SimpleNamespace(debug=lambda *a,**k:None,warning=lambda *a,**k:None)}
 exec(compile(mod,str(path),'exec'),ns)
 class Fixture:
  def __init__(self,fail=False):self.mm_processor='mm';self._readonly_mm_processor='readonly';self.calls=[];self.fail=fail
  def render_chat(self,*a):self.calls.append('chat')
  def _warmup_mm_processor(self,p,**kw):
   self.calls.append('process:'+p)
   if self.fail:raise ValueError('processor failure')
  def clear_mm_cache(self):self.calls.append('clear:mm')
  def _clear_processor_cache(self,p):self.calls.append('clear:'+p)
 for name in names:
  if name in ns:setattr(Fixture,name,ns[name])
 return Fixture

class WarmupTests(unittest.TestCase):
 def test_original_path_unchanged_without_early_callback(self):
  a=renderer(BASE)();b=renderer(NEW)();a.warmup(None);b.warmup(None);self.assertEqual(a.calls,b.calls)
 def test_early_warmup_then_chat_runs_each_operation_once(self):
  a=renderer(BASE)();b=renderer(NEW)();a.warmup(None);b.prewarm_multimodal_for_startup();b.warmup(None)
  self.assertEqual(sorted(a.calls),sorted(b.calls));self.assertEqual(b.calls[-1],'chat');self.assertFalse(b._startup_mm_warmed)
 def test_later_explicit_warmup_is_not_skipped(self):
  b=renderer(NEW)();b.prewarm_multimodal_for_startup();b.warmup(None);b.calls=[];b.warmup(None)
  a=renderer(BASE)();a.warmup(None);self.assertEqual(a.calls,b.calls)
 def test_processor_failure_still_cleans_both_caches(self):
  a=renderer(BASE)(True);b=renderer(NEW)(True);a.warmup(None);b.prewarm_multimodal_for_startup();b.warmup(None);self.assertEqual(sorted(a.calls),sorted(b.calls))
 def test_no_multimodal_processors(self):
  b=renderer(NEW)();b.mm_processor=None;b._readonly_mm_processor=None;b.prewarm_multimodal_for_startup();b.warmup(None);self.assertEqual(b.calls,['chat'])
 def test_cleanup_exception_prevents_warmed_flag(self):
  b=renderer(NEW)()
  def broken():raise RuntimeError('cleanup')
  b.clear_mm_cache=broken
  with self.assertRaisesRegex(RuntimeError,'cleanup'):b.prewarm_multimodal_for_startup()
  self.assertFalse(getattr(b,'_startup_mm_warmed',False))
if __name__=='__main__':unittest.main()
