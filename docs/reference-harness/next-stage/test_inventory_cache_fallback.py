import ast,tempfile,unittest,types,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).parents[1]
class FallbackTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.model=self.root/'model';self.model.mkdir();(self.model/'w').write_bytes(b'weights')
  for rel in ['NVIDIA_MODEL_INVENTORY.json','PROFILES.json','environment.json']:(self.root/rel).write_text('{}')
  shutil.copy2(ROOT/'recovery/source-candidate/inventory_cache.py',self.root/'inventory_cache.py')
  import importlib.util
  spec=importlib.util.spec_from_file_location('cachefixture',self.root/'inventory_cache.py');self.cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.cache)
  self.profiles={'model_host_path':str(self.model),'model_revision':'r','image':'image'};self.p2p={'driver':'d','kernel':'k','manifest_sha256':'p','gpus':['a','b']}
  def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
  inputs={'inventory_sha256':sha(self.root/'NVIDIA_MODEL_INVENTORY.json'),'profiles_sha256':sha(self.root/'PROFILES.json'),'environment_sha256':sha(self.root/'environment.json'),'model_revision':'r','image':'image','driver':'d','kernel':'k','p2p_manifest':'p','gpus':['a','b']}
  receipt={'schema':1,'inputs':inputs,'model_root':str(self.model),'files':{'w':self.cache.snapshot_file(self.model/'w',sha(self.model/'w'),7)}}
  (self.root/'model-inventory-cache.json').write_text(json.dumps(receipt));self.calls=[]
  def full(profiles):self.calls.append(profiles);return 'FULL_VERIFIED'
  def read(p):return self.p2p if str(p)=='/run/p2p-stable/verified' else json.loads(Path(p).read_text())
  tree=ast.parse((ROOT/'scripts/build_inventory_cached_release.py').read_text());hook=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='hook' for t in n.targets))
  self.ns={'ROOT':self.root,'STORAGE_ROOT':self.root,'read':read,'sha':sha,'json':json,'time':time,'_full_verify_inventory':full,'print':lambda *a,**kw:None};exec(hook,self.ns)
 def tearDown(self):self.tmp.cleanup()
 def test_unchanged_uses_sealed_identity(self):
  self.assertEqual(self.ns['verify_inventory'](self.profiles),self.ns['sha'](self.root/'NVIDIA_MODEL_INVENTORY.json'));self.assertFalse(self.calls)
 def test_changed_model_invokes_full_verifier(self):
  (self.model/'w').write_bytes(b'CHANGED');self.assertEqual(self.ns['verify_inventory'](self.profiles),'FULL_VERIFIED');self.assertEqual(len(self.calls),1)
 def test_changed_driver_invokes_full_verifier(self):
  self.p2p['driver']='new';self.ns['verify_inventory'](self.profiles);self.assertEqual(len(self.calls),1)
 def test_failed_full_sha_is_not_masked(self):
  (self.model/'w').unlink()
  def fail(profiles):raise RuntimeError('manifest mismatch')
  self.ns['_full_verify_inventory']=fail
  with self.assertRaisesRegex(RuntimeError,'manifest mismatch'):self.ns['verify_inventory'](self.profiles)
 def test_fallback_does_not_reseal_or_write_receipt(self):
  p=self.root/'model-inventory-cache.json';before=p.read_bytes();(self.model/'w').write_bytes(b'CHANGED');self.ns['verify_inventory'](self.profiles);self.assertEqual(before,p.read_bytes())
if __name__=='__main__':unittest.main()
