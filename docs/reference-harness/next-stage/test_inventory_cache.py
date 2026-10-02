import unittest,tempfile,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'recovery/source-candidate'))
import inventory_cache as c
class CacheTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.p=self.root/'weights';self.p.write_bytes(b'original');self.expected={'model':'fixed'}
  self.receipt={'schema':1,'inputs':self.expected,'model_root':str(self.root),'files':{'weights':c.snapshot_file(self.p,c.digest(self.p),8)}}
 def tearDown(self):self.tmp.cleanup()
 def test_unchanged(self):self.assertTrue(c.matches(self.root,self.receipt,self.expected))
 def test_same_size_same_mtime_content_change_invalidates_ctime(self):
  mtime=self.p.stat().st_mtime_ns;self.p.write_bytes(b'changed!');os.utime(self.p,ns=(mtime,mtime));self.assertFalse(c.matches(self.root,self.receipt,self.expected))
 def test_replacement_same_bytes_invalidates_inode(self):
  other=self.root/'other';other.write_bytes(self.p.read_bytes());other.replace(self.p);self.assertFalse(c.matches(self.root,self.receipt,self.expected))
 def test_symlink_rejected(self):
  other=self.root/'other';self.p.rename(other);self.p.symlink_to(other);self.assertFalse(c.matches(self.root,self.receipt,self.expected))
 def test_inputs_drift(self):self.assertFalse(c.matches(self.root,self.receipt,{'model':'different'}))
 def test_missing_file(self):self.p.unlink();self.assertFalse(c.matches(self.root,self.receipt,self.expected))
 def test_wrong_hash_rejected_at_seal(self):
  with self.assertRaisesRegex(RuntimeError,'SHA verification'):c.snapshot_file(self.p,'0'*64,8)
 def test_write_via_hardlink_invalidates(self):
  other=self.root/'link';os.link(self.p,other);other.write_bytes(b'changed!');self.assertFalse(c.matches(self.root,self.receipt,self.expected))
 def test_receipt_input_missing(self):self.assertFalse(c.matches(self.root,{},self.expected))
if __name__=='__main__':unittest.main()
