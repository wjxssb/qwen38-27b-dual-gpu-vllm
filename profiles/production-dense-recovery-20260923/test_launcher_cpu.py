#!/usr/bin/env python3
"""CPU tests of failure/ownership boundaries; never call Docker/systemd/GPU."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('candidate_launcher',Path(__file__).with_name('launcher.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Boundaries(unittest.TestCase):
 def test_changed_file_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'x').write_bytes(b'good');rows={'x':{'bytes':4,'sha256':hashlib.sha256(b'good').hexdigest()}}
   m.verify_files(p,rows);(p/'x').write_bytes(b'evil')
   with self.assertRaisesRegex(RuntimeError,'manifest mismatch'):m.verify_files(p,rows)
 def test_symlink_rejected_even_if_bytes_match(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'real').write_bytes(b'x');(p/'x').symlink_to('real')
   with self.assertRaisesRegex(RuntimeError,'manifest mismatch'):m.verify_files(p,{'x':{'bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}})
 def fixture(self):
  r={'container_id':'a'*64,'run_id':'b'*32,'manifest_sha256':'c'*64,'profile':'eager-no-cache','image':'sha256:'+'d'*64}
  row={'Id':r['container_id'],'Image':r['image'],'Config':{'Labels':{m.LABEL+'.campaign':str(m.CAMPAIGN),m.LABEL+'.run':r['run_id'],m.LABEL+'.manifest':r['manifest_sha256'],m.LABEL+'.profile':r['profile']}}}
  return r,row
 def test_exact_owned_identity_accepted(self):
  r,row=self.fixture();m.validate_owned(r,row)
 def test_wrong_campaign_never_stopped(self):
  r,row=self.fixture();row['Config']['Labels'][m.LABEL+'.campaign']='/production'
  with self.assertRaisesRegex(RuntimeError,'ownership mismatch'):m.validate_owned(r,row)
 def test_reused_or_other_container_rejected(self):
  r,row=self.fixture();row['Id']='f'*64
  with self.assertRaisesRegex(RuntimeError,'ID mismatch'):m.validate_owned(r,row)
 def test_changed_owned_image_rejected(self):
  r,row=self.fixture();row['Image']='other'
  with self.assertRaisesRegex(RuntimeError,'ownership mismatch'):m.validate_owned(r,row)
 def test_flash_controller_rejected_without_gpu_process(self):
  with self.assertRaisesRegex(RuntimeError,'protected Flash'):m.reject_processes([(128822,['python3','/mnt/storage/ai/qwen38-flash-next-reap320/final-validation-20260913/tools/controller.py'])])
 def test_unrelated_cpu_process_allowed(self):m.reject_processes([(7,['python3','unrelated.py'])])
 def test_tool_shell_reference_not_mistaken_for_controller(self):m.reject_processes([(19,['bash','-c','cat /home/frank/ai/qwen38-flash-next-reap320/final-validation-20260913/PROGRESS.md'])])
 def test_gpu_recovery_action_rejected(self):
  rows='\n'.join(u+', 15500, 700, 0, '+('Reset' if i else 'None') for i,u in enumerate(m.GPU_UUIDS))
  with self.assertRaisesRegex(RuntimeError,'recovery action'):m.validate_gpu_rows(rows)
 def test_gpu_uuid_duplicate_rejected(self):
  rows='\n'.join([m.GPU_UUIDS[0]+', 15500, 700, 0, None']*2)
  with self.assertRaisesRegex(RuntimeError,'absent/duplicated'):m.validate_gpu_rows(rows)
 def test_existing_lease_owner_rejected(self):
  import fcntl
  original=m.LOCKS
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lease';p.touch()
   with p.open('r+') as owned:
    fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB);m.LOCKS=[p]
    try:
     with self.assertRaisesRegex(RuntimeError,'residency owner'):
      with m.leases():pass
    finally:m.LOCKS=original
 def test_incomplete_inventory_rejected_before_hashing(self):
  original=m.read
  try:
   m.read=lambda p:{'file_integrity':'NOT_PROVEN'}
   with self.assertRaisesRegex(RuntimeError,'incomplete'):m.verify_inventory({'model_revision':'x'})
  finally:m.read=original
 def test_p2p_current_hardware_identity_accepted(self):
  m.validate_p2p_hardware({'gpus':m.GPU_UUIDS,'driver':'610'},'\n'.join(u+', 610' for u in m.GPU_UUIDS))
 def test_p2p_driver_mismatch_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'driver differs'):m.validate_p2p_hardware({'gpus':m.GPU_UUIDS,'driver':'610'},'\n'.join(u+', 611' for u in m.GPU_UUIDS))
 def test_p2p_receipt_gpu_mismatch_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'receipt GPU identity'):m.validate_p2p_hardware({'gpus':['other'],'driver':'610'},'\n'.join(u+', 610' for u in m.GPU_UUIDS))
 def test_stale_boot_resume_rejected_before_lifecycle(self):
  with self.assertRaisesRegex(RuntimeError,'stale-boot'):m.validate_resume_boot({'boot_id':'old'},'new')
 def test_same_boot_resume_identity_accepted(self):m.validate_resume_boot({'boot_id':'current'},'current')
if __name__=='__main__':unittest.main()
