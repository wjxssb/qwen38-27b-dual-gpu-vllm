"""Sealed stat identity fast path; any changed identity falls back to full SHA.

This does not replace the manifest check. The launcher verifies this module and
receipt as release files before using them. Designed for immutable local model
files, not an adversarial writer who can replace the trusted release itself.
"""
import os,stat,json,hashlib
from pathlib import Path
FIELDS=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def identity(st):return {k:getattr(st,k) for k in FIELDS}
def snapshot_file(path,expected_hash,expected_size):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
 try:
  before=os.fstat(fd)
  if not stat.S_ISREG(before.st_mode) or before.st_size!=expected_size:raise RuntimeError('Invalid model file size/type')
  h=hashlib.sha256()
  while block:=os.read(fd,8*1024*1024):h.update(block)
  after=os.fstat(fd)
  if identity(before)!=identity(after) or h.hexdigest()!=expected_hash:raise RuntimeError('Model changed during SHA verification')
  # A rename/replacement of the pathname is not accepted just because the old fd matched.
  if identity(path.lstat())!=identity(after):raise RuntimeError('Model pathname changed during SHA verification')
  return dict(identity(after),sha256=h.hexdigest())
 finally:os.close(fd)
def matches(root,receipt,expected):
 """No writes and no CUDA. False means invoke the original verifier in full."""
 try:
  if receipt['schema']!=1 or receipt['inputs']!=expected:return False
  root=Path(root).resolve()
  if str(root)!=receipt['model_root']:return False
  for rel,row in receipt['files'].items():
   p=root/rel
   if p.is_symlink() or not p.resolve().is_relative_to(root):return False
   st=p.lstat()
   if not stat.S_ISREG(st.st_mode) or identity(st)!={k:row[k] for k in FIELDS}:return False
  return True
 except (OSError,ValueError,TypeError,KeyError):return False

def seal(root,inventory,inputs):
 root=Path(root).resolve();rows={}
 if Path(inventory['model_directory']).resolve()!=root:raise RuntimeError('Model root mismatch')
 if inventory['file_integrity']!='PASS' or inventory['verified_file_count']!=inventory['expected_file_count']:raise RuntimeError('Inventory incomplete')
 for row in inventory['files']:
  p=Path(row['path']);rel=p.relative_to(Path(inventory['model_directory']))
  if row.get('integrity')!='PASS' or not p.resolve().is_relative_to(root):raise RuntimeError('Invalid inventory row')
  if str(rel) in rows:raise RuntimeError('Duplicate inventory entry')
  rows[str(rel)]=snapshot_file(root/rel,row['sha256'],row['size_bytes'])
 if len(rows)!=inventory['expected_file_count']:raise RuntimeError('File count mismatch')
 return {'schema':1,'inputs':inputs,'model_root':str(root),'files':rows,'scope':'Full SHA256 at sealing, dev/inode/size/mtime_ns/ctime_ns on reload; changed identity invokes original full SHA verifier.'}
