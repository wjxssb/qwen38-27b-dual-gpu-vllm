#!/usr/bin/env python3
"""Publish pre-collected immutable evidence; no Docker/GPU/process mutation."""
import argparse, hashlib, json, os, shutil, tempfile
from pathlib import Path
import launcher

def copy_sealed(source,target):
 source=Path(source)
 if source.is_symlink() or not source.is_file():raise RuntimeError('source must be a regular non-symlink file')
 raw=source.read_bytes();json.loads(raw)
 with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 target.chmod(0o444)
 return {'path':target.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def publish(args):
 campaign=Path(args.campaign).resolve();intent=json.loads(Path(args.old_intent).read_text())
 if campaign!=launcher.STORAGE_ROOT.resolve():raise RuntimeError('publication artifact root differs from active campaign storage')
 if intent.get('container_labels',{}).get(launcher.LABEL+'.campaign')!=str(launcher.CAMPAIGN):raise RuntimeError('archived intent campaign differs from publication identity')
 destination=launcher.reconciliation_path(intent,args.current_boot,campaign).parent
 if destination.exists():raise RuntimeError('reconciliation destination already exists; immutable overwrite refused')
 parent=campaign/'prior-boot-reconciliations';parent.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.staging-',dir=parent) as name:
  staging=Path(name);artifacts={}
  for kind in ['old_intent','old_container','current_observation','current_p2p','source_observation']:
   source=getattr(args,kind,None)
   if source:artifacts[kind]=copy_sealed(source,staging/(kind+'.json'))
  receipt={'schema':1,'kind':'PRIOR_BOOT_RECONCILIATION','scope':'ALLOW_FRESH_INSTANCE_ON_CURRENT_BOOT_ONLY','old_boot_cleanup':'NOT_PROVEN','old_incident_preserved':True,'current_boot_id':args.current_boot,'old_identity':{k:intent[k] for k in launcher.PRIOR_BOOT_IDENTITY_FIELDS},'artifacts':artifacts}
  path=staging/'reconciliation.json'
  with path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  path.chmod(0o444)
  launcher.validate_prior_boot_reconciliation(intent,args.current_boot,path)
  destination.mkdir(parents=True,exist_ok=False)
  # Link the validated receipt last: partial publication cannot grant admission.
  for artifact in artifacts.values():os.link(staging/artifact['path'],destination/artifact['path'])
  os.link(path,destination/path.name)
  fd=os.open(destination,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return {'status':'PRIOR_BOOT_RECEIPT_PUBLISHED','path':str(destination/path.name),'sha256':hashlib.sha256((destination/path.name).read_bytes()).hexdigest(),'old_boot_cleanup':'NOT_PROVEN','scope':receipt['scope'],'note':'Admission still performs fresh exact old-container reconciliation and all current-boot preflight checks.'}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--campaign',default=str(Path(__file__).resolve().parent.parent));ap.add_argument('--current-boot',required=True)
 for name in ['old-intent','old-container','current-observation','current-p2p']:ap.add_argument('--'+name,required=True)
 ap.add_argument('--source-observation');args=ap.parse_args();print(json.dumps(publish(args),indent=2))
if __name__=='__main__':main()
