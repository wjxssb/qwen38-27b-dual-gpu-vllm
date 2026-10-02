"""Count observed completed synchronization boundaries; flag incomplete telemetry."""
import json
from collections import Counter

def observe(directory,start_epoch,end_epoch):
    roles={};low=int(start_epoch*1e9);high=int(end_epoch*1e9)
    for role in ('worker-rank-0','worker-rank-1'):
        events=Counter();stages=Counter();shapes=Counter();drops=0;malformed=0
        times=[];records=0
        for path in (directory/(role+'.jsonl.1'),directory/(role+'.jsonl')):
            if not path.exists():continue
            with path.open() as stream:
                for line in stream:
                    try:r=json.loads(line)
                    except (ValueError,TypeError):malformed+=1;continue
                    stamp=r.get('wall_time_ns',0)
                    if not low<=stamp<=high:continue
                    records+=1;times.append(stamp);events[r.get('event','unknown')]+=1
                    drops+=r.get('dropped_since_previous',0)
                    d=r.get('decoder',{})
                    if r.get('event')=='decoder_sync_boundary' and d.get('status')=='pass':
                        stages[d.get('stage','unknown')]+=1
                        shapes[str(d.get('shape'))]+=1
        roles[role]={'records':records,'events':dict(events),'completed_sync_boundaries':sum(stages.values()),'completed_sync_by_stage':dict(stages),'completed_sync_by_tensor_shape':dict(shapes),'reported_drops':drops,'malformed_lines_in_retained_files':malformed,'first_event_epoch':min(times)/1e9 if times else None,'last_event_epoch':max(times)/1e9 if times else None}
    return {'roles':roles,'scope':'actual completed model fault-surfacing barriers from retained telemetry, not all CUDA synchronization calls; counts are lower bounds if logs rotated or records dropped','window':[start_epoch,end_epoch]}
