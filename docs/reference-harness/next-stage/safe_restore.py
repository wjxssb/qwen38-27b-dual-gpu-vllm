"""Qualification rollback waits for hardware recovery; never stops its probes."""
from common import *
import lifecycle

def recovery_busy(state, boot_id):
    if state.get('boot_id') != boot_id:
        return False
    return state.get('state') in ('RECOVERING', 'RETRY_WAIT')

def wait_idle(candidate, timeout=1200):
    deadline=time.monotonic()+timeout
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    last='not observed'
    while time.monotonic()<deadline:
        state=read(PROD/'gateway/state/recovery-status.json')
        if recovery_busy(state,boot_id):
            last='hardware recovery '+state['state'];time.sleep(3);continue
        # A first-failure marker can precede the monitor's next poll. In that
        # interval an HTTP endpoint may still respond. Wait for the replacement
        # instance; never interrupt the incident's probe/observation sequence.
        try:
            a=active(candidate)
            markers=list((candidate/'runs'/a['instance_id']/'logs').glob('first-failure-*.json'))
            if markers:
                last='first-failure in current instance '+a['instance_id'];time.sleep(3);continue
            assert_idle(candidate)
        except (OSError,AssertionError,subprocess.SubprocessError) as error:
            last=repr(error);time.sleep(3);continue
        return {'state':'IDLE','candidate':str(candidate),'instance_id':a['instance_id'],'recovery':state,'epoch':time.time()}
    raise TimeoutError('WAITING_FOR_RECOVERY_OR_IDLE: '+last)

def restore(fallback, force_restart=False):
    from fault_test import smoke
    current=lifecycle.selected()
    observed=wait_idle(current)
    if current!=fallback or force_restart:
        lifecycle.activate(fallback)
    return {'idle_observation':observed,'smoke':smoke()}
