"""Do not recover an inconsistent CUDA context as an unsupported tactic."""
import re
import sys

_FATAL = re.compile(
    r'illegal[ _]memory[ _]access|illegal[ _]instruction|misaligned[ _]address|'
    r'device[- ]side assert|launch timed out|launch timeout|'
    r'unspecified launch failure|cudaError(?:IllegalAddress|LaunchTimeout|Assert|'
    r'HardwareStackError|IllegalInstruction|MisalignedAddress|InvalidAddressSpace|'
    r'InvalidPc|LaunchFailure|TensorMemoryLeak|Contained|MpsClientTerminated)|'
    r'CUDA_ERROR_(?:ILLEGAL_ADDRESS|LAUNCH_TIMEOUT|ASSERT|ILLEGAL_INSTRUCTION|'
    r'MISALIGNED_ADDRESS|LAUNCH_FAILED)|CUDNN_STATUS_EXECUTION_FAILED|'
    r'ncclUnhandledCudaError|fallen off the bus', re.I)


def is_fatal_profile_error(error):
    if not isinstance(error, Exception):
        return True  # Shutdown and cancellation must not enter a GPU collective.
    seen = set()
    current = error
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if _FATAL.search(str(current)):
            return True
        current = current.__cause__ or current.__context__
    return False


def raise_if_fatal(error, **context):
    if not is_fatal_profile_error(error):
        return
    # vLLM is optional. Use only an already-loaded CPU telemetry module.
    telemetry = sys.modules.get('vllm.stability_telemetry')
    try:
        if telemetry is not None:
            telemetry.record_first_failure('FLASHINFER_FATAL_PROFILE', error,
                fatal_profile=True, **context)
    finally:
        raise error
