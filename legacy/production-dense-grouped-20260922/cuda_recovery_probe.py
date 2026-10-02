#!/usr/bin/env python3
"""Bounded fresh-context CUDA write/read and direct peer integrity probe."""
import ctypes as C
import hashlib
import json
import os
import time

PTX = b'''.version 7.0
.target sm_50
.address_size 64
.visible .entry fill(.param .u64 ptr, .param .u32 count, .param .u32 seed) {
 .reg .b32 %r<8>; .reg .b64 %rd<4>; .reg .pred %p;
 ld.param.u64 %rd0, [ptr]; ld.param.u32 %r0, [count]; ld.param.u32 %r1, [seed];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4; setp.ge.u32 %p, %r5, %r0; @%p bra done;
 mul.lo.u32 %r6, %r5, 1664525; xor.b32 %r7, %r6, %r1;
 mul.wide.u32 %rd1, %r5, 4; add.u64 %rd2, %rd0, %rd1; st.global.u32 [%rd2], %r7;
 done: ret;
}'''
N = 262144

def main():
    cuda = C.CDLL('libcuda.so.1')
    def call(name, *args):
        code = getattr(cuda, name)(*args)
        if code: raise RuntimeError(f'{name}: CUDA {code}')
    call('cuInit', 0)
    count=C.c_int(); call('cuDeviceGetCount', C.byref(count))
    if count.value != 2: raise RuntimeError('expected exactly two visible CUDA GPUs')
    rows=[]; cases=[]
    for dev in range(2):
        bus=C.create_string_buffer(32); uid=(C.c_ubyte*16)()
        call('cuDeviceGetPCIBusId', bus, 32, dev); call('cuDeviceGetUuid_v2', C.byref(uid), dev)
        import uuid
        ctx=C.c_void_p(); call('cuCtxCreate_v2', C.byref(ctx), 0, dev)
        ptr=C.c_uint64(); call('cuMemAlloc_v2', C.byref(ptr), C.c_size_t(N*4))
        mod=C.c_void_p(); call('cuModuleLoadData', C.byref(mod), C.c_char_p(PTX))
        fun=C.c_void_p(); call('cuModuleGetFunction', C.byref(fun), mod, b'fill')
        rows.append(dict(dev=dev, uuid='GPU-'+str(uuid.UUID(bytes=bytes(uid))), pci=bus.value.decode(), ctx=ctx, ptr=ptr, mod=mod, fun=fun))
    for dev in range(2):
        access=C.c_int(); call('cuDeviceCanAccessPeer', C.byref(access), dev, 1-dev)
        if access.value != 1: raise RuntimeError('bidirectional peer capability absent')
        call('cuCtxSetCurrent', rows[dev]['ctx']); call('cuCtxEnablePeerAccess', rows[1-dev]['ctx'], 0)
    def fill(writer, target, seed):
        call('cuCtxSetCurrent', rows[writer]['ctx'])
        size=C.c_uint(N); salt=C.c_uint(seed)
        args=(C.c_void_p*3)(C.addressof(rows[target]['ptr']), C.addressof(size), C.addressof(salt))
        call('cuLaunchKernel', rows[writer]['fun'], (N+255)//256,1,1,256,1,1,0,C.c_void_p(),args,C.c_void_p())
        call('cuCtxSynchronize')
    def verify(dev, seed, kind, source):
        call('cuCtxSetCurrent', rows[dev]['ctx'])
        host=(C.c_uint32*N)(); call('cuMemcpyDtoH_v2', host, rows[dev]['ptr'], C.c_size_t(N*4))
        expected=(C.c_uint32*N)(*((i*1664525)^seed for i in range(N)))
        if bytes(host) != bytes(expected): raise RuntimeError(f'{kind} {source}->{dev}: integrity mismatch')
        cases.append(dict(kind=kind, source=source, destination=dev, bytes=N*4, seed=seed, sha256=hashlib.sha256(bytes(host)).hexdigest(), mismatch_count=0))
    for dev in range(2):
        for seed in [0, 0xffffffff, 0xa5a55a5a, 0x12345678]:
            fill(dev, dev, seed); verify(dev, seed, 'local_alloc_kernel_write_read', dev)
    for repeat in range(8):
        for src in range(2):
            dst=1-src; seed=(0x9e3779b9*(repeat+1)+src)&0xffffffff
            fill(src, src, seed)
            call('cuMemcpyPeer', rows[dst]['ptr'], rows[dst]['ctx'], rows[src]['ptr'], rows[src]['ctx'], C.c_size_t(N*4))
            call('cuCtxSetCurrent', rows[dst]['ctx']); call('cuCtxSynchronize')
            verify(dst, seed, 'peer_copy', src)
            fill(src, dst, seed^0xffffffff)
            verify(dst, seed^0xffffffff, 'direct_peer_kernel_write', src)
    for row in rows:
        call('cuCtxSetCurrent', row['ctx']); call('cuMemFree_v2', row['ptr']); call('cuModuleUnload', row['mod']); call('cuCtxDestroy_v2', row['ctx'])
    return dict(status='PASS', scope='RECOVERY_ONLY_NOT_MODEL_QUALIFICATION', pid=os.getpid(), finished_epoch=time.time(),
                devices=[{k:r[k] for k in ('dev','uuid','pci')} for r in rows], cases=cases, contexts_destroyed=True)

if __name__ == '__main__':
    try: print(json.dumps(main()), flush=True)
    except BaseException as error:
        print(json.dumps(dict(status='FAIL', error=str(error), pid=os.getpid())), flush=True)
        os._exit(1)  # Never make further CUDA calls after a fatal probe failure.
