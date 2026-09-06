# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:96a70bbb56acb6c4d22b3153b090ca322da927361c48a3129a1a258f4c702e73
FROM ${BASE_IMAGE} AS native-builder
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake=3.28.3-1build7 git=1:2.43.0-1ubuntu7.3 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=1 NVCC_THREADS=1 FLASHINFER_NVCC_THREADS=1 \
    CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/include:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/include:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/include
RUN uv venv --system-site-packages --python /usr/bin/python3 /opt/build-venv

WORKDIR /input
COPY checkouts/vllm/ /input/checkouts/vllm/
COPY checkouts/flashinfer/ /input/checkouts/flashinfer/
COPY overlay-graph006/ /input/overlay-graph006/
COPY tests/ /input/tests/
COPY build/inputs.json build/verify_inputs.py /input/build/
ARG INPUT_MANIFEST_SHA256
RUN test -n "${INPUT_MANIFEST_SHA256}" \
    && test "$(sha256sum /input/build/inputs.json | cut -d' ' -f1)" = "${INPUT_MANIFEST_SHA256}" \
    && /opt/build-venv/bin/python /input/build/verify_inputs.py

# Fetch by immutable object ID. No dependence on the old experiment checkout.
RUN git init /src/cutlass \
    && git -C /src/cutlass fetch --depth=1 https://github.com/NVIDIA/cutlass.git da5e086dab31d63815acafdac9a9c5893b1c69e2 \
    && git -C /src/cutlass checkout --detach FETCH_HEAD \
    && test "$(git -C /src/cutlass rev-parse HEAD)" = da5e086dab31d63815acafdac9a9c5893b1c69e2

RUN /opt/build-venv/bin/python /input/tests/test_nvfp4_layout_cpu.py \
      --vllm-source /input/checkouts/vllm --json-out /input/cpu-layout.json

RUN cmake -S /input/checkouts/vllm -B /build/native -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/opt/build-venv/bin/python \
      -DVLLM_CUTLASS_SRC_DIR=/src/cutlass \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCUDA_nvrtc_LIBRARY=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libnvrtc.so.12 \
      -DNVCC_THREADS=1 -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
      -DCMAKE_JOB_POOLS:STRING=compile=1 -DVLLM_BUILD_ONLY_C_STABLE_LIBTORCH=ON \
      2>&1 | tee /input/configure.log \
    && cmake --build /build/native --target _C_stable_libtorch -j 1 \
      2>&1 | tee /input/native-build.log \
    && cmake --install /build/native --prefix /stage/usr/local/lib/python3.12/dist-packages \
      --component _C_stable_libtorch \
    && install -D -m 0644 /input/checkouts/vllm/vllm/v1/attention/backends/flashinfer.py \
      /stage/opt/qwen38-runtime/source/rebase-layer1/vllm/v1/attention/backends/flashinfer.py \
    && cuobjdump --list-elf /stage/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so \
      | tee /input/cubins.txt \
    && grep -Fq '.sm_120a.cubin' /input/cubins.txt \
    && readelf -d /stage/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so \
      > /input/native-dynamic.txt \
    && ! grep -E 'RPATH|RUNPATH|libpython|/build|/src' /input/native-dynamic.txt \
    && mkdir -p /stage/opt/qwen38-runtime/build \
    && cp /input/{cpu-layout.json,configure.log,native-build.log,cubins.txt,native-dynamic.txt} \
      /stage/opt/qwen38-runtime/build/ \
    && cp /input/build/inputs.json /stage/opt/qwen38-runtime/build/inputs.json \
    && cd /stage \
    && sha256sum usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so \
      > opt/qwen38-runtime/build/installed.sha256

FROM ${BASE_IMAGE} AS runtime
USER root
COPY --from=native-builder /stage/ /
# Graph006 validated runtime overlay (12 files), pinned by build/inputs.json.
COPY overlay-graph006/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY overlay-graph006/ /opt/qwen38-runtime/source/graph006-overlay/
# Root supplies the final public release/qualification contract at this path.
# A missing release.json is a build error; this draft is not a qualified image.
COPY build/release.json /opt/qwen38-runtime/release.json
COPY build/verify_overlay.py /opt/qwen38-runtime/build/verify_overlay.py
COPY tests/test_nvfp4_layout_cpu.py /opt/qwen38-runtime/tests/test_nvfp4_layout_cpu.py
COPY checkouts/vllm/tests/ /opt/qwen38-runtime/test-source/tests/
COPY checkouts/vllm/vllm/utils/torch_utils.py /opt/qwen38-runtime/source/vllm/utils/torch_utils.py
COPY checkouts/vllm/tests/kernels/quantization/nvfp4_utils.py /opt/qwen38-runtime/source/tests/kernels/quantization/nvfp4_utils.py

RUN cd / && sha256sum --strict --check /opt/qwen38-runtime/build/installed.sha256 \
    && cd /usr/local/lib/python3.12/dist-packages \
    && sha256sum vllm/_C_stable_libtorch.abi3.so \
      vllm/distributed/device_communicators/shm_broadcast.py \
      vllm/stability_telemetry.py \
      vllm/v1/attention/backends/flashinfer.py \
      vllm/v1/engine/async_llm.py \
      vllm/v1/engine/core.py \
      vllm/v1/executor/multiproc_executor.py \
      vllm/v1/executor/tp_lifecycle.py \
      vllm/v1/spec_decode/llm_base_proposer.py \
      vllm/v1/spec_decode/sm120_mtp_decode_graph.py \
      vllm/v1/worker/gpu_model_runner.py \
      vllm/v1/worker/gpu_worker.py \
      vllm/v1/worker/sm120_graph_validation.py \
      > /opt/qwen38-runtime/build/final.sha256 \
    && sha256sum --strict --check /opt/qwen38-runtime/build/final.sha256 \
    && cd / && /usr/bin/python3 /opt/qwen38-runtime/build/verify_overlay.py
ENV MAX_JOBS=1 NVCC_THREADS=1 FLASHINFER_NVCC_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.title="Qwen3.8 Dense SM120 NVFP4 runtime" \
      ai.qwen38.runtime-abi="qwen38-sm120-v1" \
      ai.qwen38.vllm.base-commit="99a10304dce8945119bd0b1a072297803c52a749" \
      ai.qwen38.flashinfer.commit="9dc1b2495b40314dec8a8cde8cd7faf5c5206702" \
      ai.qwen38.release-manifest="/opt/qwen38-runtime/release.json"
