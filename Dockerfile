ARG DEB_TAG=trixie-slim

# ─── Stage 0: ROCm 7.2 HIP runtime (Ubuntu Noble) ────────────────────────────
FROM ubuntu:noble AS rocm-libs

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget ca-certificates g++-14 \
 && wget -q https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y ./amdgpu-install_7.2.70200-1_all.deb \
 && amdgpu-install -y --usecase=rocm --no-dkms \
 && apt-get clean && rm -rf /var/lib/apt/lists/* amdgpu-install_7.2.70200-1_all.deb

# libhsakmt may land outside /opt/rocm on Ubuntu Noble — bring it in
RUN find /usr/lib -name 'libhsakmt.so*' -type f \
    | xargs -r -I{} cp {} /opt/rocm/lib/

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM docker.io/library/debian:${DEB_TAG} AS builder
ARG HIPFIRE_BRANCH=master

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git gcc build-essential pkg-config libssl-dev unzip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

RUN git clone --depth 1 --branch ${HIPFIRE_BRANCH} \
    https://github.com/Kaden-Schutt/hipfire.git /build
WORKDIR /build
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo build --release --features deltanet \
        --example daemon \
        --example infer \
        --example infer_hfq \
        --example triattn_validate \
        -p hipfire-runtime

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM docker.io/library/debian:${DEB_TAG} AS runtime

# Debian runtime deps (non-ROCm)
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates libdrm-amdgpu1 libdrm2 libnuma1 libelf1 zlib1g libzstd1 libstdc++6 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ROCm 7.2 HIP runtime — only the 6 libraries hipfire actually loads at runtime.
# (verified via /proc/<pid>/maps on a live gfx1151 container)
RUN mkdir -p /opt/rocm/lib
COPY --from=rocm-libs /opt/rocm/lib/libamdhip64.so*             /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libhsa-runtime64.so*        /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libamd_comgr.so*            /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libhsakmt.so*               /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/librocprofiler-register.so* /opt/rocm/lib/
COPY --from=rocm-libs /opt/rocm/lib/libhsa-amd-aqlprofile64.so* /opt/rocm/lib/
RUN echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf \
 && ldconfig \
 && ln -sf libamdhip64.so.7      /opt/rocm/lib/libamdhip64.so \
 && ln -sf libhsa-runtime64.so.1 /opt/rocm/lib/libhsa-runtime64.so \
 && ln -sf libamd_comgr.so.3     /opt/rocm/lib/libamd_comgr.so

# hipcc toolchain — clang-22 is statically linked (172MB), no extra LLVM .so needed
RUN mkdir -p /opt/rocm/bin /opt/rocm/lib/llvm/bin /opt/rocm/amdgcn /opt/rocm/include
COPY --from=rocm-libs /opt/rocm/bin/hipcc             /opt/rocm/bin/
COPY --from=rocm-libs /opt/rocm/lib/llvm/bin/clang-22 /opt/rocm/lib/llvm/bin/
RUN ln -sf clang-22 /opt/rocm/lib/llvm/bin/clang \
 && ln -sf clang    /opt/rocm/lib/llvm/bin/clang++ \
 && ln -sf /opt/rocm/bin/hipcc /usr/local/bin/hipcc
COPY --from=rocm-libs /opt/rocm/amdgcn/bitcode/ /opt/rocm/amdgcn/bitcode/
COPY --from=rocm-libs /opt/rocm/include/hip/    /opt/rocm/include/hip/

# Clang resource dir: built-in headers + bitcode symlink for kernel compilation
COPY --from=rocm-libs /opt/rocm/lib/llvm/lib/clang/22/include/ /opt/rocm/lib/llvm/lib/clang/22/include/
RUN mkdir -p /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn \
 && ln -s /opt/rocm/amdgcn/bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode

# C++ standard library headers (GCC 14) — required by hip_runtime.h via <cstdint>
COPY --from=rocm-libs /usr/include/c++/14/                     /usr/include/c++/14/
COPY --from=rocm-libs /usr/include/x86_64-linux-gnu/c++/14/    /usr/include/x86_64-linux-gnu/c++/14/
# GCC 14 installation dir — clang uses crtbegin.o to detect the GCC installation
COPY --from=rocm-libs /usr/lib/gcc/x86_64-linux-gnu/14/include/ /usr/lib/gcc/x86_64-linux-gnu/14/include/
COPY --from=rocm-libs /usr/lib/gcc/x86_64-linux-gnu/14/crtbegin.o /usr/lib/gcc/x86_64-linux-gnu/14/

# hipcc version file (suppresses startup warning)
COPY --from=rocm-libs /opt/rocm/share/hip/ /opt/rocm/share/hip/

ENV HIP_PATH=/opt/rocm

# Bun binary (single static binary, no install needed)
COPY --from=builder /root/.bun/bin/bun /usr/local/bin/bun

RUN mkdir -p /root/.hipfire/bin /root/.hipfire/models /root/.hipfire/cli

COPY --from=builder /build/target/release/examples/daemon           /root/.hipfire/bin/daemon
COPY --from=builder /build/target/release/examples/infer            /root/.hipfire/bin/infer
COPY --from=builder /build/target/release/examples/infer_hfq        /root/.hipfire/bin/infer_hfq
COPY --from=builder /build/target/release/examples/triattn_validate /root/.hipfire/bin/triattn_validate

COPY --from=builder /build/cli/ /root/.hipfire/cli/
RUN rm -rf /root/.hipfire/cli/node_modules \
           /root/.hipfire/cli/.gitignore \
           /root/.hipfire/cli/tsconfig.json \
           /root/.hipfire/cli/bun.lock

# Pre-compiled gfx1151 kernels — daemon looks in <exe_dir>/kernels/compiled/<arch>/
COPY kernels/gfx1151/ /root/.hipfire/bin/kernels/compiled/gfx1151/

RUN printf '#!/bin/bash\nset -e\nexec bun run "$HOME/.hipfire/cli/index.ts" "$@"\n' \
    > /root/.hipfire/bin/hipfire \
 && chmod +x /root/.hipfire/bin/hipfire \
 && ln -s /root/.hipfire/bin/hipfire /usr/local/bin/hipfire

RUN cat > /root/.hipfire/config.json << 'EOF'
{
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": 8192,
  "gpu_arch": "gfx1151",
  "host": "0.0.0.0",
  "port": 8080
}
EOF

ENV PATH="/root/.hipfire/bin:/usr/local/bin:$PATH"
WORKDIR /root/.hipfire

RUN printf '#!/bin/bash\nset -e\nln -sf /proc/1/fd/1 /root/.hipfire/serve.log\nexec hipfire serve "$@"\n' \
    > /entrypoint.sh \
 && chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["qwen3.6:27b"]
