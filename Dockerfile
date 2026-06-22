ARG DEB_TAG=trixie-slim
FROM docker.io/library/debian:${DEB_TAG} AS builder
ARG HIPFIRE_BRANCH=master

RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git gcc build-essential pkg-config libssl-dev unzip \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:$PATH"

# Bun (static binary, copy to runtime stage)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# hipfire source + build
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

# ─── Runtime ────────────────────────────────────────────────────────────────

FROM docker.io/library/debian:${DEB_TAG} AS runtime

# HIP runtime (dlopen'd by hip-bridge at runtime)
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libamdhip64-5 libamd-comgr2 libhsa-runtime64-1 libnuma1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Bun binary (single static binary, no install needed)
COPY --from=builder /root/.bun/bin/bun /usr/local/bin/bun

# Mirror install.sh structure under ~/.hipfire
RUN mkdir -p /root/.hipfire/bin /root/.hipfire/models /root/.hipfire/cli

# Inference engine binaries
COPY --from=builder /build/target/release/examples/daemon          /root/.hipfire/bin/daemon
COPY --from=builder /build/target/release/examples/infer           /root/.hipfire/bin/infer
COPY --from=builder /build/target/release/examples/infer_hfq       /root/.hipfire/bin/infer_hfq
COPY --from=builder /build/target/release/examples/triattn_validate /root/.hipfire/bin/triattn_validate

# TypeScript CLI (same pruning as install.sh)
COPY --from=builder /build/cli/ /root/.hipfire/cli/
RUN rm -rf /root/.hipfire/cli/node_modules \
           /root/.hipfire/cli/.gitignore \
           /root/.hipfire/cli/tsconfig.json \
           /root/.hipfire/cli/bun.lock

# Pre-compiled gfx1151 kernels (.hip sources + .hsaco binaries + .hash)
# Placed in WORKDIR-relative .hipfire_kernels/ — daemon probes {CWD}/.hipfire_kernels/{arch}/
COPY kernels/gfx1151/ /root/.hipfire/.hipfire_kernels/gfx1151/

# hipfire CLI wrapper (same as install.sh generates)
RUN printf '#!/bin/bash\nset -e\nexec bun run "$HOME/.hipfire/cli/index.ts" "$@"\n' \
    > /root/.hipfire/bin/hipfire \
 && chmod +x /root/.hipfire/bin/hipfire \
 && ln -s /root/.hipfire/bin/hipfire /usr/local/bin/hipfire

# Default config — host/port baked in so Deployment args only need the model tag
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

# WORKDIR = CWD for daemon kernel discovery (.hipfire_kernels/ is relative)
WORKDIR /root/.hipfire

# Entrypoint: symlink serve.log → stdout then run
RUN printf '#!/bin/bash\nset -e\nln -sf /proc/1/fd/1 /root/.hipfire/serve.log\nexec hipfire serve "$@"\n' \
    > /entrypoint.sh \
 && chmod +x /entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
CMD ["qwen3.6:27b"]
