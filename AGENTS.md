# veloGB10 — Agent Guide

## Overview

A from-scratch Rust + CUDA inference engine specialized for NVIDIA GB10-based
systems (DGX Spark and compatible OEM machines). It supports Qwen/Hy3 model
families with GatedDeltaNet, GQA, dense, and MoE architectures, plus tensor
parallelism across 2/4 nodes.

## Project Layout

| Path | Purpose |
|------|---------|
| `src/` | Core inference engine: model loading, kernels, scheduling, networking, HTTP server |
| `src/bin/` | CLI binary entrypoints |
| `kernels/` | CUDA PTX source files |
| `native/` | C FFI shim for ibverbs TP transport |
| `scripts/` | Helper scripts: profiling, quantization, model conversion, TP launch |
| `tests/` | Rust integration and unit tests |
| `benches/` | Benchmarks |
| `build.rs` | Compiles `native/net_shim.c` and discovers PTX kernels |

## Build Commands

**Prerequisites**: CUDA toolkit, `nvcc`, `libibverbs`, `libcudart`, `libcuda`.

```bash
# Release build (compiles C shim + kernels via build.rs)
cargo build --release

# The main binary
cargo run --release --bin gb10_inference -- --help
```

## Test Commands

```bash
# Run tests that do not require a live GPU model
cargo test

# A specific test
cargo test --test dsv4_cpu_test
```

## Lint / Format

```bash
cargo fmt -- --check
cargo clippy --workspace -- -D warnings
```

## Key Conventions

- **GB10-only**: hard-coded `sm_121` architecture in `build.rs`.
- **No Python runtime**: load models via `--model-dir` pointing to safetensors.
- **Tensor parallelism**: `scripts/run_tp_node.sh` / `scripts/run_tp_server.sh`
  coordinate multi-node inference over ConnectX-7 RDMA.
- **Chat templates**: each model ships a `chat_template.jinja` rendered with
  `minijinja`.
- **PTX kernels**: runtime-loaded; `build.rs` fails loudly if `nvcc` fails.

## Common Gotchas

- Build will fail without CUDA/ibverbs libraries installed.
- `build.rs` is intentionally strict: stale PTX is not allowed; nvcc errors abort
  the build.
- Many integration tests require a real GB10 and a downloaded model directory.
- The project is not portable to non-GB10 NVIDIA GPUs.

## Deployment

Releases are built from a GB10 host and published as binaries on GitHub. For
local iteration, use `cargo build --release` directly. No Dagger/CI workflow is
included in this fork.
