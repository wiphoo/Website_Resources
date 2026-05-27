# Milestone 1.2: Benchmark Machine Metadata Capture

## Goal

Extend the BGE-M3 benchmark harness so every benchmark result stores detailed metadata about the machine running the test.

This milestone focuses on **CPU metadata first**, with placeholders for **GPU metadata** to support future CUDA/NVIDIA runs.

Baseline target remains:

```text
Model: BAAI/bge-m3
Runtime: ONNX Runtime
Provider: CPUExecutionProvider
Precision: FP32
Package manager: uv
Benchmark mode: tokenization-only, embedding-only, end-to-end
Datasets: English, Thai, mixed English/Thai
```

## Why this milestone matters

Raw benchmark numbers are not useful unless we know the hardware and runtime context.

This milestone makes each JSONL row self-contained, so results can be compared across:

```text
Intel Xeon vs AMD EPYC
AVX2 vs AVX512
small VM vs bare metal
laptop CPU vs server CPU
CPU-only vs future GPU-enabled machines
local run vs Docker vs Kubernetes
```

## Scope

### Included

```text
CPU model metadata
CPU topology metadata
CPU clock metadata
CPU feature flags
Memory metadata
OS/kernel metadata
Python/runtime metadata
ONNX Runtime provider metadata
Container/Kubernetes metadata if available
GPU placeholder fields
JSON result schema update
Notebook metadata inspection table
```

### Excluded

```text
GPU benchmark execution
CUDA benchmark execution
OpenVINO benchmark execution
INT8 benchmark execution
Prometheus exporter
Kubernetes Job implementation
```

## Metadata design principle

Every benchmark JSON row should be self-contained.

For Milestone 1.2, repeat key metadata in every JSONL row. Later milestones can normalize metadata into separate tables, but denormalized JSONL is simpler for the MVP.

## Result schema addition

Add a nested field:

```json
{
  "machine_metadata": {}
}
```

Also add flattened top-level fields for notebook filtering.

Example:

```json
{
  "model": "BAAI/bge-m3",
  "runtime": "onnxruntime",
  "provider": "CPUExecutionProvider",
  "precision": "fp32",
  "dataset": "mixed",
  "batch_size": 32,
  "max_length": 512,

  "tokenize_tokens_per_sec": 1200000.0,
  "embedding_tokens_per_sec": 250000.0,
  "end_to_end_tokens_per_sec": 210000.0,

  "machine_hostname": "bench-xeon-01",
  "cpu_model_name": "Intel(R) Xeon(R) Gold 5420+",
  "cpu_vendor_id": "GenuineIntel",
  "cpu_architecture": "x86_64",
  "cpu_physical_cores": 28,
  "cpu_logical_cores": 56,
  "cpu_max_mhz": 4100.0,
  "cpu_has_avx": true,
  "cpu_has_avx2": true,
  "cpu_has_avx512": true,
  "cpu_has_avx512_vnni": true,
  "cpu_has_amx_bf16": true,
  "cpu_has_amx_int8": true,
  "memory_total_bytes": 135000000000,
  "is_container": true,
  "is_kubernetes": false,
  "gpu_available": false,
  "gpu_count": 0,

  "machine_metadata": {
    "schema_version": "1.0",
    "host": {},
    "cpu": {},
    "memory": {},
    "runtime": {},
    "container": {},
    "kubernetes": {},
    "gpu": {}
  }
}
```

## Required CPU metadata fields

```text
cpu.model_name
cpu.vendor_id
cpu.architecture
cpu.sockets
cpu.physical_cores
cpu.logical_cores
cpu.threads_per_core
cpu.cores_per_socket
cpu.min_mhz
cpu.max_mhz
cpu.current_mhz
cpu.cache_l1d
cpu.cache_l1i
cpu.cache_l2
cpu.cache_l3
cpu.flags
cpu.features.avx
cpu.features.avx2
cpu.features.avx512
cpu.features.avx512_vnni
cpu.features.avx512_bf16
cpu.features.amx_bf16
cpu.features.amx_int8
cpu.features.fma
cpu.features.sse4_2
```

## CPU feature normalization

Normalize raw CPU flags into stable booleans.

```python
def has_any_flag(flags: set[str], candidates: list[str]) -> bool:
    return any(flag in flags for flag in candidates)


features = {
    "avx": "avx" in flags,
    "avx2": "avx2" in flags,
    "avx512": has_any_flag(flags, ["avx512f"]),
    "avx512_vnni": "avx512_vnni" in flags,
    "avx512_bf16": "avx512_bf16" in flags,
    "amx_bf16": "amx_bf16" in flags,
    "amx_int8": "amx_int8" in flags,
    "fma": "fma" in flags,
    "sse4_2": "sse4_2" in flags,
}
```

## GPU placeholder design

Milestone 1.2 should not require GPU. The schema should already support future GPU metadata.

CPU-only placeholder:

```json
{
  "gpu": {
    "available": false,
    "count": 0,
    "devices": []
  }
}
```

Future NVIDIA example:

```json
{
  "gpu": {
    "available": true,
    "count": 1,
    "devices": [
      {
        "index": 0,
        "name": "NVIDIA L40",
        "uuid": "GPU-...",
        "driver_version": "550.xx",
        "cuda_version": "12.x",
        "memory_total_bytes": 48305799168,
        "power_limit_watts": 300,
        "compute_capability": "8.9"
      }
    ]
  }
}
```

If `nvidia-smi` is unavailable, do not fail the benchmark.

## Container metadata

Capture whether the script is running inside a container.

Fields:

```text
container.is_container
container.container_runtime_hint
container.cgroup_version
container.cpu_quota
container.cpu_cpuset
container.memory_limit_bytes
```

Useful files:

```text
/proc/1/cgroup
/sys/fs/cgroup/cpu.max
/sys/fs/cgroup/cpuset.cpus
/sys/fs/cgroup/memory.max
```

This matters because Kubernetes CPU limits can change benchmark results dramatically.

## Kubernetes metadata

Read Kubernetes metadata from environment variables when available.

Recommended env vars for the future Kubernetes Job:

```yaml
env:
  - name: KUBERNETES_NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
  - name: KUBERNETES_POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: KUBERNETES_NODE_NAME
    valueFrom:
      fieldRef:
        fieldPath: spec.nodeName
  - name: KUBERNETES_POD_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
```

Milestone 1.2 should read:

```text
KUBERNETES_NAMESPACE
KUBERNETES_POD_NAME
KUBERNETES_NODE_NAME
KUBERNETES_POD_IP
```

## Task 1: Add machine metadata module

Create:

```text
scripts/machine_metadata.py
```

Main function:

```python
def collect_machine_metadata(session=None) -> dict:
    return {
        "schema_version": "1.0",
        "collected_at_unix": time.time(),
        "host": collect_host_metadata(),
        "cpu": collect_cpu_metadata(),
        "memory": collect_memory_metadata(),
        "runtime": collect_runtime_metadata(session=session),
        "container": collect_container_metadata(),
        "kubernetes": collect_kubernetes_metadata(),
        "gpu": collect_gpu_metadata(),
    }
```

Helper functions:

```python
def collect_host_metadata() -> dict:
    ...

def collect_cpu_metadata() -> dict:
    ...

def collect_memory_metadata() -> dict:
    ...

def collect_runtime_metadata(session=None) -> dict:
    ...

def collect_container_metadata() -> dict:
    ...

def collect_kubernetes_metadata() -> dict:
    ...

def collect_gpu_metadata() -> dict:
    ...
```

## Task 2: CPU metadata implementation plan

Use Linux-first logic:

```text
Primary:
  lscpu --json

Fallback:
  /proc/cpuinfo
  os.cpu_count()
  platform module
```

Recommended command helper:

```python
import json
import shutil
import subprocess


def run_command(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def collect_lscpu_json() -> dict | None:
    if shutil.which("lscpu") is None:
        return None

    code, stdout, _ = run_command(["lscpu", "--json"])
    if code != 0:
        return None

    return json.loads(stdout)
```

Fallback for CPU flags:

```python
def collect_cpu_flags_from_proc() -> set[str]:
    flags = set()

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    _, value = line.split(":", 1)
                    flags.update(value.strip().split())
                    break
    except FileNotFoundError:
        pass

    return flags
```

## Task 3: Memory metadata implementation plan

```python
def collect_memory_metadata() -> dict:
    result = {
        "total_bytes": None,
        "available_bytes": None,
    }

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            meminfo = {}
            for line in f:
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()

        def kb_value(name: str):
            raw = meminfo.get(name)
            if not raw:
                return None
            return int(raw.split()[0]) * 1024

        result["total_bytes"] = kb_value("MemTotal")
        result["available_bytes"] = kb_value("MemAvailable")
    except FileNotFoundError:
        pass

    return result
```

## Task 4: Runtime metadata implementation plan

```python
import platform
import sys

import numpy as np
import onnxruntime as ort
import transformers


def collect_runtime_metadata(session=None) -> dict:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "transformers_version": transformers.__version__,
        "available_providers": ort.get_available_providers(),
        "active_providers": session.get_providers() if session else None,
    }
```

## Task 5: Container metadata implementation plan

```python
from pathlib import Path


def read_text_or_none(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return text if text else None


def is_container() -> bool:
    if Path("/.dockerenv").exists():
        return True

    text = read_text_or_none("/proc/1/cgroup")
    if not text:
        return False

    return any(token in text for token in ["docker", "containerd", "kubepods"])
```

Cgroup fields:

```text
/sys/fs/cgroup/cpu.max
/sys/fs/cgroup/cpuset.cpus
/sys/fs/cgroup/memory.max
```

Convert memory limit:

```text
If value is "max":
  memory_limit_bytes = null

Else:
  memory_limit_bytes = int(value)
```

## Task 6: Kubernetes metadata implementation plan

```python
import os
from pathlib import Path


def collect_kubernetes_metadata() -> dict:
    namespace = os.getenv("KUBERNETES_NAMESPACE")
    pod_name = os.getenv("KUBERNETES_POD_NAME")
    node_name = os.getenv("KUBERNETES_NODE_NAME")
    pod_ip = os.getenv("KUBERNETES_POD_IP")

    service_account_exists = Path(
        "/var/run/secrets/kubernetes.io/serviceaccount"
    ).exists()

    is_kubernetes = any([namespace, pod_name, node_name, pod_ip]) or service_account_exists

    return {
        "is_kubernetes": is_kubernetes,
        "namespace": namespace,
        "pod_name": pod_name,
        "node_name": node_name,
        "pod_ip": pod_ip,
    }
```

## Task 7: GPU placeholder implementation plan

```python
def collect_gpu_metadata() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {
            "available": False,
            "count": 0,
            "devices": [],
        }

    code, stdout, _ = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ])

    if code != 0:
        return {
            "available": False,
            "count": 0,
            "devices": [],
        }

    devices = []
    for line in stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue

        index, name, uuid, driver_version, memory_total_mb, power_limit_watts = parts

        devices.append({
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "driver_version": driver_version,
            "memory_total_bytes": int(float(memory_total_mb) * 1024 * 1024),
            "power_limit_watts": float(power_limit_watts),
        })

    return {
        "available": bool(devices),
        "count": len(devices),
        "devices": devices,
    }
```

## Task 8: Integrate metadata into benchmark script

Update:

```text
scripts/bench_onnx_cpu_fp32_stage_breakdown.py
```

Add:

```python
from machine_metadata import collect_machine_metadata
```

After creating the ONNX Runtime session:

```python
machine_metadata = collect_machine_metadata(session=session)
```

Add to final result:

```python
result["machine_metadata"] = machine_metadata
```

## Task 9: Add flattened metadata fields

Nested metadata is good for completeness. Flatten common fields for analysis.

Add helper:

```python
def flatten_machine_metadata(metadata: dict) -> dict:
    cpu = metadata.get("cpu", {})
    features = cpu.get("features", {})
    host = metadata.get("host", {})
    memory = metadata.get("memory", {})
    container = metadata.get("container", {})
    kubernetes = metadata.get("kubernetes", {})
    gpu = metadata.get("gpu", {})

    return {
        "machine_hostname": host.get("hostname"),
        "cpu_model_name": cpu.get("model_name"),
        "cpu_vendor_id": cpu.get("vendor_id"),
        "cpu_architecture": cpu.get("architecture"),
        "cpu_physical_cores": cpu.get("physical_cores"),
        "cpu_logical_cores": cpu.get("logical_cores"),
        "cpu_max_mhz": cpu.get("max_mhz"),
        "cpu_current_mhz": cpu.get("current_mhz"),
        "cpu_has_avx": features.get("avx"),
        "cpu_has_avx2": features.get("avx2"),
        "cpu_has_avx512": features.get("avx512"),
        "cpu_has_avx512_vnni": features.get("avx512_vnni"),
        "cpu_has_amx_bf16": features.get("amx_bf16"),
        "cpu_has_amx_int8": features.get("amx_int8"),
        "memory_total_bytes": memory.get("total_bytes"),
        "memory_available_bytes": memory.get("available_bytes"),
        "is_container": container.get("is_container"),
        "is_kubernetes": kubernetes.get("is_kubernetes"),
        "gpu_available": gpu.get("available"),
        "gpu_count": gpu.get("count"),
    }
```

Then:

```python
result.update(flatten_machine_metadata(machine_metadata))
```

## Task 10: Notebook updates

Update:

```text
notebooks/plot_onnx_cpu_fp32_stage_breakdown.ipynb
```

Add metadata summary cell:

```python
metadata_cols = [
    "machine_hostname",
    "cpu_model_name",
    "cpu_vendor_id",
    "cpu_architecture",
    "cpu_physical_cores",
    "cpu_logical_cores",
    "cpu_max_mhz",
    "cpu_current_mhz",
    "cpu_has_avx",
    "cpu_has_avx2",
    "cpu_has_avx512",
    "cpu_has_avx512_vnni",
    "cpu_has_amx_bf16",
    "cpu_has_amx_int8",
    "memory_total_bytes",
    "is_container",
    "is_kubernetes",
    "gpu_available",
    "gpu_count",
]

df[metadata_cols].drop_duplicates()
```

Add metadata-aware summary:

```python
summary = (
    df.groupby(
        [
            "cpu_model_name",
            "cpu_physical_cores",
            "cpu_logical_cores",
            "cpu_has_avx2",
            "cpu_has_avx512",
            "dataset",
            "batch_size",
            "max_length",
        ],
        as_index=False,
    )
    .agg(
        embedding_tokens_per_sec=("embedding_tokens_per_sec", "median"),
        end_to_end_tokens_per_sec=("end_to_end_tokens_per_sec", "median"),
        embedding_latency_ms_p95=("embedding_latency_ms_p95", "median"),
    )
)

summary
```

## Smoke Test

Run:

```bash
uv run python scripts/bench_onnx_cpu_fp32_stage_breakdown.py   --model-dir models/bge-m3-fp32   --dataset th   --batch-size 4   --max-length 128   --target-words 64   --warmup 1   --batches 2   --out results/onnx_cpu_fp32_stage_breakdown.jsonl
```

Inspect nested CPU metadata:

```bash
tail -n 1 results/onnx_cpu_fp32_stage_breakdown.jsonl | jq '.machine_metadata.cpu'
```

Inspect flattened fields:

```bash
tail -n 1 results/onnx_cpu_fp32_stage_breakdown.jsonl | jq '{
  cpu_model_name,
  cpu_physical_cores,
  cpu_logical_cores,
  cpu_has_avx2,
  cpu_has_avx512,
  memory_total_bytes,
  gpu_available
}'
```

## Definition of Done

Milestone 1.2 is complete when:

```text
- scripts/machine_metadata.py exists.
- collect_machine_metadata() returns host, CPU, memory, runtime, container, Kubernetes, and GPU-placeholder metadata.
- Benchmark JSONL includes nested machine_metadata.
- Benchmark JSONL includes flattened metadata fields.
- CPU model name is captured.
- CPU physical/logical cores are captured.
- CPU max/current clock fields are captured when available.
- CPU feature flags are captured.
- AVX/AVX2/AVX512/AMX helper booleans are captured.
- Memory total/available bytes are captured.
- Container metadata is captured if running inside Docker/containerd.
- Kubernetes metadata is captured if environment variables are present.
- GPU placeholder fields exist even on CPU-only machines.
- Notebook can display a metadata summary table.
```

## Interpretation Guide

Use metadata to explain benchmark differences.

```text
Machine A has higher embedding_tokens_per_sec than Machine B:
  Check CPU model, core count, max clock, AVX2/AVX512, and memory.

Machine performs worse inside Kubernetes than locally:
  Check container CPU quota, cpuset, and memory limit.

AMD x86 and Intel x86 differ:
  Check AVX512 availability and CPU clock.

Cloud VM varies across runs:
  Check whether model is running under CPU quota or shared noisy-neighbor hardware.

Future GPU results differ:
  Check GPU name, driver version, CUDA version, VRAM, and power limit.
```

## Next Milestone

```text
Milestone 1.3:
  Add ONNX CPU INT8 comparison using the same metadata schema.

Milestone 1.4:
  Add ONNX CUDA FP32/FP16 with GPU metadata enabled.

Milestone 1.5:
  Add Docker image and Kubernetes Job using metadata env vars.
```
