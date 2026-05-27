import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


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


def has_any_flag(flags: set[str], candidates: list[str]) -> bool:
    return any(flag in flags for flag in candidates)


def collect_host_metadata() -> dict:
    hostname = platform.node()
    return {
        "hostname": hostname,
        "platform": platform.platform(),
    }


def collect_cpu_metadata() -> dict:
    lscpu_data = collect_lscpu_json()
    flags = collect_cpu_flags_from_proc()

    fields = {}
    if lscpu_data:
        items = lscpu_data.get("lscpu", []) if isinstance(lscpu_data, dict) else lscpu_data
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "field" in item and "data" in item:
                    key = item["field"].rstrip(":").strip()
                    fields[key] = item["data"]

    def get_field(name: str, default=None):
        return fields.get(name, default)

    if fields:
        threads_per_core_str = get_field("Thread(s) per core", "1")
        cores_per_socket_str = get_field("Core(s) per socket", "0")
        sockets_str = get_field("Socket(s)", "1")

        threads_per_core = int(threads_per_core_str) if threads_per_core_str else 1
        cores_per_socket = int(cores_per_socket_str) if cores_per_socket_str else 0
        sockets = int(sockets_str) if sockets_str else 1
        physical_cores = cores_per_socket * sockets

        max_mhz_str = get_field("CPU max MHz", "0")
        min_mhz_str = get_field("CPU min MHz", "0")
        current_mhz_str = get_field("CPU MHz", "0")

        result = {
            "model_name": get_field("Model name"),
            "vendor_id": get_field("Vendor ID"),
            "architecture": get_field("Architecture", platform.machine()),
            "sockets": sockets,
            "physical_cores": physical_cores,
            "logical_cores": int(get_field("CPU(s)", os.cpu_count() or 0) or os.cpu_count() or 0),
            "threads_per_core": threads_per_core,
            "cores_per_socket": cores_per_socket,
            "min_mhz": float(min_mhz_str) if min_mhz_str else None,
            "max_mhz": float(max_mhz_str) if max_mhz_str else None,
            "current_mhz": float(current_mhz_str) if current_mhz_str else None,
            "cache_l1d": get_field("L1d cache"),
            "cache_l1i": get_field("L1i cache"),
            "cache_l2": get_field("L2 cache"),
            "cache_l3": get_field("L3 cache"),
        }
    else:
        result = {
            "model_name": None,
            "vendor_id": None,
            "architecture": platform.machine(),
            "sockets": 1,
            "physical_cores": os.cpu_count() or 0,
            "logical_cores": os.cpu_count() or 0,
            "threads_per_core": 1,
            "cores_per_socket": os.cpu_count() or 0,
            "min_mhz": None,
            "max_mhz": None,
            "current_mhz": None,
            "cache_l1d": None,
            "cache_l1i": None,
            "cache_l2": None,
            "cache_l3": None,
        }

    result["flags"] = sorted(flags) if flags else []

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
    result["features"] = features

    return result


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


def collect_runtime_metadata(session=None) -> dict:
    import sys

    import numpy as np
    import onnxruntime as ort
    import transformers

    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "transformers_version": transformers.__version__,
        "available_providers": ort.get_available_providers(),
        "active_providers": session.get_providers() if session else None,
    }


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


def collect_container_metadata() -> dict:
    is_cont = is_container()

    container_runtime_hint = None
    cgroup_version = None
    cpu_quota = None
    cpu_cpuset = None
    memory_limit_bytes = None

    if is_cont:
        cgroup_text = read_text_or_none("/proc/1/cgroup")
        if cgroup_text:
            if "cgroup2" in cgroup_text or "/sys/fs/cgroup/unified" in cgroup_text:
                cgroup_version = "2"
            elif "cgroup" in cgroup_text:
                cgroup_version = "1"

        cpu_max = read_text_or_none("/sys/fs/cgroup/cpu.max")
        if cpu_max:
            parts = cpu_max.split()
            if len(parts) == 2:
                if parts[0] != "max":
                    cpu_quota = int(parts[0])

        cpu_cpuset = read_text_or_none("/sys/fs/cgroup/cpuset.cpus")

        memory_max = read_text_or_none("/sys/fs/cgroup/memory.max")
        if memory_max:
            if memory_max != "max":
                memory_limit_bytes = int(memory_max)

    return {
        "is_container": is_cont,
        "container_runtime_hint": container_runtime_hint,
        "cgroup_version": cgroup_version,
        "cpu_quota": cpu_quota,
        "cpu_cpuset": cpu_cpuset,
        "memory_limit_bytes": memory_limit_bytes,
    }


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


if __name__ == "__main__":
    import pprint
    pprint.pprint(collect_machine_metadata())