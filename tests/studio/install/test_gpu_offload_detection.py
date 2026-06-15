# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Install-time GPU-offload detection: the classifier, the validate_server gate
that rejects a CPU-only intended-GPU bundle, and the CPU-prebuilt last resort
the resolver advances to (instead of a source build). No GPU, no network."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)

server_log_shows_gpu_offload = MOD.server_log_shows_gpu_offload
GpuOffloadFailure = MOD.GpuOffloadFailure
PrebuiltFallback = MOD.PrebuiltFallback
validate_server = MOD.validate_server
validate_prebuilt_attempts = MOD.validate_prebuilt_attempts
_GPU_OFFLOAD_REQUIRED_KINDS = MOD._GPU_OFFLOAD_REQUIRED_KINDS


# ── classifier ──────────────────────────────────────────────────────────

def test_offloaded_count_positive_true():
    assert server_log_shows_gpu_offload("offloaded 33/33 layers to GPU") is True


def test_offloaded_count_zero_false_even_with_gpu_buffer():
    log = "offloaded 0/33 layers to GPU\nCUDA0 model buffer size = 21000 MiB"
    assert server_log_shows_gpu_offload(log) is False


def test_offloaded_zero_then_nonzero_true():
    log = "offloaded 0/2 layers to GPU\noffloaded 33/33 layers to GPU"
    assert server_log_shows_gpu_offload(log) is True


@pytest.mark.parametrize(
    "marker",
    ["CUDA0", "ROCm0", "HIP0", "Metal", "Vulkan0", "OpenCL0", "SYCL0", "MUSA0", "CANN0"],
)
def test_gpu_model_buffer_markers_true(marker):
    assert server_log_shows_gpu_offload(f"{marker} model buffer size = 8000 MiB") is True


def test_host_pinned_buffer_excluded_false():
    log = "CUDA_Host model buffer size = 500 MiB\nCPU model buffer size = 21000 MiB"
    assert server_log_shows_gpu_offload(log) is False


def test_device_info_gpu_row_true():
    assert server_log_shows_gpu_offload("device_info:\n  - CUDA0 : 24564 MiB") is True


def test_device_info_cpu_only_false():
    assert server_log_shows_gpu_offload("device_info:\n  - CPU : 64000 MiB") is False


def test_system_info_cuda_before_device_info_does_not_count():
    log = "system_info: CUDA = 1\ndevice_info:\n  - CPU : 64000 MiB"
    assert server_log_shows_gpu_offload(log) is False


def test_no_signal_none():
    assert server_log_shows_gpu_offload("INFO starting server") is None


# ── validate_server offload gate ────────────────────────────────────────

class _FakeProc:
    def poll(self):
        return None  # alive, so the HTTP probe runs

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


class _FakeResponse:
    status = 200

    def read(self):
        return b'{"content": "x"}'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_validate_server(monkeypatch, log_text, install_kind):
    monkeypatch.setattr(MOD.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(MOD.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(MOD, "binary_env", lambda *a, **k: {})
    monkeypatch.setattr(MOD, "free_local_port", lambda: 49999)
    monkeypatch.setattr(MOD, "read_full_log", lambda *a, **k: log_text)
    monkeypatch.setattr(MOD, "read_log_excerpt", lambda *a, **k: log_text)
    return validate_server(
        Path("/fake/llama-server"),
        Path("/fake/probe.gguf"),
        types.SimpleNamespace(),
        Path("/fake/install"),
        install_kind=install_kind,
    )


def test_gpu_kind_cpu_only_log_rejected(monkeypatch):
    with pytest.raises(GpuOffloadFailure):
        _run_validate_server(monkeypatch, "offloaded 0/33 layers to GPU", "linux-cuda")


def test_gpu_kind_gpu_log_passes(monkeypatch):
    assert _run_validate_server(monkeypatch, "offloaded 33/33 layers to GPU", "linux-cuda") is None


def test_gpu_kind_no_signal_passes(monkeypatch):
    # A no-signal log must never reject during install (avoid false negatives).
    assert _run_validate_server(monkeypatch, "INFO starting server", "linux-cuda") is None


def test_cpu_kind_cpu_log_passes(monkeypatch):
    assert _run_validate_server(monkeypatch, "offloaded 0/33 layers to GPU", "linux-cpu") is None


def test_macos_metal_exempt_from_rejection(monkeypatch):
    # macos-arm64 is exercised but not rejected on a CPU-only load.
    assert "macos-arm64" not in _GPU_OFFLOAD_REQUIRED_KINDS
    assert _run_validate_server(monkeypatch, "offloaded 0/33 layers to GPU", "macos-arm64") is None


# ── advance-to-CPU-prebuilt (no source build) ───────────────────────────

def test_gpu_offload_failure_is_prebuilt_fallback():
    # Subclassing is what lets validate_prebuilt_attempts catch and advance.
    assert issubclass(GpuOffloadFailure, PrebuiltFallback)


def test_attempt_loop_advances_from_offload_failure_to_cpu(monkeypatch, tmp_path):
    cuda = types.SimpleNamespace(
        name="cuda.zip", install_kind="linux-cuda", runtime_line="cuda12", coverage_class="x"
    )
    cpu = types.SimpleNamespace(
        name="cpu.zip", install_kind="linux-cpu", runtime_line=None, coverage_class="x"
    )

    def fake_validate_choice(attempt, *a, **k):
        if attempt.install_kind == "linux-cuda":
            raise GpuOffloadFailure("ran on CPU")
        return None  # the CPU prebuilt validates fine

    monkeypatch.setattr(MOD, "validate_prebuilt_choice", fake_validate_choice)
    monkeypatch.setattr(MOD, "create_install_staging_dir", lambda d: tmp_path / "staging")
    monkeypatch.setattr(MOD, "remove_tree", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "prune_install_staging_root", lambda *a, **k: None)

    chosen, _staging, tried_fallback = validate_prebuilt_attempts(
        [cuda, cpu],
        types.SimpleNamespace(),
        tmp_path,
        tmp_path,
        Path("/fake/probe.gguf"),
        requested_tag="t",
        llama_tag="t",
        release_tag="t",
        approved_checksums=None,
    )
    assert chosen.install_kind == "linux-cpu"
    assert tried_fallback is True


# ── resolver keeps the CPU prebuilt as a labelled last resort ────────────

def test_linux_nvidia_plan_appends_cpu_last_resort(monkeypatch):
    cuda = types.SimpleNamespace(install_kind="linux-cuda", name="cuda.zip")
    cpu = types.SimpleNamespace(install_kind="linux-cpu", name="cpu.zip")
    host = types.SimpleNamespace(has_usable_nvidia=True, has_rocm=False)
    bundle = types.SimpleNamespace()

    monkeypatch.setattr(
        MOD, "detect_torch_cuda_runtime_preference",
        lambda h: types.SimpleNamespace(runtime_line="cuda12", selection_log=""),
    )
    monkeypatch.setattr(
        MOD, "linux_cuda_choice_from_release",
        lambda *a, **k: types.SimpleNamespace(attempts=[cuda]),
    )
    monkeypatch.setattr(
        MOD, "published_asset_choice_for_kind",
        lambda b, kind: cpu if kind == "linux-cpu" else None,
    )

    attempts = MOD._linux_published_attempts(host, bundle)
    assert [a.install_kind for a in attempts] == ["linux-cuda", "linux-cpu"]
