"""Compile-and-run tests for standalone C++ projects from generate_cpp_project.

Each test exports a tiny torch model, generates the standalone C++ project,
builds it against the vendored ggml, executes it on fixed vectors, and
compares the output against torch.
Single-output float32 models only, driven by tests/codegen/standalone/
(static driver.cpp + wrapper CMakeLists), so no toolchain flags live in
test code. Linux-only until the harness is validated elsewhere.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from ggmlc.codegen import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.serialization.gguf import save_to_gguf
from torch import nn

pytestmark = pytest.mark.skipif(
    sys.platform not in ("linux", "win32"),
    reason="standalone compile-and-run harness validated on Linux and Windows only",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GGML_SRC = REPO_ROOT / "third_party" / "ggml"
GGML_INCLUDE = GGML_SRC / "include"
RT_INCLUDE = REPO_ROOT / "runtime" / "include"
STANDALONE_DIR = Path(__file__).resolve().parent / "standalone"
DRIVER_SRC = STANDALONE_DIR / "driver.cpp"
WRAPPER_DIR = STANDALONE_DIR


@pytest.fixture(scope="session")
def ggml_standalone_libs():
    """Provide static ggml libs, building vendored ggml once and caching by source hash.

    The build lands in ~/.cache/ggmlc/standalone-ggml/<hash>/ and is reused
    across pytest runs until third_party/ggml, the flags, or the compiler
    change. Delete that directory to force a rebuild. ccache is used
    automatically when installed.
    """
    assert shutil.which("cmake") is not None, "cmake is required on PATH"
    # Toolchain identity for the cache key: explicit env or cmake defaults.
    # No compiler probing here — toolchain detection belongs to cmake.
    key = {
        "sources": _ggml_source_key(),
        "build_type": "Release",
        "shared": False,
        "tests": False,
        "examples": False,
        "cc": os.environ.get("CC", "default"),
        "cxx": os.environ.get("CXX", "default"),
    }
    cache_dir = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "ggmlc"
        / "standalone-ggml"
        / hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    )
    stamp = cache_dir / "stamp.json"
    if _stamp_matches(stamp, key):
        configs = sorted((cache_dir / "install").rglob("ggml-config.cmake"))
        if len(configs) == 1:
            print(f"\nreusing cached ggml build: {cache_dir}")
            return {"build_dir": str(configs[0].parent)}
    shutil.rmtree(cache_dir, ignore_errors=True)
    build_dir = cache_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(GGML_SRC),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_BUILD_TESTS=OFF",
        "-DGGML_BUILD_EXAMPLES=OFF",
    ]
    if shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    env = _cmake_env()
    _run_logged(configure, timeout=600, env=env)
    _run_logged(
        ["cmake", "--build", str(build_dir), "--parallel"],
        timeout=1200,
        env=env,
    )
    install_dir = cache_dir / "install"
    _run_logged(
        ["cmake", "--install", str(build_dir), "--prefix", str(install_dir)],
        timeout=600,
        env=env,
    )
    (config_dir,) = _require_one(
        sorted(install_dir.rglob("ggml-config.cmake")), "ggml-config.cmake", install_dir
    )
    stamp.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return {"build_dir": str(config_dir.parent)}


def _run_logged(cmd, **kwargs):
    """subprocess.run that appends captured output to the failure message."""
    check = kwargs.pop("check", True)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=check, **kwargs)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"command failed ({e.returncode}): {' '.join(e.cmd)}\n"
            f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}"
        ) from e


def _require_one(matches, what, where):
    assert len(matches) == 1, f"expected one {what} in {where}, found {len(matches)}"
    return matches


def _cmake_env():
    """Environment for cmake builds: MSVC detection on Windows, ccache launchers when available."""
    env = os.environ.copy()
    if sys.platform == "win32" and shutil.which("cl", path=env.get("PATH")) is None:
        vswhere = (
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Microsoft Visual Studio"
            / "Installer"
            / "vswhere.exe"
        )
        if vswhere.exists():
            try:
                out = subprocess.run(
                    [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                inst = out.stdout.strip()
                if inst:
                    vcvars = Path(inst) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if vcvars.exists():
                        p = subprocess.run(
                            f'"{vcvars}" && set',
                            shell=True,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        for line in p.stdout.splitlines():
                            if "=" in line:
                                k, v = line.split("=", 1)
                                env[k] = v
            except (subprocess.SubprocessError, OSError):
                pass
    if shutil.which("ccache", path=env.get("PATH")):
        env["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
    return env


def _ggml_source_key():
    """Content key for the vendored ggml tree: git tree hash, mtime fallback."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD:third_party/ggml"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return "git-tree:" + out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        latest = 0.0
        for suffix in ("src", "include"):
            root = GGML_SRC / suffix
            if root.is_dir():
                for p in root.rglob("*"):
                    try:
                        latest = max(latest, p.stat().st_mtime)
                    except OSError:
                        pass
        return f"mtime:{latest:.0f}"


def _stamp_matches(stamp, key):
    if not stamp.is_file():
        return False
    try:
        return json.loads(stamp.read_text(encoding="utf-8")) == key
    except (OSError, ValueError):
        return False


def _torch_shape_dims(tensor):
    return [int(d.evaluate({})) for d in tensor.shape.dims]


def _run_standalone(model, example_args, test_name, ggml_libs, tmp_path, atol=1e-5):
    """Export torch model, compile the standalone project, run it, compare vs torch."""
    model_name = "".join(part.capitalize() for part in test_name.split("_"))
    with torch.inference_mode():
        ref = model(*example_args)
    assert isinstance(ref, torch.Tensor), "single-tensor outputs only"
    ref_array = np.ascontiguousarray(ref.detach().cpu().numpy(), dtype=np.float32)

    exported = export_torch_model(model, example_args, model_name=test_name)
    ggml_graph = lower_to_ggml(exported.main_graph)
    proj_dir = tmp_path / "proj"
    generate_cpp_project(ggml_graph, proj_dir, model_name=model_name)
    gguf_path = save_to_gguf(ggml_graph, tmp_path / "weights.gguf")

    input_ids = list(exported.main_graph.inputs)
    out_path = tmp_path / "output.bin"
    spec_args = []
    for i, (inp_id, arg) in enumerate(zip(input_ids, example_args)):
        in_tensor = exported.main_graph.get_tensor(inp_id)
        dims = _torch_shape_dims(in_tensor)
        ne = list(reversed(dims)) + [1] * (4 - len(dims))
        bin_path = tmp_path / f"input_{i}.bin"
        np.ascontiguousarray(arg.detach().cpu().numpy(), dtype=np.float32).tofile(bin_path)
        spec_args += [in_tensor.name, *[str(d) for d in ne], str(bin_path)]

    test_build_dir = tmp_path / "test_build"
    configure = [
        "cmake",
        "-S",
        str(WRAPPER_DIR),
        "-B",
        str(test_build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGGML_BUILD_DIR={ggml_libs['build_dir']}",
        f"-DMODEL_PROJ_DIR={proj_dir}",
        f"-DRT_INCLUDE_DIR={RT_INCLUDE}",
        f"-DTEST_DRIVER_SRC={DRIVER_SRC}",
        f"-DTEST_MODEL={model_name}",
    ]
    if shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    env = _cmake_env()
    _run_logged(configure, timeout=300, env=env)
    _run_logged(
        ["cmake", "--build", str(test_build_dir), "--parallel"],
        timeout=600,
        env=env,
    )
    exe_path = test_build_dir / ("standalone_test.exe" if os.name == "nt" else "standalone_test")
    if not exe_path.exists() and os.name == "nt":
        rel_exe = test_build_dir / "Release" / "standalone_test.exe"
        if rel_exe.exists():
            exe_path = rel_exe
    run = _run_logged(
        [str(exe_path), str(gguf_path), str(out_path), str(len(input_ids)), *spec_args],
        timeout=120,
    )
    print(run.stdout)

    got = np.fromfile(out_path, dtype=np.float32)
    assert got.size == ref_array.size, f"element mismatch: {got.size} vs {ref_array.size}"
    print(f"\nref[0:8]: {ref_array.ravel()[:8]}")
    print(f"got[0:8]: {got.ravel()[:8]}")
    print(f"max abs diff: {np.max(np.abs(got - ref_array.ravel()))}")
    np.testing.assert_allclose(got, ref_array.ravel(), atol=atol)


def test_standalone_linear_bias(ggml_standalone_libs, tmp_path):
    """Biased LINEAR through generated standalone code matches torch."""
    torch.manual_seed(0)
    model = nn.Linear(16, 16).eval()  # default init: random weights, nonzero bias
    assert torch.count_nonzero(model.bias) > 0
    x = torch.randn(1, 16)
    _run_standalone(model, (x,), "tiny_linear_bias", ggml_standalone_libs, tmp_path)


def test_standalone_matmul_nobias(ggml_standalone_libs, tmp_path):
    """Bias-free 2-input MATMUL through generated standalone code matches torch."""
    torch.manual_seed(0)

    class MatMul(nn.Module):
        def forward(self, a, b):
            return a @ b

    a = torch.randn(2, 16)
    b = torch.randn(16, 8)
    _run_standalone(MatMul().eval(), (a, b), "tiny_matmul_nobias", ggml_standalone_libs, tmp_path)


def test_standalone_expand_broadcast(ggml_standalone_libs, tmp_path):
    """EXPAND (single-input REPEAT) through generated standalone code matches torch."""
    torch.manual_seed(0)

    class Expand(nn.Module):
        def forward(self, x):
            return x.expand(4, 16)

    x = torch.randn(1, 16)
    _run_standalone(Expand().eval(), (x,), "tiny_expand_broadcast", ggml_standalone_libs, tmp_path)


def test_standalone_reshape_noncontiguous(ggml_standalone_libs, tmp_path):
    """RESHAPE of a non-contiguous transpose matches torch (needs the cont guard)."""
    torch.manual_seed(0)

    class ReshapeTransposed(nn.Module):
        def forward(self, x):
            return x.transpose(0, 1).reshape(16)

    x = torch.randn(1, 16)
    _run_standalone(
        ReshapeTransposed().eval(), (x,), "tiny_reshape_transposed", ggml_standalone_libs, tmp_path
    )


def test_standalone_slice_offset(ggml_standalone_libs, tmp_path):
    """Nonzero-start SLICE (VIEW byte offset) through generated standalone code matches torch."""
    torch.manual_seed(0)

    class Slice(nn.Module):
        def forward(self, x):
            return x[:, 4:12]

    x = torch.randn(8, 16)
    _run_standalone(Slice().eval(), (x,), "tiny_slice_dim1", ggml_standalone_libs, tmp_path)


def test_standalone_permute_4d(ggml_standalone_libs, tmp_path):
    """4D PERMUTE through generated standalone code matches torch."""
    torch.manual_seed(0)

    class Permute(nn.Module):
        def forward(self, x):
            return x.permute(0, 2, 1, 3)

    x = torch.randn(2, 4, 8, 16)
    _run_standalone(Permute().eval(), (x,), "tiny_permute_4d", ggml_standalone_libs, tmp_path)


def test_standalone_sum_rows_dim0(ggml_standalone_libs, tmp_path):
    """SUM over the last dim (GGML_OP_SUM_ROWS) through generated code matches torch."""
    torch.manual_seed(0)

    class SumLast(nn.Module):
        def forward(self, x):
            return x.sum(dim=-1)

    x = torch.randn(4, 16)
    _run_standalone(SumLast().eval(), (x,), "tiny_sum_rows_dim0", ggml_standalone_libs, tmp_path)


def test_standalone_sqrt(ggml_standalone_libs, tmp_path):
    """SQRT through generated standalone code matches torch."""
    torch.manual_seed(0)

    class Sqrt(nn.Module):
        def forward(self, x):
            return torch.sqrt(x.abs() + 0.5)

    x = torch.randn(4, 16)
    _run_standalone(Sqrt().eval(), (x,), "tiny_sqrt", ggml_standalone_libs, tmp_path)


def test_standalone_l2_normalize(ggml_standalone_libs, tmp_path):
    """L2-normalize (x / norm(keepdim)) through generated code matches torch."""
    torch.manual_seed(0)

    class L2Normalize(nn.Module):
        def forward(self, img_embeds):
            return img_embeds / img_embeds.norm(dim=-1, keepdim=True)

    img_embeds = torch.randn(2, 8, 16)
    _run_standalone(
        L2Normalize().eval(), (img_embeds,), "tiny_l2_normalize", ggml_standalone_libs, tmp_path
    )
