"""Run untrusted code in a subprocess with a wall-clock timeout.

The compile stage executes model-generated programs (CadQuery wrapper scripts,
the gmsh meshing child, the OpenSCAD / Node CLIs). None of it is trusted, and
some of it can hang indefinitely (CGAL booleans, gmsh's ``interruptible=False``
loop). The only robust bound is *isolate in a separate process and kill it on a
wall-clock timeout*. This module is the small shared helper for that pattern.

There is no real in-process sandbox here (the research code ran the CadQuery
wrapper with ``exec(..., {"cq": cq, "__builtins__": __builtins__})`` — full
builtins, no restriction); the isolation comes entirely from
"separate subprocess + ``subprocess.run(timeout=...)``".

Best-effort hardening on Linux: ``PR_SET_PDEATHSIG`` asks the kernel to deliver
``SIGKILL`` to the child the moment *this* process dies, so an OOM-killed or
``kill -9``'d parent does not leak hours-long zombies. It is a no-op on
non-Linux platforms (the parent-side timeout already covers the normal case).
"""

from __future__ import annotations

import ctypes
import signal
import subprocess
from typing import Optional, Sequence

# PR_SET_PDEATHSIG from <linux/prctl.h>.
_PR_SET_PDEATHSIG = 1
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None  # non-Linux (macOS / Windows) — parent-side timeout only.


def set_pdeathsig() -> None:
    """``preexec_fn``: have the kernel SIGKILL this child when its parent dies.

    Runs in the child after ``fork()`` and before ``exec()``; the setting is
    preserved across ``execve`` on Linux >= 2.6.23 for non-setuid binaries
    (python / node / openscad all qualify). No-op on non-Linux.
    """
    if _libc is None:
        return
    _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)


def _preexec_fn():
    """Return the platform-appropriate ``preexec_fn`` (None on non-Linux)."""
    return set_pdeathsig if _libc is not None else None


def run_subprocess(
    cmd: Sequence[str],
    *,
    timeout: float,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run *cmd* to completion under a wall-clock *timeout*, capturing output.

    Thin wrapper over ``subprocess.run`` that captures stdout/stderr as text and
    installs the PDEATHSIG ``preexec_fn`` on Linux. Raises
    ``subprocess.TimeoutExpired`` if the child outlives *timeout* — callers
    handle that to surface a clean "timed out" error.
    """
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        preexec_fn=_preexec_fn(),
    )
