"""Per-format compile dispatch: code -> STEP/STL (+ per-part STLs).

``compile_code(code, format_slug, output_dir) -> CompileResult`` is the single
entry point, imported by ``p3dbench.formats.base.Format.compile``. Format slugs:
``cadquery`` | ``openscad`` | ``threejs`` | ``minimal-json``.

Validity is purely ``bool(stl)``: an STL was produced. STEP-only outputs are
not valid (geometry metrics need a mesh). A non-empty ``errors`` list does not
by itself invalidate, and per-part failures never invalidate the case.

Assembly-3D per-part contract (all formats converge on it):
  ``output_dir/parts/part_NN.stl`` + ``output_dir/parts_meta.json``
  (a list of ``{index, name, semantic, stl|None, <format key>, error?}``).
``parts_dir`` / ``parts_meta`` on the result point at those.

Per-format part contracts parsed here:
  * cadquery     -> a ``parts`` list of ``{name, semantic, model}`` in the code.
  * openscad     -> a ``// parts_meta: [...]`` header + one ``module`` per entry.
  * threejs      -> a ``// parts_meta: [...]`` header + ``THREE.Group`` walk.
  * minimal-json -> a top-level ``parts_meta`` object grouping feature keys.

Three.js runtime: an optional (uncommitted) ``three/`` data directory under this
package. The files ``three/build/three.module.js`` and
``three/examples/jsm/exporters/STLExporter.js`` are required for STL export; if
they (or ``node``) are absent the threejs path returns a clean invalid result.
See ``docs/FORMATS.md`` for the expected layout.

Stripped from the research code: OpenSCAD concurrency cap / filelock, the
DeepCAD legacy-JSON path, the OpenSCAD preview PNG (+ xvfb), the Three.js
``viewer.html``, the ``code_path`` resume param, and ``allow_fallbacks``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..formats.base import CompileResult
from .sandbox import run_subprocess
from .step_mesh import export_step_to_stl

logger = logging.getLogger(__name__)

# STEP -> mesh tessellation size for gmsh fallback (fraction of bbox, relative).
_STEP_MESH_SIZE = 0.02
_STEP_MESH_SIZE_MODE = "relative"

# Per-call subprocess wall-clock budgets (correctness hang guards).
_CADQUERY_TIMEOUT_S = 180
_OPENSCAD_UNION_TIMEOUT_S = 480
_OPENSCAD_PART_TIMEOUT_S = 300
_THREEJS_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# CadQuery
# ---------------------------------------------------------------------------

def _load_export_error_details(diagnostic_path: Path) -> list[dict[str, Any]]:
    """Load structured exporter diagnostics emitted by the wrapper script."""
    try:
        data = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    finally:
        diagnostic_path.unlink(missing_ok=True)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _extract_export_error_messages(stderr_text: str) -> list[str]:
    """Extract concise ERROR lines from exporter stderr."""
    messages = []
    for line in stderr_text.splitlines():
        line = line.strip()
        if line.startswith("ERROR"):
            messages.append(line)

    if messages:
        return messages

    fallback = stderr_text.strip()
    return [fallback] if fallback else []


# The wrapper runs in a subprocess; keep it self-contained. There is no
# in-process sandbox — isolation is "separate subprocess + timeout".
_CADQUERY_WRAPPER_TEMPLATE = '''\
import sys
import json
import traceback
from functools import reduce

try:
    import cadquery as cq
except ImportError:
    print("ERROR: cadquery not installed. Run: pip install cadquery", file=sys.stderr)
    sys.exit(1)

CODE_FILE = {code_file!r}
DIAGNOSTIC_FILE = {diagnostic_path!r}
UNION_STEP = {step_path!r}
PARTS_DIR = {parts_dir!r}
PARTS_MANIFEST = {parts_manifest_path!r}


def _persist_error_details(details):
    try:
        with open(DIAGNOSTIC_FILE, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _error_detail(stage, exc, tb, source_code):
    frames = traceback.extract_tb(tb) if tb else []
    line_number = None
    for frame in reversed(frames):
        if frame.filename == CODE_FILE:
            line_number = frame.lineno
            break
    line_text = None
    if line_number is not None:
        source_lines = source_code.splitlines()
        if 1 <= line_number <= len(source_lines):
            line_text = source_lines[line_number - 1]
    return {{
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, tb)),
        "line_number": line_number,
        "line_text": line_text,
    }}


def _looks_like_cq(obj):
    if obj is None:
        return False
    if isinstance(obj, cq.Workplane):
        return True
    return hasattr(obj, "val") or hasattr(obj, "Objects")


error_details = []

exec_globals = {{"cq": cq, "__builtins__": __builtins__}}
with open(CODE_FILE, "r") as f:
    src = f.read()

try:
    compiled = compile(src, CODE_FILE, "exec")
    exec(compiled, exec_globals)
except Exception as e:
    error_details.append(_error_detail("execute", e, e.__traceback__, src))
    _persist_error_details(error_details)
    print("ERROR executing CadQuery code: {{}}".format(e), file=sys.stderr)
    sys.exit(1)


def _validate_parts(raw):
    if not isinstance(raw, (list, tuple)):
        return None, "`parts` must be a list"
    if not raw:
        return None, "`parts` is empty"
    out = []
    seen_names = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None, "parts[{{}}] is not a dict".format(i)
        name = entry.get("name")
        semantic = entry.get("semantic")
        model = entry.get("model")
        if not isinstance(name, str) or not name.strip():
            return None, "parts[{{}}].name missing or not a string".format(i)
        if not isinstance(semantic, str) or not semantic.strip():
            return None, "parts[{{}}].semantic missing or not a string".format(i)
        if not _looks_like_cq(model):
            return None, "parts[{{}}].model is not a CadQuery Workplane".format(i)
        safe_name = name.strip()
        if safe_name in seen_names:
            return None, "parts[{{}}].name '{{}}' is not unique".format(i, safe_name)
        seen_names.add(safe_name)
        out.append({{"name": safe_name, "semantic": semantic.strip(), "model": model}})
    return out, None


raw_parts = exec_globals.get("parts")
parts_info = None
if raw_parts is not None:
    validated, err = _validate_parts(raw_parts)
    if err:
        error_details.append({{"stage": "validate_parts", "error_type": "ValueError",
                              "message": err, "traceback": "", "line_number": None,
                              "line_text": None}})
        _persist_error_details(error_details)
        print("ERROR: {{}}".format(err), file=sys.stderr)
        sys.exit(1)
    parts_info = validated

result_obj = exec_globals.get("result")
if result_obj is None and not parts_info:
    for name in ["model", "part", "shape", "body"]:
        if name in exec_globals and _looks_like_cq(exec_globals[name]):
            result_obj = exec_globals[name]
            break

if result_obj is None and not parts_info:
    err = "No `result` variable or `parts` list found in generated code"
    error_details.append({{"stage": "locate_result", "error_type": "ValueError",
                          "message": err, "traceback": "", "line_number": None,
                          "line_text": None}})
    _persist_error_details(error_details)
    print("ERROR: {{}}".format(err), file=sys.stderr)
    sys.exit(1)


per_part_manifest = []
if parts_info:
    import os
    os.makedirs(PARTS_DIR, exist_ok=True)
    for i, entry in enumerate(parts_info):
        part_step = os.path.join(PARTS_DIR, "part_{{:02d}}.step".format(i))
        try:
            cq.exporters.export(entry["model"], part_step)
        except Exception as e:
            error_details.append(_error_detail("export_part_step_{{}}".format(i),
                                               e, e.__traceback__, src))
            print("ERROR exporting part {{}} STEP: {{}}".format(i, e), file=sys.stderr)
            continue
        per_part_manifest.append({{
            "index": i,
            "name": entry["name"],
            "semantic": entry["semantic"],
            "step": part_step,
        }})

    with open(PARTS_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(per_part_manifest, f, ensure_ascii=False, indent=2)

    if result_obj is None:
        try:
            result_obj = reduce(
                lambda acc, m: acc.union(m) if acc is not None else m,
                (entry["model"] for entry in parts_info),
                None,
            )
        except Exception as e:
            error_details.append(_error_detail("compute_union", e, e.__traceback__, src))
            print("ERROR computing union from parts: {{}}".format(e), file=sys.stderr)
            result_obj = None

if result_obj is not None:
    try:
        cq.exporters.export(result_obj, UNION_STEP)
        print("Exported STEP: {{}}".format(UNION_STEP))
    except Exception as e:
        error_details.append(_error_detail("export_step", e, e.__traceback__, src))
        print("ERROR exporting STEP: {{}}".format(e), file=sys.stderr)

if error_details:
    _persist_error_details(error_details)
'''


def export_cadquery(code: str, output_dir: Path) -> Dict[str, Any]:
    """Execute CadQuery code in an isolated subprocess and export STL + STEP.

    Two auto-detected contracts:
      (a) single-part: code assigns ``result`` (or ``model``/``part``/...).
      (b) assembly: code assigns a ``parts`` list of ``{name, semantic, model}``;
          per-part STEP+STL go to ``output_dir/parts/`` with ``parts_meta.json``,
          and the union (from ``result`` or computed) is exported as the model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / "model.stl"
    step_path = output_dir / "model.step"
    parts_dir = output_dir / "parts"
    parts_manifest_path = output_dir / "parts_meta.json"
    result: Dict[str, Any] = {
        "stl": None, "step": None, "errors": [], "error_details": [],
        "parts": None,
    }

    code_file = output_dir / "_tmp_cadquery_code.py"
    code_file.write_text(code, encoding="utf-8")
    diagnostic_path = output_dir / "_tmp_export_diagnostics.json"
    diagnostic_path.unlink(missing_ok=True)

    wrapper_code = _CADQUERY_WRAPPER_TEMPLATE.format(
        code_file=str(code_file),
        diagnostic_path=str(diagnostic_path),
        step_path=str(step_path),
        parts_dir=str(parts_dir),
        parts_manifest_path=str(parts_manifest_path),
    )

    script_file = output_dir / "_tmp_export_wrapper.py"
    script_file.write_text(wrapper_code, encoding="utf-8")

    try:
        proc = run_subprocess(
            [sys.executable, str(script_file)],
            timeout=_CADQUERY_TIMEOUT_S,
        )
        result["error_details"] = _load_export_error_details(diagnostic_path)
        if proc.returncode != 0:
            error_messages = _extract_export_error_messages(proc.stderr)
            if not error_messages:
                error_messages = ["CadQuery export failed"]
            logger.error(f"CadQuery export failed: {error_messages[0]}")
            result["errors"].extend(error_messages)
        else:
            if step_path.exists():
                result["step"] = str(step_path)
                stl_error, stl_stats = export_step_to_stl(
                    step_path,
                    stl_path,
                    gmsh_size=_STEP_MESH_SIZE,
                    gmsh_size_mode=_STEP_MESH_SIZE_MODE,
                )
                if stl_error:
                    result["errors"].append(stl_error)
                elif stl_path.exists():
                    result["stl"] = str(stl_path)
                    logger.info(f"Exported STL: {stl_path} ({stl_stats})")
            if proc.stderr.strip():
                result["errors"].extend(_extract_export_error_messages(proc.stderr))
            if proc.stdout.strip():
                logger.info(proc.stdout.strip())

            # Multipart: convert per-part STEP -> STL, mirroring the union path.
            if parts_manifest_path.exists():
                try:
                    per_part_manifest = json.loads(
                        parts_manifest_path.read_text(encoding="utf-8")
                    )
                except Exception as e:
                    result["errors"].append(f"parts_meta.json unreadable: {e}")
                    per_part_manifest = []
                for entry in per_part_manifest:
                    part_step = Path(entry.get("step", ""))
                    if not part_step.exists():
                        continue
                    part_stl = part_step.with_suffix(".stl")
                    err, _ = export_step_to_stl(
                        part_step,
                        part_stl,
                        gmsh_size=_STEP_MESH_SIZE,
                        gmsh_size_mode=_STEP_MESH_SIZE_MODE,
                    )
                    if err:
                        result["errors"].append(
                            f"per-part STL failed for {entry.get('name')}: {err}"
                        )
                    elif part_stl.exists():
                        entry["stl"] = str(part_stl)
                if per_part_manifest:
                    parts_manifest_path.write_text(
                        json.dumps(per_part_manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    result["parts"] = per_part_manifest
    except subprocess.TimeoutExpired:
        result["errors"].append(f"CadQuery export timed out ({_CADQUERY_TIMEOUT_S}s)")
    finally:
        script_file.unlink(missing_ok=True)
        code_file.unlink(missing_ok=True)
        diagnostic_path.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# OpenSCAD
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _find_openscad() -> Optional[str]:
    """Locate an openscad binary.

    Resolution order: ``~/bin/openscad-nightly.AppImage`` (2024+ snapshots ship
    the Manifold backend, 10-100x faster than CGAL on LLM CSG) -> ``which`` ->
    hard-coded distro paths.
    """
    import os

    home_appimage = os.path.expanduser("~/bin/openscad-nightly.AppImage")
    if os.path.isfile(home_appimage) and os.access(home_appimage, os.X_OK):
        return home_appimage

    found = shutil.which("openscad")
    if found:
        return found
    for path in ["/usr/bin/openscad", "/usr/local/bin/openscad", "/snap/bin/openscad"]:
        if os.path.isfile(path):
            return path
    return None


@lru_cache(maxsize=4)
def _openscad_supports_manifold(openscad_cmd: str) -> bool:
    """True iff ``openscad_cmd --help`` advertises ``--backend`` (2023+ snapshots).

    Cached because the help probe is a ~200ms subprocess. The Manifold boolean
    backend is 10-100x faster than CGAL on LLM-generated geometry.
    """
    try:
        proc = subprocess.run(
            [openscad_cmd, "--help"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "--backend" in (proc.stdout + proc.stderr)


_OPENSCAD_PARTS_META_RE = re.compile(
    r"//\s*parts_meta\s*:\s*(\[.*?\])",
    re.DOTALL | re.IGNORECASE,
)

# Top-level `$fn`/`$fa`/`$fs` assignments. `use <...>` does NOT import these,
# so per-part wrappers would otherwise render at default resolution and
# disagree with the union STL's tessellation.
_OPENSCAD_SPECIAL_VAR_RE = re.compile(
    r"^\s*\$(?:fn|fa|fs)\s*=\s*[^;]+;",
    re.MULTILINE,
)


def _extract_openscad_special_vars(code: str) -> str:
    """Return the concatenation of top-level ``$fn``/``$fa``/``$fs`` assignments.

    Tracks ``{`` / ``}`` depth so matches inside modules/functions are ignored.
    String literals and ``//`` / ``/* */`` comments are masked to avoid false
    matches.
    """
    def _mask(text: str) -> str:
        out, i, n = [], 0, len(text)
        in_string = None
        while i < n:
            ch = text[i]
            if in_string:
                out.append(ch if ch != "\n" else "\n")
                if ch == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                if ch == in_string:
                    in_string = None
                i += 1
                continue
            if ch == '"':
                in_string = '"'
                out.append(" ")
                i += 1
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                while i < n and text[i] != "\n":
                    out.append(" ")
                    i += 1
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                out.extend("  ")
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    out.append("\n" if text[i] == "\n" else " ")
                    i += 1
                i += 2
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    masked = _mask(code)
    depth = 0
    kept = []
    idx = 0
    while idx < len(masked):
        ch = masked[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if depth == 0:
            m = _OPENSCAD_SPECIAL_VAR_RE.match(masked, idx)
            if m:
                kept.append(m.group(0).strip())
                idx = m.end()
                continue
        idx += 1
    return "\n".join(kept)


def _extract_openscad_parts_meta(code: str) -> Tuple[Optional[list], Optional[str]]:
    """Parse the ``// parts_meta: [...]`` JSON header comment block."""
    m = _OPENSCAD_PARTS_META_RE.search(code)
    if not m:
        return None, None

    raw = m.group(1)
    cleaned_lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            stripped = stripped[2:].lstrip()
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)
    try:
        entries = json.loads(cleaned)
    except Exception as exc:
        return None, f"parts_meta JSON parse error: {exc}"

    if not isinstance(entries, list) or not entries:
        return None, "parts_meta must be a non-empty list"

    out = []
    seen_names = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return None, f"parts_meta[{i}] is not an object"
        name = entry.get("name")
        semantic = entry.get("semantic")
        module = entry.get("module")
        if not isinstance(name, str) or not name.strip():
            return None, f"parts_meta[{i}].name missing or invalid"
        if not isinstance(semantic, str) or not semantic.strip():
            return None, f"parts_meta[{i}].semantic missing or invalid"
        if not isinstance(module, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", module):
            return None, f"parts_meta[{i}].module '{module}' is not a valid OpenSCAD identifier"
        if name in seen_names:
            return None, f"parts_meta[{i}].name '{name}' is not unique"
        seen_names.add(name)
        out.append({"name": name.strip(), "semantic": semantic.strip(), "module": module})
    return out, None


def _stl_looks_complete(stl_path: Path) -> bool:
    """Check that *stl_path* is a structurally complete STL (not a partial write).

    Binary STL: 80-byte header + uint32 triangle count + 50 bytes per triangle;
    accept iff ``file_size == 84 + 50 * triangle_count``. ASCII STL: must end
    with an ``endsolid`` token within the last 256 bytes.
    """
    try:
        size = stl_path.stat().st_size
        if size < 84:
            return False
        with stl_path.open("rb") as f:
            head = f.read(5)
            if head == b"solid":
                f.seek(max(0, size - 256))
                if b"endsolid" in f.read():
                    return True
            f.seek(80)
            count_bytes = f.read(4)
        if len(count_bytes) != 4:
            return False
        ntri = int.from_bytes(count_bytes, byteorder="little", signed=False)
        return size == 84 + ntri * 50
    except Exception:
        return False


def _run_openscad_stl(openscad_cmd: str, scad_path: Path, stl_path: Path,
                      timeout_s: int = _OPENSCAD_UNION_TIMEOUT_S) -> Optional[str]:
    """Invoke openscad to produce an STL. Returns an error string or None.

    ``--backend Manifold`` is prepended when supported (much faster). ``stl_path``
    is unlinked up-front so any post-timeout file is unambiguously from this
    invocation; on timeout we accept a structurally complete STL.
    """
    cmd = [openscad_cmd]
    if _openscad_supports_manifold(openscad_cmd):
        cmd += ["--backend", "Manifold"]
    cmd += ["-o", str(stl_path), str(scad_path)]

    stl_path.unlink(missing_ok=True)
    try:
        proc = run_subprocess(cmd, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if stl_path.exists() and _stl_looks_complete(stl_path):
            logger.warning(
                f"OpenSCAD exceeded {timeout_s}s but STL appears complete "
                f"({stl_path}); accepting post-timeout write."
            )
            return None
        return f"OpenSCAD STL export timed out ({timeout_s}s)"
    if proc.returncode != 0 or not stl_path.exists():
        return f"STL export failed: {proc.stderr.strip() or 'Unknown error'}"
    return None


def export_openscad(code: str, output_dir: Path) -> Dict[str, Any]:
    """Save OpenSCAD code and export STL via the OpenSCAD CLI.

    When the source carries a ``// parts_meta: [...]`` header, also export
    per-part STLs (one openscad invocation per module) into
    ``output_dir/parts/`` and write ``parts_meta.json``. No STEP (mesh-only).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scad_path = output_dir / "model.scad"
    stl_path = output_dir / "model.stl"
    result: Dict[str, Any] = {"stl": None, "errors": [], "parts": None}

    try:
        scad_path.write_text(code, encoding="utf-8")
    except Exception as e:
        result["errors"].append(f"Failed to write .scad file: {e}")
        return result

    openscad_cmd = _find_openscad()
    if not openscad_cmd:
        result["errors"].append(
            "OpenSCAD not found. Install with: sudo apt install openscad"
        )
        return result

    # Union STL.
    stl_err = _run_openscad_stl(openscad_cmd, scad_path, stl_path)
    if stl_err:
        result["errors"].append(stl_err)
    else:
        result["stl"] = str(stl_path)
        logger.info(f"Exported STL: {stl_path}")

    # Per-part STLs (if parts_meta header present).
    parts_meta, parts_err = _extract_openscad_parts_meta(code)
    if parts_err:
        result["errors"].append(parts_err)
    elif parts_meta:
        parts_dir = output_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        # `use <...>` imports modules but NOT top-level $fn/$fa/$fs, so copy
        # them into each wrapper to keep per-part tessellation consistent.
        special_vars = _extract_openscad_special_vars(code)
        special_prefix = f"{special_vars}\n" if special_vars else ""
        manifest: list = []
        for i, entry in enumerate(parts_meta):
            wrapper_path = parts_dir / f"part_{i:02d}.scad"
            part_stl = parts_dir / f"part_{i:02d}.stl"
            wrapper_path.write_text(
                f'{special_prefix}use <{scad_path.resolve()}>\n{entry["module"]}();\n',
                encoding="utf-8",
            )
            err = _run_openscad_stl(
                openscad_cmd, wrapper_path, part_stl, timeout_s=_OPENSCAD_PART_TIMEOUT_S
            )
            if err:
                result["errors"].append(f"per-part STL failed for {entry['name']}: {err}")
                continue
            manifest.append({
                "index": i,
                "name": entry["name"],
                "semantic": entry["semantic"],
                "module": entry["module"],
                "stl": str(part_stl),
            })
        if manifest:
            (output_dir / "parts_meta.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result["parts"] = manifest

    return result


# ---------------------------------------------------------------------------
# Three.js
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _find_node() -> Optional[str]:
    """Find the node executable."""
    return shutil.which("node")


def _get_threejs_package_root() -> Tuple[Optional[Path], Optional[str]]:
    """Resolve the vendored Three.js runtime under ``p3dbench/compile/three/``.

    Required for STL export:
      ``build/three.module.js`` and ``examples/jsm/exporters/STLExporter.js``.
    Returns ``(pkg_root, None)`` or ``(None, error)``. The runtime is not
    vendored in this repo by default — see ``docs/FORMATS.md``.
    """
    pkg_root = Path(__file__).parent / "three"
    required_files = [
        pkg_root / "build" / "three.module.js",
        pkg_root / "examples" / "jsm" / "exporters" / "STLExporter.js",
    ]

    if not pkg_root.exists() or any(not p.exists() for p in required_files):
        return None, "threejs runtime not vendored (see docs/FORMATS.md)"

    return pkg_root, None


_THREEJS_PARTS_META_RE = re.compile(
    r"//\s*parts_meta\s*:\s*(\[.*?\])",
    re.DOTALL | re.IGNORECASE,
)


def _extract_threejs_parts_meta(code: str) -> Tuple[Optional[list], Optional[str]]:
    """Parse the ``// parts_meta: [...]`` header (with ``group_var`` per entry)."""
    m = _THREEJS_PARTS_META_RE.search(code)
    if not m:
        return None, None

    raw = m.group(1)
    cleaned_lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            stripped = stripped[2:].lstrip()
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)
    try:
        entries = json.loads(cleaned)
    except Exception as exc:
        return None, f"threejs parts_meta JSON parse error: {exc}"

    if not isinstance(entries, list) or not entries:
        return None, "threejs parts_meta must be a non-empty list"

    out = []
    seen_names = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return None, f"threejs parts_meta[{i}] is not an object"
        name = entry.get("name")
        semantic = entry.get("semantic")
        group_var = entry.get("group_var")
        if not isinstance(name, str) or not name.strip():
            return None, f"threejs parts_meta[{i}].name missing or invalid"
        if not isinstance(semantic, str) or not semantic.strip():
            return None, f"threejs parts_meta[{i}].semantic missing or invalid"
        if not isinstance(group_var, str) or not re.match(
            r"^[A-Za-z_$][A-Za-z0-9_$]*$", group_var
        ):
            return None, (
                f"threejs parts_meta[{i}].group_var '{group_var}' is not a valid JS identifier"
            )
        if name in seen_names:
            return None, f"threejs parts_meta[{i}].name '{name}' is not unique"
        seen_names.add(name)
        out.append({
            "name": name.strip(),
            "semantic": semantic.strip(),
            "group_var": group_var,
        })
    return out, None


def _threejs_to_stl(
    code: str,
    stl_path: Path,
    pkg_root: Path,
    node_cmd: str,
    parts_meta: Optional[list] = None,
    parts_dir: Optional[Path] = None,
) -> Tuple[Optional[str], Optional[list]]:
    """Run LLM Three.js code in Node.js and export the scene to STL.

    When ``parts_meta`` is supplied, the wrapper also walks ``scene.children``,
    matches each ``THREE.Group`` (by ``userData.part_name`` then ``group.name``),
    clones it into a fresh scene, and exports ``parts_dir/part_NN.stl``.
    Returns ``(error|None, per_part_manifest|None)``.
    """
    # node runs with cwd=pkg_root.parent (so the vendored `three` package self-
    # resolves), so every path handed to node must be absolute — a relative
    # work dir would otherwise resolve against the package dir, not the cwd.
    stl_path = Path(stl_path).resolve()
    if parts_dir is not None:
        parts_dir = Path(parts_dir).resolve()
    code_file = stl_path.parent / "_tmp_threejs_code.mjs"
    wrapper_file = stl_path.parent / "_tmp_threejs_export.mjs"
    parts_report_file = stl_path.parent / "_tmp_threejs_parts_report.json"

    code_file.write_text(code, encoding="utf-8")

    three_module = json.dumps(str((pkg_root / "build" / "three.module.js").resolve()))
    stl_exporter_module = json.dumps(
        str((pkg_root / "examples" / "jsm" / "exporters" / "STLExporter.js").resolve())
    )
    code_file_js = json.dumps(str(code_file))
    stl_path_js = json.dumps(str(stl_path))

    parts_block = ""
    parts_report_js = json.dumps(str(parts_report_file))
    if parts_meta and parts_dir is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)
        parts_meta_js = json.dumps(parts_meta)
        part_stl_paths = [
            str(parts_dir / f"part_{i:02d}.stl") for i in range(len(parts_meta))
        ]
        part_stl_paths_js = json.dumps(part_stl_paths)
        parts_block = f"""
// Per-part STL export. Walk scene.children, match each top-level THREE.Group
// whose userData.part_name === entry.name (or group.name fallback). Clone the
// match into a fresh scene (carrying its baked world transform), export STL.
const partsMeta = {parts_meta_js};
const partStlPaths = {part_stl_paths_js};
const partsReport = [];
const sceneGroups = scene.children.filter((c) => c && c.isGroup === true);
for (let i = 0; i < partsMeta.length; i++) {{
    const entry = partsMeta[i];
    const targetName = entry.name;
    const targetVar = entry.group_var;
    let match = null;
    for (const g of sceneGroups) {{
        const ud = g.userData || {{}};
        if (ud.part_name === targetName) {{ match = g; break; }}
    }}
    if (!match) {{
        for (const g of sceneGroups) {{
            if (g.name === targetVar || g.name === targetName) {{ match = g; break; }}
        }}
    }}
    if (!match) {{
        partsReport.push({{
            index: i, name: targetName, semantic: entry.semantic,
            group_var: targetVar, stl: null,
            error: 'group not found in scene.children (matched on userData.part_name and group.name)'
        }});
        continue;
    }}
    const subScene = new THREE.Scene();
    const groupClone = match.clone(true);
    subScene.add(groupClone);
    subScene.updateMatrixWorld(true);
    const partExporter = new STLExporter();
    try {{
        const partStl = partExporter.parse(subScene, {{ binary: false }});
        let partMeshCount = 0;
        subScene.traverse((o) => {{ if (o.isMesh) partMeshCount++; }});
        if (partMeshCount === 0) {{
            partsReport.push({{
                index: i, name: targetName, semantic: entry.semantic,
                group_var: targetVar, stl: null,
                error: 'group has no meshes after traversal',
            }});
            continue;
        }}
        fs.writeFileSync(partStlPaths[i], partStl);
        partsReport.push({{
            index: i, name: targetName, semantic: entry.semantic,
            group_var: targetVar, stl: partStlPaths[i],
        }});
    }} catch (e) {{
        partsReport.push({{
            index: i, name: targetName, semantic: entry.semantic,
            group_var: targetVar, stl: null,
            error: 'STLExporter failed: ' + (e && e.message ? e.message : String(e)),
        }});
    }}
}}
fs.writeFileSync({parts_report_js}, JSON.stringify(partsReport));
console.log('Per-part export attempted for', partsMeta.length, 'group(s)');
"""

    wrapper = f"""\
import * as THREE from {three_module};
import {{ STLExporter }} from {stl_exporter_module};
import fs from 'fs';
import {{ readFileSync }} from 'fs';

// Headless runtime mirroring the browser viewer contract closely enough for export.
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 10000);
camera.position.set(30, 30, 30);
camera.lookAt(0, 0, 0);
const renderer = {{
    domElement: null,
    render() {{}},
    setSize() {{}},
    setPixelRatio() {{}},
    setClearColor() {{}},
    setAnimationLoop() {{}},
    shadowMap: {{ enabled: false }},
}};
const controls = {{
    target: new THREE.Vector3(0, 0, 0),
    update() {{}},
}};

globalThis.THREE = THREE;
globalThis.scene = scene;
globalThis.camera = camera;
globalThis.renderer = renderer;
globalThis.controls = controls;

const userCode = readFileSync({code_file_js}, 'utf-8');
try {{
    const fn = new Function('THREE', 'scene', 'camera', 'renderer', 'controls', userCode);
    fn(THREE, scene, camera, renderer, controls);
}} catch (e) {{
    console.error('ERROR executing Three.js code:', e.message);
    process.exit(1);
}}

let meshCount = 0;
scene.traverse((obj) => {{ if (obj.isMesh) meshCount++; }});
if (meshCount === 0) {{
    console.error('ERROR: No meshes found in scene after executing code');
    process.exit(1);
}}

// Update world matrices so nested Group transforms are baked into vertices.
scene.updateMatrixWorld(true);
const exporter = new STLExporter();
const stlData = exporter.parse(scene, {{ binary: false }});
fs.writeFileSync({stl_path_js}, stlData);
console.log('Exported STL with', meshCount, 'mesh(es)');
{parts_block}
"""
    wrapper_file.write_text(wrapper, encoding="utf-8")

    try:
        proc = run_subprocess(
            [node_cmd, str(wrapper_file)],
            timeout=_THREEJS_TIMEOUT_S,
            cwd=str(pkg_root.parent),
        )
        if proc.returncode != 0:
            return f"Node.js STL export failed: {proc.stderr.strip()[:300]}", None
        if not stl_path.exists():
            return "Node.js STL export produced no file", None
        if proc.stdout.strip():
            logger.info(proc.stdout.strip())
        per_part_manifest: Optional[list] = None
        if parts_meta and parts_report_file.exists():
            try:
                per_part_manifest = json.loads(parts_report_file.read_text())
            except Exception as exc:
                logger.warning("threejs per-part report unreadable: %s", exc)
        return None, per_part_manifest
    except subprocess.TimeoutExpired:
        return f"Node.js STL export timed out ({_THREEJS_TIMEOUT_S}s)", None
    except Exception as e:
        return f"Node.js STL export error: {e}", None
    finally:
        code_file.unlink(missing_ok=True)
        wrapper_file.unlink(missing_ok=True)
        parts_report_file.unlink(missing_ok=True)


def export_threejs(code: str, output_dir: Path) -> Dict[str, Any]:
    """Export Three.js scene geometry to STL via a headless Node.js runtime.

    Requires the vendored Three.js runtime and ``node``; if either is absent the
    result has no STL and a clear error (the case is invalid). When the source
    carries a ``// parts_meta: [...]`` header, also export per-group STLs into
    ``output_dir/parts/`` and write ``parts_meta.json``. No STEP (mesh-only).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / "model.stl"
    result: Dict[str, Any] = {"stl": None, "errors": [], "parts": None}

    node_cmd = _find_node()
    if not node_cmd:
        result["errors"].append("threejs runtime not vendored (see docs/FORMATS.md)")
        return result

    pkg_root, pkg_error = _get_threejs_package_root()
    if pkg_error:
        result["errors"].append(pkg_error)
        return result

    # Per-part export gating: only when the source carries the parts_meta header.
    # Parse errors are recorded as warnings; the unified STL is still attempted.
    parts_meta, parts_meta_err = _extract_threejs_parts_meta(code)
    if parts_meta_err:
        result["errors"].append(parts_meta_err)
    parts_dir = output_dir / "parts" if parts_meta else None

    stl_error, per_part_manifest = _threejs_to_stl(
        code, stl_path, pkg_root, node_cmd, parts_meta=parts_meta, parts_dir=parts_dir,
    )
    if stl_error:
        logger.warning(f"Three.js STL export: {stl_error}")
        result["errors"].append(stl_error)
    else:
        result["stl"] = str(stl_path)

    if parts_meta and per_part_manifest:
        successful = [m for m in per_part_manifest if m.get("stl")]
        for m in per_part_manifest:
            if m.get("error"):
                result["errors"].append(
                    f"threejs per-part '{m.get('name')}': {m['error']}"
                )
        if successful:
            try:
                (output_dir / "parts_meta.json").write_text(
                    json.dumps(per_part_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result["parts"] = per_part_manifest
            except Exception as exc:
                result["errors"].append(f"threejs parts_meta.json write failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# minimal-json (Text2CAD)
# ---------------------------------------------------------------------------

def _extract_json_parts_meta(json_data: dict) -> Tuple[Optional[list], Optional[str]]:
    """Validate the top-level ``parts_meta`` field added by stage-2.

    Schema: ``{"<name>": {"semantic": str, "features": [feature_key, ...]}}``.
    Returns ``(entries, error)`` where each entry is ``{name, semantic, features}``
    in insertion order. Every ``parts`` key must appear in exactly one group.
    """
    if not isinstance(json_data, dict):
        return None, "json_data is not an object"
    meta = json_data.get("parts_meta")
    if meta is None:
        return None, None
    if not isinstance(meta, dict) or not meta:
        return None, "parts_meta must be a non-empty object"

    parts_dict = json_data.get("parts")
    if not isinstance(parts_dict, dict) or not parts_dict:
        return None, "parts_meta requires a non-empty 'parts' dictionary"

    all_keys = set(parts_dict.keys())
    seen_keys: set = set()
    entries: list = []
    for i, (name, group) in enumerate(meta.items()):
        if not isinstance(name, str) or not name.strip():
            return None, f"parts_meta key #{i} is not a non-empty string"
        if not isinstance(group, dict):
            return None, f"parts_meta['{name}'] is not an object"
        semantic = group.get("semantic")
        features = group.get("features")
        if not isinstance(semantic, str) or not semantic.strip():
            return None, f"parts_meta['{name}'].semantic missing or invalid"
        if not isinstance(features, list) or not features:
            return None, f"parts_meta['{name}'].features must be a non-empty list"
        for fk in features:
            if not isinstance(fk, str):
                return None, f"parts_meta['{name}'].features contains a non-string"
            if fk not in all_keys:
                return None, (
                    f"parts_meta['{name}'].features references unknown feature '{fk}'"
                )
            if fk in seen_keys:
                return None, (
                    f"feature '{fk}' is referenced by more than one parts_meta group"
                )
            seen_keys.add(fk)
        entries.append({
            "name": name.strip(),
            "semantic": semantic.strip(),
            "features": list(features),
        })

    missing = sorted(all_keys - seen_keys)
    if missing:
        return None, (
            f"parts_meta does not cover every feature key in 'parts'; missing: {missing}"
        )

    return entries, None


def _build_part_subset_json(json_data: dict, feature_keys: list) -> dict:
    """Return a fresh Text2CAD JSON containing only the requested features.

    Preserves construction order by iterating ``json_data['parts']`` (insertion-
    ordered). Top-level fields other than ``parts`` / ``parts_meta`` carry through.
    """
    feature_set = set(feature_keys)
    parts = json_data.get("parts", {})
    subset = {k: parts[k] for k in parts.keys() if k in feature_set}
    out: dict = {}
    for k, v in json_data.items():
        if k in ("parts", "parts_meta"):
            continue
        out[k] = v
    out["parts"] = subset
    return out


def export_json_cad(json_str: str, output_dir: Path) -> Dict[str, Any]:
    """Parse LLM-generated Text2CAD minimal JSON and convert to STEP/STL.

    Only the minimal-json shape (top-level ``parts``) is supported; the legacy
    DeepCAD ``entities``/``sequence`` shape is rejected with a clear error.

    When a top-level ``parts_meta`` object is present (stage-2 contract), also
    export per-group STLs into ``output_dir/parts/`` and write ``parts_meta.json``.
    A Cut/Join subset without its parent NewBody renders empty -> recorded as a
    per-part failure (success=False), which is intentional.
    """
    from .text2cad_interpreter import export_minimal_json

    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"step": None, "stl": None, "errors": [], "parts": None}

    # Parse JSON (may be wrapped in a markdown code block).
    json_match = re.search(r'```(?:json)?\s*\n(.*?)```', json_str, re.DOTALL)
    raw_json = json_match.group(1).strip() if json_match else json_str.strip()

    try:
        json_data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON from LLM: {e}")
        return result

    # Save pred.json for sequence/topology metrics (kept as an artifact).
    pred_json_path = output_dir / "pred.json"
    pred_json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    result["pred"] = str(pred_json_path)

    if not (isinstance(json_data, dict) and "parts" in json_data):
        result["errors"].append(
            "unsupported JSON shape: expected Text2CAD minimal JSON with a "
            "top-level 'parts' object (the legacy DeepCAD 'entities'/'sequence' "
            "form is not supported)"
        )
        return result

    conversion_result = export_minimal_json(
        str(pred_json_path), str(output_dir),
        assembly_mode=True, output_prefix="model",
    )
    if "error" in conversion_result:
        result["errors"].append(conversion_result["error"])
    else:
        if conversion_result.get("step"):
            result["step"] = conversion_result["step"]
        if conversion_result.get("stl"):
            result["stl"] = conversion_result["stl"]

    # Per-part STL export — only when stage-2 wrote a top-level parts_meta dict.
    parts_meta_entries, parts_meta_err = _extract_json_parts_meta(json_data)
    if parts_meta_err:
        result["errors"].append(f"json parts_meta: {parts_meta_err}")
    elif parts_meta_entries:
        parts_dir = output_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        manifest: list = []
        for i, entry in enumerate(parts_meta_entries):
            subset = _build_part_subset_json(json_data, entry["features"])
            sub_json_path = parts_dir / f"part_{i:02d}.json"
            sub_json_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
            sub_stl_target = parts_dir / f"part_{i:02d}.stl"
            try:
                sub_result = export_minimal_json(
                    str(sub_json_path),
                    str(parts_dir),
                    assembly_mode=True,
                    output_prefix=f"part_{i:02d}",
                )
            except Exception as exc:
                sub_result = {"error": f"exception: {exc}"}
            sub_err = sub_result.get("error") if isinstance(sub_result, dict) else None
            if sub_err or not sub_stl_target.exists():
                manifest.append({
                    "index": i,
                    "name": entry["name"],
                    "semantic": entry["semantic"],
                    "features": entry["features"],
                    "stl": None,
                    "error": sub_err or (
                        "minimal-json subset produced no STL "
                        "(likely a Cut/Join feature subset without its parent NewBody)"
                    ),
                })
                result["errors"].append(
                    f"json per-part '{entry['name']}': {manifest[-1]['error']}"
                )
                continue
            manifest.append({
                "index": i,
                "name": entry["name"],
                "semantic": entry["semantic"],
                "features": entry["features"],
                "stl": str(sub_stl_target),
            })
        if manifest:
            try:
                (output_dir / "parts_meta.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                result["parts"] = manifest
            except Exception as exc:
                result["errors"].append(f"json parts_meta.json write failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compile_code(code: str, format_slug: str, output_dir: Path) -> CompileResult:
    """Compile *code* for *format_slug* into *output_dir* -> :class:`CompileResult`.

    ``valid`` is ``bool(stl)``. Per-part outputs (if any) live under
    ``output_dir/parts/`` with ``output_dir/parts_meta.json``; ``parts_dir`` /
    ``parts_meta`` on the result point at those.
    """
    output_dir = Path(output_dir)
    slug = (format_slug or "").lower()

    if slug == "cadquery":
        raw = export_cadquery(code, output_dir)
    elif slug == "openscad":
        raw = export_openscad(code, output_dir)
    elif slug == "threejs":
        raw = export_threejs(code, output_dir)
    elif slug == "minimal-json":
        raw = export_json_cad(code, output_dir)
    else:
        return CompileResult(
            valid=False,
            errors=[f"Unsupported format for compile: {format_slug}"],
        )

    stl = raw.get("stl")
    parts_meta_path = None
    parts_dir_path = None
    if raw.get("parts"):
        manifest_path = output_dir / "parts_meta.json"
        if manifest_path.exists():
            parts_meta_path = str(manifest_path)
        parts_subdir = output_dir / "parts"
        if parts_subdir.exists():
            parts_dir_path = str(parts_subdir)

    return CompileResult(
        valid=bool(stl),
        stl=stl,
        step=raw.get("step"),
        parts_meta=parts_meta_path,
        parts_dir=parts_dir_path,
        errors=list(raw.get("errors") or []),
        error_details=list(raw.get("error_details") or []),
    )
