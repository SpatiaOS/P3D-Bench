"""OpenCASCADE (OCP) STEP single-view renderer for the **input image**.

Ported verbatim-ish from the source ``render/occ_renderer.py`` so the OCC
``gt_render.png`` produced by the PREPARE stage is pixel-equivalent to the
research ``_shared_cache`` it reproduces. Renders a STEP file directly via
OCP V3d + AIS (no mesh tessellation), preserving exact B-Rep geometry.

This is the canonical **input** render for image-/assembly-3d (and the
text-to-3d QA answerer render). The judge multiview is Blender clay
(:mod:`p3dbench.render.blender`) — there is no pyrender path here.

The camera geometry reuses the canonical ``ORIENT_MAT`` / ``VIEW_ANGLES``
constants from :mod:`p3dbench.render.occ` so OCC and Blender renders share one
orientation. Requires Xvfb (headless X11) + OCP (ships with the ``cadquery``
extra). The public entry point is :func:`render_occ_single`, which runs the
renderer in a subprocess for OCP resource isolation.
"""

from __future__ import annotations

import logging
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

from .occ import DEFAULT_RENDER_RESOLUTION, ORIENT_MAT

logger = logging.getLogger(__name__)

# Repo root (…/P3D-Bench) so the rendering subprocess can ``import p3dbench``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Tuned (in the source pipeline) to match the CAD render look: darker silver
# faces, crisp dark edges, oversampled output. Kept identical for parity.
_OCC_FACE_RGB = (0.55, 0.55, 0.57)
_OCC_EDGE_RGB = (0.05, 0.05, 0.05)
_OCC_EDGE_WIDTH = 2.2
_OCC_MSAA_SAMPLES = 8
_OCC_RENDER_SCALE = 2.0

# conda's libstdc++ may predate Mesa's swrast GLIBCXX requirement; prefer the
# system copy when present. Harmless no-op if the path is absent.
_SYS_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


class XvfbManager:
    """Context manager for a headless Xvfb virtual X11 display."""

    def __init__(self, display_num: int = 99,
                 screen_size: Tuple[int, int] = (1600, 1600)):
        self.display_num = display_num
        self.screen_size = screen_size
        self._proc = None
        self._old_display = None

    def __enter__(self):
        self._old_display = os.environ.get("DISPLAY")
        w, h = self.screen_size
        if not self._start_with_displayfd(w, h):
            self._start_legacy(w, h)
        os.environ["DISPLAY"] = f":{self.display_num}"
        logger.info("Xvfb started on :%s", self.display_num)
        return self

    def _start_with_displayfd(self, width: int, height: int) -> bool:
        read_fd, write_fd = os.pipe()
        try:
            self._proc = subprocess.Popen(
                ["Xvfb", "-displayfd", str(write_fd), "-screen", "0",
                 f"{width}x{height}x24", "+extension", "GLX", "-ac"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,),
            )
        except Exception:
            self._proc = None
            os.close(read_fd)
            os.close(write_fd)
            return False
        os.close(write_fd)
        try:
            num = self._read_display_number(read_fd)
        finally:
            os.close(read_fd)
        if num is None:
            self._stop()
            return False
        self.display_num = num
        if not self._wait_for_socket(num):
            self._stop()
            return False
        return True

    def _start_legacy(self, width: int, height: int) -> None:
        for num in range(self.display_num, self.display_num + 100):
            if not os.path.exists(f"/tmp/.X{num}-lock"):
                self.display_num = num
                break
        self._proc = subprocess.Popen(
            ["Xvfb", f":{self.display_num}", "-screen", "0",
             f"{width}x{height}x24", "+extension", "GLX", "-ac"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not self._wait_for_socket(self.display_num):
            raise RuntimeError(f"Xvfb failed to start on :{self.display_num}")

    def _read_display_number(self, read_fd: int, timeout_sec: float = 5.0) -> Optional[int]:
        data = b""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                break
            ready, _, _ = select.select([read_fd], [], [], 0.1)
            if not ready:
                continue
            chunk = os.read(read_fd, 32)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        text = data.decode(errors="ignore").strip()
        return int(text) if text.isdigit() else None

    def _wait_for_socket(self, display_num: int, timeout_sec: float = 5.0) -> bool:
        deadline = time.time() + timeout_sec
        sock = f"/tmp/.X11-unix/X{display_num}"
        while time.time() < deadline:
            if os.path.exists(sock):
                return True
            if self._proc and self._proc.poll() is not None:
                return False
            time.sleep(0.1)
        return False

    def _stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._proc:
            self._stop()
            time.sleep(0.1)  # let X11 connections close before exit
        if self._old_display is not None:
            os.environ["DISPLAY"] = self._old_display
        elif "DISPLAY" in os.environ:
            del os.environ["DISPLAY"]
        return False


class OCCRenderer:
    """OCP-based STEP renderer (V3d + AIS). Create inside an Xvfb context."""

    def __init__(self, resolution: int = DEFAULT_RENDER_RESOLUTION):
        import numpy as np
        from OCP.Aspect import Aspect_DisplayConnection
        from OCP.OpenGl import OpenGl_GraphicDriver
        from OCP.V3d import V3d_Viewer
        from OCP.AIS import AIS_InteractiveContext, AIS_Shaded
        from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCP.Xw import Xw_Window

        self.resolution = resolution
        self._orient_inv = np.linalg.inv(np.array(ORIENT_MAT, dtype=float))

        self._display_conn = Aspect_DisplayConnection()
        self._driver = OpenGl_GraphicDriver(self._display_conn)
        self._driver.ChangeOptions().buffersNoSwap = True

        self._viewer = V3d_Viewer(self._driver)
        self._viewer.SetDefaultLights()
        self._viewer.SetLightOn()

        self._view = self._viewer.CreateView()
        self._window = Xw_Window(self._display_conn, "OCC", 0, 0, resolution, resolution)
        self._window.Map()
        self._view.SetWindow(self._window)
        self._view.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))
        params = self._view.ChangeRenderingParams()
        params.NbMsaaSamples = _OCC_MSAA_SAMPLES
        params.RenderResolutionScale = _OCC_RENDER_SCALE

        self._ctx = AIS_InteractiveContext(self._viewer)
        self._ctx.SetDisplayMode(AIS_Shaded, True)
        self._current_ais = None

    def close(self):
        try:
            if self._current_ais is not None:
                self._ctx.Remove(self._current_ais, False)
                self._current_ais = None
            self._view.Remove()
        except Exception:
            pass

    def _load_step(self, step_path: str):
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone

        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            raise RuntimeError(f"STEP read failed: {step_path}")
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            raise RuntimeError(f"Empty shape from {step_path}")
        return shape

    def _display_shape(self, shape):
        from OCP.AIS import AIS_Shape, AIS_Shaded
        from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCP.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NameOfMaterial
        from OCP.Prs3d import Prs3d_LineAspect
        from OCP.Aspect import Aspect_TypeOfLine

        if self._current_ais is not None:
            self._ctx.Remove(self._current_ais, False)

        ais = AIS_Shape(shape)
        ais.SetColor(Quantity_Color(*_OCC_FACE_RGB, Quantity_TOC_RGB))
        ais.SetMaterial(Graphic3d_MaterialAspect(Graphic3d_NameOfMaterial.Graphic3d_NOM_SILVER))
        ais.SetTransparency(0.0)

        drawer = ais.Attributes()
        drawer.SetFaceBoundaryDraw(True)
        drawer.SetWireDraw(True)
        edge = Quantity_Color(*_OCC_EDGE_RGB, Quantity_TOC_RGB)
        drawer.SetFaceBoundaryAspect(
            Prs3d_LineAspect(edge, Aspect_TypeOfLine.Aspect_TOL_SOLID, _OCC_EDGE_WIDTH))
        drawer.SetWireAspect(
            Prs3d_LineAspect(edge, Aspect_TypeOfLine.Aspect_TOL_SOLID, _OCC_EDGE_WIDTH))

        self._ctx.Display(ais, AIS_Shaded, 0, True)
        self._current_ais = ais

    def _get_fit_params(self):
        import numpy as np
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        bbox = Bnd_Box()
        BRepBndLib.Add_s(self._current_ais.Shape(), bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
        half_diag = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) / 2.0
        # Match the pyrender fit: distance = half_diag / tan(fov/2) * padding.
        dist = half_diag / np.tan(0.6 / 2.0) * 1.35
        return center, dist

    def _set_camera(self, center, distance, elev_deg: float, azim_deg: float):
        import numpy as np
        from OCP.gp import gp_Pnt, gp_Dir

        elev, azim = np.radians(elev_deg), np.radians(azim_deg)
        eye_dir = np.array([np.cos(elev) * np.sin(azim),
                            np.cos(elev) * np.cos(azim),
                            np.sin(elev)])
        up_dir = np.array([0.0, 0.0, 1.0])
        eye_step = self._orient_inv @ eye_dir
        up_step = self._orient_inv @ up_dir
        eye = center + distance * eye_step

        cam = self._view.Camera()
        cam.SetEye(gp_Pnt(float(eye[0]), float(eye[1]), float(eye[2])))
        cam.SetCenter(gp_Pnt(float(center[0]), float(center[1]), float(center[2])))
        cam.SetUp(gp_Dir(float(up_step[0]), float(up_step[1]), float(up_step[2])))
        self._view.FitAll(0.08, False)

    def _save_view(self, output_path: str):
        from OCP.Image import Image_AlienPixMap
        from OCP.TCollection import TCollection_AsciiString
        from OCP.V3d import V3d_ImageDumpOptions

        self._view.Redraw()
        opts = V3d_ImageDumpOptions()
        opts.Width = self.resolution
        opts.Height = self.resolution
        pixmap = Image_AlienPixMap()
        if not self._view.ToPixMap(pixmap, opts):
            raise RuntimeError("ToPixMap failed")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if not pixmap.Save(TCollection_AsciiString(str(tmp_path))):
                raise RuntimeError(f"Failed to save temporary render {tmp_path}")
            self._finalize_image(tmp_path, out)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _finalize_image(self, input_path: Path, output_path: Path):
        try:
            from PIL import Image, ImageEnhance, ImageFilter, ImageOps
        except ImportError:
            input_path.replace(output_path)
            return
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            if img.size != (self.resolution, self.resolution):
                img = img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            img = self._recenter_model(img)
            img = ImageEnhance.Brightness(img).enhance(1.01)
            img = ImageEnhance.Contrast(img).enhance(1.08)
            img = ImageOps.autocontrast(img, cutoff=0.2)
            img = img.filter(ImageFilter.UnsharpMask(radius=1.1, percent=125, threshold=2))
            img.save(output_path)

    def _recenter_model(self, img):
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            return img
        arr = np.array(img)
        mask = ~((arr[:, :, 0] > 250) & (arr[:, :, 1] > 250) & (arr[:, :, 2] > 250))
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return img
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        img_center = img.width / 2
        shift_x, shift_y = int(img_center - cx), int(img_center - cy)
        threshold = img.width * 0.05

        content_w = max(1, x_max - x_min + 1)
        content_h = max(1, y_max - y_min + 1)
        target_extent = int(img.width * 0.88)
        scale = min(1.0, target_extent / max(content_w, content_h))
        if scale < 0.999:
            cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            resized = cropped.resize(
                (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
                Image.Resampling.LANCZOS)
            new_img = Image.new("RGB", img.size, (255, 255, 255))
            new_img.paste(resized, ((img.width - resized.width) // 2,
                                    (img.height - resized.height) // 2))
            return new_img
        if abs(shift_x) < threshold and abs(shift_y) < threshold:
            return img
        new_img = Image.new("RGB", img.size, (255, 255, 255))
        new_img.paste(img, (shift_x, shift_y))
        return new_img

    def render_step(self, step_path: str, output_path: str,
                    elev_deg: float = 30, azim_deg: float = 45) -> bool:
        try:
            shape = self._load_step(step_path)
            self._display_shape(shape)
            center, dist = self._get_fit_params()
            self._set_camera(center, dist, elev_deg, azim_deg)
            self._save_view(output_path)
            return True
        except Exception as exc:
            logger.error("OCC render failed for %s: %s", step_path, exc)
            return False


# Subprocess body: render one STEP view, then hard-exit so OCP/X11 teardown
# can't leak into the parent. argv: step out resolution elev azim.
_SUBPROCESS_SCRIPT = (
    "import os, sys; "
    "from p3dbench.render.occ_single import XvfbManager, OCCRenderer; "
    "xvfb = XvfbManager(); xvfb.__enter__(); "
    "r = OCCRenderer(resolution=int(sys.argv[3])); "
    "ok = r.render_step(sys.argv[1], sys.argv[2], float(sys.argv[4]), float(sys.argv[5])); "
    "r.close(); xvfb.__exit__(None, None, None); "
    "os._exit(0 if ok else 1)"
)


def render_occ_single(step_path: str, output_path: str,
                      resolution: int = DEFAULT_RENDER_RESOLUTION,
                      elev_deg: float = 30, azim_deg: float = 45,
                      timeout_sec: int = 180) -> bool:
    """Render a STEP file to a single-view PNG via OCC (subprocess-isolated).

    Returns True iff a non-empty PNG was written. Requires Xvfb on PATH and OCP
    (``cadquery`` extra); the caller (PREPARE) treats False as a hard error.
    """
    import sys

    step = Path(step_path).resolve()
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    env = os.environ.copy()
    if os.path.exists(_SYS_LIBSTDCPP):
        env["LD_PRELOAD"] = _SYS_LIBSTDCPP
    env.pop("DISPLAY", None)

    try:
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(step), str(out),
             str(resolution), str(elev_deg), str(azim_deg)],
            env=env, capture_output=True, text=True,
            timeout=timeout_sec, cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        logger.warning("OCC subprocess render timed out for %s", step)
        return False

    ok = out.exists() and out.stat().st_size > 0
    if not ok:
        logger.warning(
            "OCC subprocess render failed for %s -> %s; rc=%s stderr=%s",
            step, out, result.returncode, (result.stderr or "").strip()[-2000:])
    return ok
