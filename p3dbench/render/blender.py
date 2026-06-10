"""Optional Blender-clay multiview renderer for the Judge bucket.

A thin OPTIONAL alternative to :mod:`p3dbench.render.occ`. The demo judge
defaults to the pyrender path (``occ.py``); this module is only used when a
Blender binary is configured via the ``P3DBENCH_BLENDER`` environment variable.

Output is a porcelain "white clay" render (Cycles, warm key + cool fill + rim
three-point rig over a subtle studio gradient) -- paper-figure quality. The
public entry point mirrors ``occ.render_multiview``:

    render_multiview(mesh_or_step_path, output_dir, *, n_views=4) -> list[str]

Returns ``[]`` if ``P3DBENCH_BLENDER`` is unset / the binary is missing, or on
any render failure.

Dual-mode file (same pattern as the source ``render_blender_clay.py``):

  1. Imported as a regular Python module -> defines the orchestrator API. The
     orchestrator spawns Blender as a subprocess:
     ``blender -b -P this_file -- <mesh.npz> <out_dir> <yfov> <res> <views.json>``.
  2. Re-invoked inside that Blender subprocess (the ``-P`` script) -> runs the
     bpy entry point that builds the studio scene + clay shader and writes PNGs.

Mode is detected by ``__name__ == "__main__" and "--" in sys.argv``.

Mesh geometry crosses the subprocess boundary as a temporary ``.npz`` (vertices
+ faces). Blender's STL import would silently rotate axes and break alignment
with the pyrender / OCC outputs, so we never use it.

Stripped vs. the production source (per the port contract): no concurrency-cap
filelock pool, no ``BlenderRenderError`` taxonomy / ``raise_on_failure``, no
hardcoded user binary path (env ``P3DBENCH_BLENDER`` only).
"""

from __future__ import annotations

import sys as _sys

# Mode detection: are we running as Blender's `-P` script, or imported as a
# regular Python module? The two branches have disjoint import lists (the
# orchestrator pulls numpy + the project mesh helpers; the bpy branch pulls
# bpy / mathutils which the project env lacks). Importing either eagerly would
# crash the other.
_IN_BLENDER = (__name__ == "__main__" and "--" in _sys.argv)


if _IN_BLENDER:
    # =======================================================================
    # bpy entry point
    # Runs inside Blender as `blender -b -P blender.py -- <args>`. Builds the
    # studio scene + porcelain clay shader, writes one PNG per view entry.
    # =======================================================================
    import json
    import math
    import os
    import sys
    from pathlib import Path

    import bpy  # type: ignore[import-not-found]
    import numpy as np  # bundled with Blender
    from mathutils import Vector  # type: ignore[import-not-found]

    # Adjacent faces meeting at < this angle smooth; sharper edges stay crisp.
    AUTO_SMOOTH_ANGLE_DEG = 30.0

    def _find_studio_hdri():
        """Locate Blender's bundled studio.exr (indirect bounce light), or None."""
        try:
            local_root = Path(bpy.utils.resource_path("LOCAL"))
        except Exception:
            return None
        candidate = local_root / "datafiles" / "studiolights" / "world" / "studio.exr"
        return str(candidate) if candidate.exists() else None

    def parse_args():
        argv = sys.argv
        if "--" not in argv:
            raise RuntimeError("expected '--' separator in argv")
        rest = argv[argv.index("--") + 1:]
        mesh_npz = rest[0]
        out_dir = rest[1]
        yfov = float(rest[2])
        resolution = int(rest[3])
        views = json.loads(Path(rest[4]).read_text())
        return mesh_npz, out_dir, yfov, resolution, views

    def reset_scene():
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def build_mesh_from_npz(mesh_npz: str):
        data = np.load(mesh_npz)
        verts = data["vertices"].astype(np.float64)
        faces = data["faces"].astype(np.int64)

        mesh = bpy.data.meshes.new("Mesh")
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new("MeshObj", mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(AUTO_SMOOTH_ANGLE_DEG))
        return obj

    def model_extent(obj):
        """World-space bbox center, half-largest-extent, and zmin."""
        bb = np.array([list(obj.matrix_world @ Vector(v)) for v in obj.bound_box])
        center = bb.mean(axis=0)
        extent = (bb.max(axis=0) - bb.min(axis=0)).max()
        zmin = float(bb[:, 2].min())
        return center, extent / 2.0, zmin

    def build_clay_material(sss_scale: float):
        """Warm porcelain clay with subtle subsurface scattering."""
        mat = bpy.data.materials.new(name="Clay")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)

        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.95, 0.85, 0.69, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.32

        # Specular key name differs between Blender 3.x and 4.x.
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.55
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.55
        if "Specular Tint" in bsdf.inputs:
            try:
                bsdf.inputs["Specular Tint"].default_value = (1.0, 0.97, 0.92, 1.0)
            except (TypeError, ValueError):
                bsdf.inputs["Specular Tint"].default_value = 0.15

        # Subsurface inputs were renamed in Blender 4.0.
        sss_radius = (1.0, 0.5, 0.3)
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.28
            bsdf.inputs["Subsurface Radius"].default_value = sss_radius
            if "Subsurface Scale" in bsdf.inputs:
                bsdf.inputs["Subsurface Scale"].default_value = sss_scale
        elif "Subsurface" in bsdf.inputs:
            bsdf.inputs["Subsurface"].default_value = 0.28
            bsdf.inputs["Subsurface Radius"].default_value = sss_radius

        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs[0], out.inputs[0])
        return mat

    def assign_material(obj, mat):
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    def setup_world(hdri_path, hdri_strength: float = 0.10):
        """HDRI for bounces; vertical gradient x radial vignette for camera bg."""
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        world = bpy.context.scene.world
        world.use_nodes = True
        nt = world.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)

        # HDRI branch (lighting only).
        coord_h = nt.nodes.new("ShaderNodeTexCoord")
        mapping_h = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(coord_h.outputs["Generated"], mapping_h.inputs["Vector"])
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        if hdri_path:
            env.image = bpy.data.images.load(hdri_path)
        nt.links.new(mapping_h.outputs[0], env.inputs[0])
        bg_light = nt.nodes.new("ShaderNodeBackground")
        bg_light.inputs["Strength"].default_value = hdri_strength
        nt.links.new(env.outputs[0], bg_light.inputs[0])

        # Camera-visible: vertical gradient x radial vignette.
        coord_g = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(coord_g.outputs["Window"], sep.inputs[0])

        ramp_v = nt.nodes.new("ShaderNodeValToRGB")
        ev = ramp_v.color_ramp.elements
        ev[0].position = 0.0
        ev[0].color = (0.30, 0.31, 0.34, 1.0)
        ev[1].position = 1.0
        ev[1].color = (0.58, 0.56, 0.52, 1.0)
        nt.links.new(sep.outputs["Y"], ramp_v.inputs["Fac"])

        grad = nt.nodes.new("ShaderNodeTexGradient")
        grad.gradient_type = "SPHERICAL"
        map_n = nt.nodes.new("ShaderNodeMapping")
        map_n.inputs["Location"].default_value = (-0.5, -0.5, 0.0)
        map_n.inputs["Scale"].default_value = (2.0, 2.0, 2.0)
        nt.links.new(coord_g.outputs["Window"], map_n.inputs["Vector"])
        nt.links.new(map_n.outputs["Vector"], grad.inputs["Vector"])
        ramp_r = nt.nodes.new("ShaderNodeValToRGB")
        er = ramp_r.color_ramp.elements
        er[0].position = 0.0
        er[0].color = (0.85, 0.85, 0.85, 1.0)
        er[1].position = 0.7
        er[1].color = (1.05, 1.05, 1.05, 1.0)
        nt.links.new(grad.outputs["Color"], ramp_r.inputs["Fac"])

        mult = nt.nodes.new("ShaderNodeMixRGB")
        mult.blend_type = "MULTIPLY"
        mult.inputs["Fac"].default_value = 1.0
        nt.links.new(ramp_v.outputs[0], mult.inputs["Color1"])
        nt.links.new(ramp_r.outputs[0], mult.inputs["Color2"])
        bg_cam = nt.nodes.new("ShaderNodeBackground")
        bg_cam.inputs["Strength"].default_value = 1.0
        nt.links.new(mult.outputs[0], bg_cam.inputs[0])

        # Mix: camera ray -> gradient, others -> HDRI.
        light_path = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
        nt.links.new(bg_light.outputs[0], mix.inputs[1])
        nt.links.new(bg_cam.outputs[0], mix.inputs[2])

        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(mix.outputs[0], out.inputs[0])

    def setup_render(resolution: int):
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"

        # Prefer GPU (OPTIX > CUDA); fall back to CPU. Force CPU with
        # P3DBENCH_CYCLES_DEVICE=cpu (headless CI without a usable GPU).
        force_cpu = os.environ.get("P3DBENCH_CYCLES_DEVICE", "").lower() == "cpu"
        if force_cpu:
            scene.cycles.device = "CPU"
        else:
            cprefs = bpy.context.preferences.addons["cycles"].preferences
            selected_backend = None
            for backend in ("OPTIX", "CUDA"):
                try:
                    cprefs.compute_device_type = backend
                except TypeError:
                    continue
                cprefs.refresh_devices()
                if any(d.type == backend for d in cprefs.devices):
                    selected_backend = backend
                    break
            if selected_backend:
                for d in cprefs.devices:
                    d.use = (d.type == selected_backend)
                scene.cycles.device = "GPU"
            else:
                scene.cycles.device = "CPU"

        scene.cycles.samples = 128
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 0
        scene.cycles.transparent_max_bounces = 1

        # AgX (Blender 4.0+) has softer highlight rolloff than Filmic.
        view_xform_items = scene.view_settings.bl_rna.properties[
            "view_transform"
        ].enum_items
        if "AgX" in view_xform_items:
            scene.view_settings.view_transform = "AgX"
            scene.view_settings.look = "AgX - Medium High Contrast"
        else:
            scene.view_settings.view_transform = "Filmic"
            scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = -0.4
        scene.view_settings.gamma = 1.0
        scene.display_settings.display_device = "sRGB"

        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = False

    def make_camera(yfov: float, max_dist: float):
        cam_data = bpy.data.cameras.new("Cam")
        cam_data.type = "PERSP"
        cam_data.lens_unit = "FOV"
        cam_data.sensor_fit = "VERTICAL"
        cam_data.angle_y = yfov
        cam_data.clip_start = max_dist * 0.001
        cam_data.clip_end = max_dist * 100.0
        cam_obj = bpy.data.objects.new("Cam", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        return cam_obj

    def place_camera(cam_obj, eye, center):
        cam_obj.location = Vector(eye)
        direction = Vector(center) - Vector(eye)
        cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    def _ensure_collection_empty(name_prefix: str):
        to_remove = [o for o in bpy.data.objects if o.name.startswith(name_prefix)]
        for o in to_remove:
            bpy.data.objects.remove(o, do_unlink=True)

    def _add_area_light(name, pos, target, size_xy, energy, color=(1.0, 1.0, 1.0)):
        light_data = bpy.data.lights.new(name=f"{name}_data", type="AREA")
        light_data.shape = "RECTANGLE"
        light_data.size = max(size_xy[0], 0.05)
        light_data.size_y = max(size_xy[1], 0.05)
        light_data.energy = energy
        light_data.color = color
        light_obj = bpy.data.objects.new(name, light_data)
        light_obj.location = Vector([float(x) for x in pos])
        direction = Vector([float(x) for x in target]) - light_obj.location
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(light_obj)
        return light_obj

    def add_three_point_lights(center, radius, right, up, fwd):
        """Camera-relative three-point rig: warm key + cool fill + rim.

        ``fwd`` = model->camera. Energy is normalized by distance^2 so on-surface
        irradiance is roughly view-invariant.
        """
        right = np.asarray(right, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        fwd = np.asarray(fwd, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)

        key_pos = center + radius * (-right * 2.4 + up * 1.0 + fwd * 1.6)
        fill_pos = center + radius * (right * 2.0 - up * 0.2 + fwd * 1.5)
        rim_pos = center + radius * (right * 0.6 + up * 1.4 - fwd * 2.0)

        def _power(target_irradiance, pos):
            d = float(np.linalg.norm(pos - center))
            return target_irradiance * 4.0 * math.pi * (d ** 2)

        _add_area_light(
            "Key", key_pos, center,
            size_xy=(radius * 1.6, radius * 1.2),
            energy=_power(1.7, key_pos),
            color=(1.00, 0.93, 0.83),
        )
        _add_area_light(
            "Fill", fill_pos, center,
            size_xy=(radius * 3.5, radius * 2.5),
            energy=_power(0.40, fill_pos),
            color=(0.75, 0.86, 1.00),
        )
        _add_area_light(
            "Rim", rim_pos, center,
            size_xy=(radius * 1.8, radius * 1.0),
            energy=_power(1.0, rim_pos),
            color=(1.00, 0.95, 0.88),
        )

    def maybe_add_floor(eye, zmin, radius):
        """Shadow-catcher plane at z = zmin; skipped for bottom (looking-up) views."""
        if float(eye[2]) <= zmin + 1e-6:
            return None

        s = radius * 30.0
        v0 = (s, s, zmin)
        v1 = (-s, s, zmin)
        v2 = (-s, -s, zmin)
        v3 = (s, -s, zmin)

        mesh = bpy.data.meshes.new("Floor_mesh")
        mesh.from_pydata([v0, v1, v2, v3], [], [(0, 1, 2, 3)])
        mesh.update()
        obj = bpy.data.objects.new("Floor", mesh)
        obj.is_shadow_catcher = True
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def render_to(path: str):
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)

    def bpy_main():
        mesh_npz, out_dir, yfov, resolution, views = parse_args()
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        reset_scene()
        obj = build_mesh_from_npz(mesh_npz)
        center, radius, zmin = model_extent(obj)
        mat = build_clay_material(sss_scale=radius * 0.04)
        assign_material(obj, mat)
        setup_world(_find_studio_hdri())
        setup_render(resolution)

        max_dist = max(
            float(np.linalg.norm(np.asarray(v["eye"]) - center)) for v in views
        )
        cam_obj = make_camera(yfov, max_dist)

        for v in views:
            for prefix in ("Key", "Fill", "Rim", "Floor"):
                _ensure_collection_empty(prefix)
            place_camera(cam_obj, v["eye"], v["center"])
            add_three_point_lights(center, radius, v["right"], v["up"], v["fwd"])
            maybe_add_floor(v["eye"], zmin, radius)
            render_to(v["out"])

    if __name__ == "__main__":
        bpy_main()


else:
    # =======================================================================
    # Orchestrator (regular project Python). Spawns Blender as a subprocess
    # and exposes the public render_multiview API.
    # =======================================================================
    import json
    import logging
    import os
    import subprocess
    import tempfile
    from pathlib import Path
    from typing import List, Optional

    from ..utils import require

    logger = logging.getLogger(__name__)

    _RENDER_EXTRA = "render"

    PYRENDER_DEFAULT_YFOV = 0.6
    CLAY_DEFAULT_RESOLUTION = 768
    DEFAULT_BLENDER_TIMEOUT = 900  # seconds for one subprocess (all views)

    # Re-invoke this very file: Blender's `-P` runs it, hits _IN_BLENDER, lands
    # in bpy_main().
    _BPY_SCRIPT = str(Path(__file__).resolve())

    def _resolve_blender_bin() -> Optional[str]:
        """Blender binary from $P3DBENCH_BLENDER, or None when unset."""
        return os.environ.get("P3DBENCH_BLENDER") or None

    def is_blender_available() -> bool:
        binary = _resolve_blender_bin()
        return bool(binary) and Path(binary).exists()

    def _np():
        return require("numpy", _RENDER_EXTRA, "Blender multiview rendering")

    def _coerce_to_oriented_mesh(mesh_or_path):
        """Load a path -> orient -> nudge coincident components; pass meshes through."""
        if hasattr(mesh_or_path, "vertices") and hasattr(mesh_or_path, "faces"):
            return mesh_or_path
        from ..compile.step_mesh import load_mesh
        from .occ import orient_mesh_for_render

        mesh = load_mesh(str(mesh_or_path))
        if mesh is None:
            return None
        mesh = orient_mesh_for_render(mesh)
        mesh = _disambiguate_coincident_components(mesh, eps=1e-4)
        return mesh

    def _disambiguate_coincident_components(mesh, eps: float = 1e-4):
        """Nudge each non-largest connected component by ``eps`` to break
        coincident CSG bbox planes (Cycles/Embree shadow-ray epsilon black-face
        bug). Offset ~0.0085% of body size -- below any metric threshold.
        """
        np = _np()
        ccs = mesh.split(only_watertight=False)
        if len(ccs) <= 1:
            return mesh
        main_idx = max(range(len(ccs)), key=lambda i: len(ccs[i].faces))
        main_centroid = ccs[main_idx].centroid
        moved = []
        for i, c in enumerate(ccs):
            if i == main_idx:
                moved.append(c)
                continue
            c2 = c.copy()
            d = c.centroid - main_centroid
            n = np.linalg.norm(d)
            if n < 1e-9:
                d = np.array([0.0, -1.0, 0.0])
                n = 1.0
            c2.apply_translation(d / n * eps)
            moved.append(c2)
        import trimesh  # already guarded by load_mesh / _coerce caller
        return trimesh.util.concatenate(moved)

    def _write_mesh_npz(mesh, out_path: Path) -> None:
        np = _np()
        # float32 vertices + int32 faces -- NOT Blender's STL import (axis swap).
        np.savez(
            out_path,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
        )

    def _build_view_entry(eye, center, out_path: str) -> dict:
        from .occ import look_at_axes

        np = _np()
        forward, right, up = look_at_axes(np.asarray(eye), np.asarray(center))
        cam_z = -forward  # model->camera; the bpy side uses this as 'fwd'
        return {
            "eye": [float(x) for x in eye],
            "center": [float(x) for x in center],
            "right": [float(x) for x in right],
            "up": [float(x) for x in up],
            "fwd": [float(x) for x in cam_z],
            "out": out_path,
        }

    def _run_blender(mesh_npz: Path, views, out_dir: Path,
                     yfov: float, resolution: int, timeout_s: int) -> bool:
        blender_bin = _resolve_blender_bin()
        if not blender_bin or not Path(blender_bin).exists():
            logger.warning("Blender binary unavailable (set $P3DBENCH_BLENDER)")
            return False

        with tempfile.NamedTemporaryFile(
            "w", suffix=".views.json", delete=False, dir=str(out_dir)
        ) as f:
            json.dump(list(views), f)
            views_json = f.name

        cmd = [
            blender_bin, "-b", "-P", _BPY_SCRIPT, "--",
            str(mesh_npz), str(out_dir), str(yfov), str(resolution), views_json,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            logger.warning("Blender clay render timed out after %ds", timeout_s)
            return False
        finally:
            try:
                os.unlink(views_json)
            except OSError:
                pass

        if proc.returncode != 0:
            logger.warning(
                "Blender clay render failed (rc=%d). stderr tail:\n%s",
                proc.returncode, "\n".join((proc.stderr or "").splitlines()[-20:]))
            return False

        for v in views:
            if not Path(v["out"]).exists():
                logger.warning("Blender returned 0 but %s missing", v["out"])
                return False
        return True

    def render_multiview(mesh_or_step_path, output_dir, *,
                         n_views: int = 4) -> List[str]:
        """Render ``n_views`` clay judge views via a single Blender subprocess.

        Returns ``[]`` if ``P3DBENCH_BLENDER`` is unset / missing, the mesh fails
        to load, or the subprocess fails (all-or-nothing).
        """
        from .occ import VIEW_ANGLES, eye_from_angles, fit_perspective_camera

        if not is_blender_available():
            logger.info("Blender unavailable; skipping clay multiview "
                        "(set $P3DBENCH_BLENDER to enable)")
            return []
        if n_views < 1:
            return []
        n_views = min(n_views, len(VIEW_ANGLES))

        try:
            mesh = _coerce_to_oriented_mesh(mesh_or_step_path)
        except Exception as exc:
            logger.warning("Failed to load mesh from %s: %s", mesh_or_step_path, exc)
            return []
        if mesh is None:
            logger.warning("Failed to load mesh from %s", mesh_or_step_path)
            return []

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Fit once at view 0 (elev 30 / azim 45); reuse so all views share scale.
            center, distance, _ = fit_perspective_camera(
                mesh, elev_deg=30.0, azim_deg=45.0, yfov=PYRENDER_DEFAULT_YFOV)

            views: List[dict] = []
            out_paths: List[str] = []
            for i in range(n_views):
                elev, azim = VIEW_ANGLES[i]
                eye = eye_from_angles(center, distance, elev, azim)
                out_path = str(out_dir / f"view_{i:03d}.png")
                views.append(_build_view_entry(eye, center, out_path))
                out_paths.append(out_path)
        except Exception as exc:
            logger.warning("Failed to prepare views for %s: %s",
                           mesh_or_step_path, exc)
            return []

        with tempfile.TemporaryDirectory(prefix="blender_clay_") as tmpdir:
            npz = Path(tmpdir) / "mesh.npz"
            try:
                _write_mesh_npz(mesh, npz)
            except Exception as exc:
                logger.warning("Failed to write mesh npz: %s", exc)
                return []
            ok = _run_blender(
                npz, views, Path(tmpdir),
                yfov=PYRENDER_DEFAULT_YFOV,
                resolution=CLAY_DEFAULT_RESOLUTION,
                timeout_s=DEFAULT_BLENDER_TIMEOUT,
            )
            if not ok:
                return []

        return out_paths
