# Output formats

A model writes a program in one of four executable CAD formats; the compile
stage turns that program into a STEP and/or STL. **Valid** means a non-empty STL
was produced. Each format is one module under [`p3dbench/formats/`](../p3dbench/formats/)
(prompt side: system guidelines + code extraction) delegating to
[`p3dbench/compile/`](../p3dbench/compile/) (compile side).

| Slug | Display | Ext | Compiler | STEP? | Runtime needed |
|------|---------|-----|----------|-------|----------------|
| `minimal-json` | JSON | `.json` | CadQuery interpreter (`compile/text2cad_interpreter.py`) | yes | `[geometry]` extra |
| `openscad` | OpenSCAD | `.scad` | `openscad` CLI → STL | no | `openscad` binary |
| `cadquery` | CadQuery | `.py` | sandboxed `cadquery` subprocess → STEP → STL | yes | `[cadquery]` extra |
| `threejs` | Three.js | `.js` | Node.js + vendored three.js → STL | no | `node` (runtime ships in-repo) |

## Code extraction

`Format.extract_code` takes the **first** fenced code block tagged with the
format's own language (or `python`/`javascript`/`scad`/`json`), falling back to
the whole stripped response.

## minimal-json (Text2CAD construction history)

Top-level `parts`, each with `coordinate_system` (Euler ZYX degrees + translation),
`sketch` (faces → loops → `line_*`/`arc_*`/`circle_*`), and `extrusion`
(`extrude_depth_towards_normal` / `_opposite_normal`, `operation` ∈
`NewBody`/`Join`/`Cut`/`Intersect`). Coordinates are already in physical units —
`sketch_scale` is **not** applied by this exporter.

Faithfulness notes (the interpreter reproduces the research behavior, it does not
"fix" it): a two-sided extrusion (`d_fwd > 0` and `d_rev > 0`) is approximated as
symmetric via CadQuery `both=True`; hole loops extrude single-direction only;
parts are replayed in **insertion order**.

## openscad

Compiled by the `openscad` binary (the Manifold backend is auto-detected and
preferred for speed). Mesh-only — no STEP. Per-part decomposition uses a
`// parts_meta:` JSON header comment naming one `module` per part; top-level
`$fn/$fa/$fs` specials are copied into each per-part wrapper so per-part
tessellation matches the union.

## cadquery

Executed in a **subprocess with a timeout** (correctness isolation for untrusted
code, not a throughput pool). A single-part program exports `result`; an assembly
program exports a `parts = [{name, semantic, model}, ...]` list. STEP → STL via
the OCP tessellator (linear deflection 0.01, adaptively refined to 0.05 % of the
bbox; gmsh fallback at char-length 0.02 × bbox guards against degenerate solids).

## threejs

Run under a headless Node.js runtime that provides the same `scene/camera/
renderer/controls` globals the browser viewer would; the scene is exported to STL
via `STLExporter`. Per-part decomposition groups meshes under named `THREE.Group`s.

**Vendored runtime:** the Three.js runtime ships in-repo under `p3dbench/compile/three/`
(a minimal Three.js distribution: `build/three.module.js` + `build/three.core.js`,
`examples/jsm/exporters/STLExporter.js`, and `package.json` whose `exports`/`name`
let `STLExporter.js` self-resolve its `import ... from 'three'`). No `npm install`
is needed; only `node` must be on PATH. If `node` is absent the Three.js compiler
returns `valid=False` with a clear message. See the module docstring in
`p3dbench/compile/exporter.py` for the exact layout.
