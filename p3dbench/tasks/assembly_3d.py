"""Assembly-3D task: image + assembly/part text -> unified CAD program.

Adds a fixed, single-call decomposition step (no refine loop) that converts a
stage-1 unified program into a parts-structured form, needed for Part metrics.
The decomposition model never sees the GT part inventory, keeping part-name
alignment honest.
"""

from __future__ import annotations

from ..data.schema import Case
from ..formats.base import Format
from .base import PromptBundle, Task

PROMPT_TEMPLATE = """\
You are reconstructing a Fusion360 mechanical assembly from one rendered image and a structured text description.
Generate {display_name} code that reproduces the assembly as a single unified model.

Text Description (overall caption + part inventory; may end with an "Annotation Caveats" section):
{text}

Modeling guidance:
- Use the rendered image and structured text together, keeping in mind that the annotations may contain small inaccuracies; resolve clear conflicts using the visible geometry, overall proportions, internal consistency, and any caveats noted in the text.
- You MAY use intermediate variables or helper modules internally, but do NOT emit a `parts` list or `parts_meta` header — output a single unified model only.
- Use parametric design with clearly named dimensions.
- Make the code clean; comments only where non-obvious.
- Output {display_name} code only, nothing else.\
"""

# -- stage-2 decomposition --------------------------------------------------

_FENCE = {"cadquery": "python", "openscad": "scad", "threejs": "javascript", "minimal-json": "json"}

_IMAGE_CLAUSE = {
    True: (
        "A rendered preview of the unioned geometry is attached. Use it as a visual cue when "
        "grouping volumes into parts — but the part split must follow the code; do not invent "
        "geometry the code does not express."
    ),
    False: "No render is attached; infer part boundaries from the code's variable / module structure.",
}

_STAGE2_TEMPLATE = """\
You are converting an existing {display_name} CAD program into a PARTS-STRUCTURED form.

The current program (reproduce its geometry EXACTLY; do not alter shapes, dimensions, or placement):
```{fence}
{code}
```

{image_clause}

{contract}

Rules:
- The unioned geometry MUST be identical to the input program (visually and volumetrically). Do NOT redesign or simplify.
- Decide the part decomposition from the code itself: variable names, helper modules, repeated subassemblies, and clearly distinct geometric volumes are all signals. Do NOT assume a target part list — there is no inventory.
- Give each part a short, descriptive `name` that reflects what the part IS (role + geometry class), based purely on what you can read from the code. Example good names: `seal_ring`, `mounting_bracket`, `socket_head`. Avoid generic names like `part_1` unless the code truly offers no semantic cue.
- If a physical feature clearly belongs to one part, assign it to that part. Do not duplicate volume across parts.
- Keep `semantic` labels short and focused on WHAT the part is (role + geometry class), not HOW it was modeled.
- Output {display_name} code only, nothing else.\
"""

_CADQUERY_CONTRACT = """\
REQUIRED OUTPUT STRUCTURE (CadQuery):
After building each part as its own `cq.Workplane`, declare a `parts` list
where every entry is a dict with:
  - "name": a short, stable identifier for this instance (e.g. "seal_ring_1")
  - "semantic": a one-line description of what the part is (<= 25 words)
  - "model": the `cq.Workplane` object for this part, positioned in assembly space
Instances of the same underlying part still need their own entry with a unique
`name` (e.g. seal_ring_1, seal_ring_2). Positioning each part in its own
coordinate frame is fine — apply `.translate(...)` / `.rotate(...)` inside the
entry's `model`.

Example (follow this pattern exactly):
```python
import cadquery as cq

socket_head = cq.Workplane("XY").sphere(6.4)  # bulbous dome body
seal_ring   = cq.Workplane("XY").center(0, 0).circle(7.4).circle(3.4).extrude(8.0)

parts = [
    {"name": "spherical_socket_head", "semantic": "Bulbous dome with concave socket at top",
     "model": socket_head},
    {"name": "seal_ring_1", "semantic": "Toroidal seal ring (O-ring)",
     "model": seal_ring.translate((0, 20, 0))},
    {"name": "seal_ring_2", "semantic": "Toroidal seal ring (O-ring)",
     "model": seal_ring.translate((0, -20, 0))},
]
# Optional: a `result` variable with the union. If omitted, the exporter
# will compute union(parts) automatically.
```\
"""

_OPENSCAD_CONTRACT = """\
REQUIRED OUTPUT STRUCTURE (OpenSCAD):
At the top of the file, emit a JSON header comment named `parts_meta` that
lists one entry per assembly instance. Then define one `module` per entry,
and at the top level render the union.

Example (follow this pattern exactly):
```scad
// parts_meta: [
//   {"name": "spherical_socket_head", "semantic": "Bulbous dome with concave socket at top",
//    "module": "part_spherical_socket_head"},
//   {"name": "seal_ring_1", "semantic": "Toroidal seal ring (O-ring)",
//    "module": "part_seal_ring_1"},
//   {"name": "seal_ring_2", "semantic": "Toroidal seal ring (O-ring)",
//    "module": "part_seal_ring_2"}
// ]

$fn = 64;
module seal_ring() { rotate_extrude() translate([7.4, 0]) circle(r=4.0); }
module part_spherical_socket_head() { sphere(r=6.4); }
module part_seal_ring_1() { translate([0, 20, 0]) seal_ring(); }
module part_seal_ring_2() { translate([0, -20, 0]) seal_ring(); }

union() {
    part_spherical_socket_head();
    part_seal_ring_1();
    part_seal_ring_2();
}
```
Each `parts_meta` entry MUST match a real module name defined below.\
"""

_THREEJS_CONTRACT = """\
REQUIRED OUTPUT STRUCTURE (Three.js):
Group every mesh under a `THREE.Group` named for the assembly part it
represents. Emit ONE header comment `// parts_meta: [...]` listing one
entry per group, then create each group, attach the meshes that compose
it, and `scene.add(group)`. EVERY mesh MUST belong to exactly one group;
do NOT call `scene.add(mesh)` on bare meshes. Apply group-level
`translate` / `rotate` via `group.position.set(...)` /
`group.rotation.set(...)` so the per-group transforms are baked into the
exported geometry when the runner walks `scene.children`.

Each `parts_meta` entry MUST have:
  - "name": short, stable identifier (e.g. "seal_ring_1")
  - "semantic": one-line description of what the part is (<= 25 words)
  - "group_var": the JS variable name holding the THREE.Group

Example (follow this pattern exactly):
```javascript
// parts_meta: [
//   {"name": "spherical_socket_head", "semantic": "Bulbous dome with concave socket at top",
//    "group_var": "g_socket_head"},
//   {"name": "seal_ring_1", "semantic": "Toroidal seal ring (O-ring)",
//    "group_var": "g_seal_ring_1"},
//   {"name": "seal_ring_2", "semantic": "Toroidal seal ring (O-ring)",
//    "group_var": "g_seal_ring_2"}
// ]
const mat = new THREE.MeshStandardMaterial({ color: 0x808080 });

const g_socket_head = new THREE.Group();
g_socket_head.userData = { part_name: "spherical_socket_head",
                            part_semantic: "Bulbous dome with concave socket at top" };
g_socket_head.add(new THREE.Mesh(new THREE.SphereGeometry(6.4), mat));
scene.add(g_socket_head);

const seal_geom = new THREE.TorusGeometry(7.4, 4.0, 16, 64);
const g_seal_ring_1 = new THREE.Group();
g_seal_ring_1.userData = { part_name: "seal_ring_1", part_semantic: "Toroidal seal ring (O-ring)" };
g_seal_ring_1.add(new THREE.Mesh(seal_geom, mat));
g_seal_ring_1.position.set(0, 20, 0);
scene.add(g_seal_ring_1);

const g_seal_ring_2 = new THREE.Group();
g_seal_ring_2.userData = { part_name: "seal_ring_2", part_semantic: "Toroidal seal ring (O-ring)" };
g_seal_ring_2.add(new THREE.Mesh(seal_geom, mat));
g_seal_ring_2.position.set(0, -20, 0);
scene.add(g_seal_ring_2);
```
Each `parts_meta` entry MUST match a real `THREE.Group` variable defined
below; the runner picks groups up by walking `scene.children` and
matching `group_var` against `userData.part_name`.\
"""

_JSON_CONTRACT = """\
REQUIRED OUTPUT STRUCTURE (Text2CAD JSON):
Keep the existing top-level `"parts"` feature dictionary EXACTLY as it
is (do not rename keys, do not reorder, do not change any coordinates,
extrusion depths, or operations — the unioned geometry must remain
identical). Add ONE new top-level field `"parts_meta"` mapping each
semantic part name to the ordered list of feature keys that compose it.

Schema for `"parts_meta"`:
  {"<semantic_name>": {"semantic": "<<= 25 words>",
                        "features": ["part_<N>", "part_<M>", ...]}}

Every key in `"parts"` MUST appear in exactly one group's `features`
list. Order within `features` MUST match the original construction
order in `"parts"` so the exporter can replay NewBody / Join / Cut
operations consistently inside the subset.

Grouping rules (apply in priority order):
1. A `NewBodyFeatureOperation` starts a new semantic part — it MUST be
   the FIRST entry in its group's `features` list.
2. A `JoinFeatureOperation` belongs to the most recent NewBody at the
   same spatial neighborhood (compare `Translation Vector` and
   extrusion extent).
3. A `CutFeatureOperation` belongs to the body it geometrically
   subtracts from (the body whose extruded volume contains the cut
   profile). If you cannot confidently attribute a Cut to one body,
   attach it to the most recent NewBody in construction order.

Example (follow this pattern exactly):
```json
{
  "parts_meta": {
    "bracket": {
      "semantic": "L-shaped mounting bracket with through-hole",
      "features": ["part_1", "part_2"]
    },
    "lid": {
      "semantic": "Round cover plate",
      "features": ["part_3"]
    }
  },
  "parts": {
    "part_1": { "coordinate_system": {}, "sketch": {},
                "extrusion": { "operation": "NewBodyFeatureOperation" } },
    "part_2": { "coordinate_system": {}, "sketch": {},
                "extrusion": { "operation": "CutFeatureOperation" } },
    "part_3": { "coordinate_system": {}, "sketch": {},
                "extrusion": { "operation": "NewBodyFeatureOperation" } }
  }
}
```
Every `part_N` key in `"parts"` MUST appear in exactly one group; no
duplicates, no orphans. The exporter will replay each group's feature
list (in order) to produce that group's STL — a Cut whose parent
NewBody is not in the same group will render empty and that part will
be flagged success=False, which is intentional.\
"""

_CONTRACTS = {
    "cadquery": _CADQUERY_CONTRACT,
    "openscad": _OPENSCAD_CONTRACT,
    "threejs": _THREEJS_CONTRACT,
    "minimal-json": _JSON_CONTRACT,
}


class Assembly3DTask(Task):
    slug = "assembly-3d"
    # Paper: Assembly-3D uses OpenSCAD + CadQuery (minimal-json too limited for
    # multi-part; Three.js meshes don't decompose into per-part solids).
    supported_formats = ("openscad", "cadquery")
    condition_inputs = "image+text"

    def build_prompt(self, fmt: Format, case: Case, image_paths: list[str]) -> PromptBundle:
        self.check_format(fmt)
        if not image_paths:
            raise ValueError(f"Assembly-3D case {case.id} has no input image")
        user = PROMPT_TEMPLATE.format(display_name=fmt.display_name, text=case.input.text)
        return PromptBundle(system=fmt.system_guidelines, user=user, images=image_paths[:1])

    def build_decompose_prompt(self, fmt: Format, code: str, has_image: bool) -> str:
        contract = _CONTRACTS.get(fmt.slug)
        if contract is None:
            raise ValueError(f"No stage-2 decomposition contract for format '{fmt.slug}'")
        return _STAGE2_TEMPLATE.format(
            display_name=fmt.display_name,
            fence=_FENCE[fmt.slug],
            code=code,
            image_clause=_IMAGE_CLAUSE[bool(has_image)],
            contract=contract,
        )


TASK = Assembly3DTask()
