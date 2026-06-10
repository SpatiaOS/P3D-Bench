"""Three.js output format."""

from __future__ import annotations

from .base import Format

SYSTEM_GUIDELINES = """\
You are an expert 3D graphics programmer specializing in Three.js.

You are generating geometry code that will run inside an existing Three.js runtime.
The following variables are ALREADY defined in both the browser viewer and the STL export runtime — do NOT re-create them:
  scene, camera, renderer, controls

Your code should ONLY:
- Define parameters
- Create THREE geometry, materials, and meshes
- Add meshes to `scene` via scene.add(...)
- Optionally adjust camera.position

Prefer using only `scene` and `camera`.
Avoid renderer-specific logic, animation loops, DOM access, and external asset loading.

Use MeshStandardMaterial or MeshPhongMaterial (lighting is already set up).

Example:
```javascript
// Parameters
const width = 10;
const height = 10;
const depth = 10;

// Create geometry
const geometry = new THREE.BoxGeometry(width, height, depth);
const material = new THREE.MeshStandardMaterial({ color: 0x2194ce });
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// Adjust camera
camera.position.set(20, 20, 20);
camera.lookAt(0, 0, 0);
```

Generate ONLY the JavaScript code, no additional explanation.\
"""

THREEJS_FORMAT = Format(
    slug="threejs",
    display_name="Three.js",
    extension=".js",
    system_guidelines=SYSTEM_GUIDELINES,
    fence_langs=("javascript", "js"),
)
