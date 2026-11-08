# Model Viewer

The Model Viewer displays static scenes from self-contained glTF 2.0 GLB
files. Open **App > Model Viewer**, then choose a `.glb` file. The same
application can be launched from the repository root:

```sh
python3 -m solvcon.apps.modelview /path/to/model.glb
```

Use the Python interpreter matching the built `_solvcon` extension, with
PySide6 available. Build the extension with `make` first.

The part tree preserves node hierarchy and names. A checkbox toggles a
subtree; selection reports its vertices, triangles, and extent. Drag in the
3D view to orbit, scroll to zoom, and use the view buttons or Fit to frame
the model. Opening another file replaces the scene. A malformed file leaves
the existing scene intact.

The display uses metres and a proper basis rotation: source glTF +Z maps to
display +X, source +X to display +Y, and source +Y to display +Z. Front, Side,
and Top look along the display +X, -Y, and +Z axes. These are coordinate views,
not automatic recognition of a vehicle's heading. The source node origin,
scale, hierarchy, and pivots remain in the loaded model. Static render buffers
have accumulated transforms baked in; normals use inverse-transpose
transforms and reflected instances reverse winding.

## Supported subset

- GLB 2.0 with an embedded geometry buffer and a selected/default scene.
- Triangle primitives, authored normals or area-weighted normal fallback,
  float32 positions/normals/UVs, unsigned 8/16/32-bit indices, unindexed geometry,
  interleaved attributes, and multiple primitives.
- Hierarchies with matrix or translation/rotation/scale transforms, shared
  meshes, and repeated display names.

The viewer displays `baseColorFactor` and embedded PNG/JPEG base-color
textures through `TEXCOORD_0`, with repeat/clamp/mirrored-repeat sampling and
OPAQUE/MASK/BLEND alpha modes. Textures are sampled as sRGB; material factors
and lighting are applied in linear color space. Full PBR reflections,
metalness/roughness textures, and normal maps are not rendered. Transparent
objects use a separate pass, with approximate ordering rather than per-pixel
transparency sorting. Animation-bearing assets can show their
static pose, with a notice. Skinning, morph targets, sparse/normalized
attributes, external geometry/image files, non-triangle primitives, compression,
and required extensions are rejected with an explanation. Singular transforms
cannot produce valid surface normals and are rejected by the scene adapter.

GLB assets can be exported from Blender. Disable mesh compression and export
static mesh geometry for this initial subset. A successful structural load
does not establish fidelity of materials or animation.

## Rendering API

`RDomainWidget.addSurfaceObject(name, positions, normals, indices)` accepts
`SimpleArrayFloat32` position/normal arrays of shape `(nvert, 3)` and a
`SimpleArrayUint32` index array of shape `(ntri, 3)`. It owns the uploaded
geometry and does not require a volume mesh. Existing named-object transform,
color, opacity, and visibility controls apply. `removeObject(name)` and
`clearObjects()` remove imported objects and rebuild the framing bounds.

`addTexturedSurfaceObject(name, positions, normals, texcoords, indices,
image_data, alpha_mode='OPAQUE', alpha_cutoff=0.5, alpha_factor=1.0,
wrap_s=10497, wrap_t=10497)` adds a base-color texture. UVs are float32
`(nvert, 2)` arrays; `image_data` contains encoded PNG/JPEG bytes. Images are
limited to 8192 pixels per dimension. Wrap codes are the glTF/OpenGL values
10497 (repeat), 33071 (clamp), and 33648 (mirrored repeat). For this textured
path `setObjectColor` supplies a linear RGB material factor.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
