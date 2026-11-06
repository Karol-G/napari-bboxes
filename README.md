# napari-bboxes

<p align="center">
  <a href="https://pypi.org/project/napari-bboxes/"><img src="https://img.shields.io/pypi/v/napari-bboxes?logo=pypi&color=brightgreen" alt="PyPI" align="middle" /></a>
  <a href="https://pypi.org/project/napari-bboxes/"><img src="https://img.shields.io/pypi/pyversions/napari-bboxes?logo=python" alt="Python Version" align="middle" /></a>
  <a href="https://github.com/Karol-G/napari-bboxes/actions"><img src="https://img.shields.io/github/actions/workflow/status/Karol-G/napari-bboxes/test_and_deploy.yml?branch=main&logo=github" alt="Tests" align="middle" /></a>
  <a href="https://github.com/Karol-G/napari-bboxes/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Karol-G/napari-bboxes" alt="License" align="middle" /></a>
</p>

Axis-aligned 2-D and 3-D bounding boxes (detections, annotations) as [napari](https://napari.org) layers that you
can see **in every 2-D plane and in 3-D**, aligned with the image they belong to.

napari's own layers cannot do this for 3-D boxes: a Shapes rectangle lives in one plane, and a Surface mesh shows
nothing in 2-D slices (only faces lying entirely in the slice are drawn). napari-bboxes draws a 3-D box as a
one-voxel **outline** on the image grid, computed on demand: an axial, coronal or sagittal slice shows the rectangle
where it cuts the box, and the 3-D view shows only the box **edges**, so the content inside stays visible. The array
costs no memory.

```bash
pip install napari-bboxes
```

## Usage

```python
import napari
from napari_bboxes import add_bboxes

viewer = napari.Viewer()
image = viewer.add_image(ct, affine=ct_affine)
add_bboxes(
    viewer,
    mins=[[40, 60, 70], [10, 20, 30]],      # inclusive corners in index coordinates of the image grid
    maxs=[[60, 90, 110], [18, 35, 44]],
    labels=[1, 2],                          # class per box
    scores=[0.97, 0.41],                    # optional
    names={1: "tumor", 2: "node"},          # optional display names
    colors={1: (230, 60, 60)},              # optional; other classes get palette colours
    grid_shape=ct.shape,
    affine=image.affine.affine_matrix,      # same index -> world transform as the image: boxes align with it
)
napari.run()
```

- **3-D boxes** become a Labels layer: each box has its own number, coloured by class, and hovering shows
  `name (label) score`.
- **2-D boxes** become a Shapes layer of rectangles, with the same text next to each box.
- `bbox_layers(...)` returns the same as layer data tuples `(data, kwargs, layer_type)`, for napari readers.
- `BoxShells(shape, bounds, mode)` is the lazy outline array on its own. `mode="auto"` (default) draws the box
  surface for 2-D slices and the edges for the 3-D view; `"faces"` or `"edges"` force one of them.

Coordinates are array indices of a grid (voxel centres, inclusive bounds). Boxes in world or physical coordinates
are converted by the caller: map the corners through the inverse of the image affine and take the bounding box.

Used by [napari-mlarray](https://github.com/MIC-DKFZ/napari-mlarray) for the boxes stored in MLArray files.

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
