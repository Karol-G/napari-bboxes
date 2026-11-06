"""Bounding boxes as napari layers that are visible in every 2-D plane and in 3-D."""

from __future__ import annotations

import numbers
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

#: default class colours (RGB, 0..1), assigned to classes in order of appearance
PALETTE = [
    (0.90, 0.10, 0.12), (0.00, 0.48, 1.00), (0.20, 0.80, 0.20), (0.98, 0.60, 0.00), (0.60, 0.20, 0.80),
    (0.10, 0.75, 0.80), (0.80, 0.80, 0.00), (0.95, 0.40, 0.60), (0.90, 0.30, 0.00), (0.00, 0.70, 0.40),
]

LayerData = Tuple[Any, Dict[str, Any], str]


class BoxShells:
    """A lazy integer array of box outlines: voxel = box number (1-based) on the shell of a box, else 0.

    Each box is given by inclusive index bounds per dim (``bounds``: (n, ndim, 2)). Only the requested slice is
    computed, so any grid size costs no memory. Any slice through a box shows its outline: a rectangle in a 2-D
    plane, a hollow box in 3-D. Where boxes overlap, the later box wins.
    """

    def __init__(self, shape: Sequence[int], bounds: np.ndarray):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.bounds = np.asarray(bounds, dtype=np.int64).reshape(-1, self.ndim, 2)
        self.dtype = np.dtype(np.uint16 if len(self.bounds) < 65535 else np.uint32)
        self.size = int(np.prod(self.shape))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = next(n for n, k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim or not all(isinstance(k, (slice, numbers.Integral)) for k in key):
            return np.asarray(self)[key]
        steps = [k.indices(self.shape[d])[2] if isinstance(k, slice) else 1 for d, k in enumerate(key)]
        if any(step != 1 for step in steps):
            # contiguous range first, then subsample (also handles negative steps)
            contiguous, strided = [], []
            for d, k in enumerate(key):
                if isinstance(k, slice):
                    contiguous.append(slice(None))
                    strided.append(k)
                else:
                    contiguous.append(k)
            return self[tuple(contiguous)][tuple(strided)]
        ranges, squeeze = [], []
        for d, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, _ = k.indices(self.shape[d])
                ranges.append((start, max(stop, start)))
            else:
                k = int(k) + (self.shape[d] if int(k) < 0 else 0)
                if not 0 <= k < self.shape[d]:
                    raise IndexError(f"index {k} is out of bounds for axis {d} with size {self.shape[d]}")
                ranges.append((k, k + 1))
                squeeze.append(d)
        out = np.zeros([b - a for a, b in ranges], dtype=self.dtype)
        for number, box in enumerate(self.bounds, start=1):
            lo, hi = box[:, 0], box[:, 1]
            inter = [(max(a, lo[d]), min(b, hi[d] + 1)) for d, (a, b) in enumerate(ranges)]
            if any(a >= b for a, b in inter):
                continue
            for d in range(self.ndim):
                a, b = ranges[d]
                for plane in {int(lo[d]), int(hi[d])}:
                    if not a <= plane < b:
                        continue
                    region = [slice(s - ranges[e][0], t - ranges[e][0]) for e, (s, t) in enumerate(inter)]
                    region[d] = slice(plane - a, plane - a + 1)
                    out[tuple(region)] = number
        return out.squeeze(axis=tuple(squeeze)) if squeeze else out

    def __array__(self, dtype=None, copy=None):
        array = self[...]
        return array.astype(dtype) if dtype is not None else array

    def __len__(self):
        return self.shape[0]


def _rgba(color) -> Tuple[float, float, float, float]:
    values = np.asarray(color, dtype=float).ravel()
    if values.size not in (3, 4):
        from napari.utils.colormaps.standardize_color import transform_color

        return tuple(float(v) for v in transform_color(color)[0])
    if values.max() > 1.0:
        values = values / 255.0
    return tuple(float(v) for v in values) + ((1.0,) if values.size == 3 else ())


def box_colors(labels: Optional[Sequence], colors: Optional[Mapping] = None, n: int = 0) -> np.ndarray:
    """One RGBA colour per box: ``colors[label]`` when given, else a palette colour per class (or per box)."""
    colors = colors or {}
    keys = list(labels) if labels is not None else list(range(n))
    classes = list(dict.fromkeys(str(k) for k in keys))
    out = []
    for key in keys:
        given = colors.get(key, colors.get(str(key)))
        out.append(_rgba(given) if given is not None else (*PALETTE[classes.index(str(key)) % len(PALETTE)], 1.0))
    return np.array(out, dtype=np.float32).reshape(-1, 4)


def box_texts(labels: Optional[Sequence], scores: Optional[Sequence], names: Optional[Mapping], n: int) -> List[str]:
    """``"name (label) score"`` per box, from what is given."""
    names = names or {}
    texts = []
    for i in range(n):
        parts = []
        if labels is not None:
            label = labels[i]
            name = names.get(label, names.get(str(label)))
            parts.append(f"{name} ({label})" if name is not None and str(name) != str(label) else str(label))
        if scores is not None and scores[i] is not None and np.isfinite(scores[i]):
            parts.append(f"{float(scores[i]):.3f}")
        texts.append(" ".join(parts))
    return texts


def bbox_layers(
    mins,
    maxs,
    *,
    labels: Optional[Sequence] = None,
    scores: Optional[Sequence[float]] = None,
    names: Optional[Mapping] = None,
    colors: Optional[Mapping] = None,
    grid_shape: Optional[Sequence[int]] = None,
    affine: Optional[np.ndarray] = None,
    name: str = "boxes",
    axis_labels: Optional[Sequence[str]] = None,
    units: Optional[Sequence[Union[str, None]]] = None,
    metadata: Optional[dict] = None,
    text: bool = True,
) -> List[LayerData]:
    """napari layer data for axis-aligned boxes, ready for ``viewer._add_layer_from_data`` or a reader.

    Args:
        mins, maxs: (n, ndim) inclusive box corners in index coordinates of a grid (voxel centres), ndim 2 or 3.
        labels: one class per box (int or str); colours and names are looked up by it.
        scores: one confidence per box (shown with the label).
        names: class -> display name, e.g. ``{1: "tumor"}``.
        colors: class -> colour (RGB or RGBA, 0..1 or 0..255, or a colour name); default: a palette per class.
        grid_shape: the grid the boxes live on (3-D boxes are drawn on it); default: just large enough.
        affine: (ndim+1, ndim+1) grid index -> world transform, e.g. the affine of the image the boxes belong
            to; the boxes then align with that image however it is displayed.
        name, axis_labels, units, metadata: passed to the layer.
        text: show label and score next to 2-D boxes.

    Returns:
        2-D boxes: one Shapes layer of rectangles. 3-D boxes: one Labels layer of box outlines (a slice in any
        plane shows the rectangle where it cuts the box; the 3-D view shows the box); hovering shows label, name
        and score.
    """
    mins = np.atleast_2d(np.asarray(mins, dtype=np.float64))
    maxs = np.atleast_2d(np.asarray(maxs, dtype=np.float64))
    if mins.shape != maxs.shape:
        raise ValueError(f"mins {mins.shape} and maxs {maxs.shape} differ in shape")
    lo, hi = np.minimum(mins, maxs), np.maximum(mins, maxs)
    n, ndim = lo.shape
    if ndim not in (2, 3):
        raise ValueError(f"boxes must be 2-D or 3-D, got {ndim}-D")
    for given, what in ((labels, "labels"), (scores, "scores")):
        if given is not None and len(given) != n:
            raise ValueError(f"{what} has {len(given)} entries for {n} boxes")
    affine = np.eye(ndim + 1) if affine is None else np.asarray(affine, dtype=np.float64)
    if affine.shape != (ndim + 1, ndim + 1):
        raise ValueError(f"affine must be {ndim + 1}x{ndim + 1} for {ndim}-D boxes, got {affine.shape}")
    color = box_colors(labels, colors, n)
    texts = box_texts(labels, scores, names, n)
    common: Dict[str, Any] = {"name": name, "affine": affine}
    if axis_labels is not None:
        common["axis_labels"] = list(axis_labels)
    if units is not None:
        common["units"] = list(units)
    if metadata is not None:
        common["metadata"] = metadata

    if ndim == 2:
        rects = np.stack([np.stack([lo[:, 0], lo[:, 1]], 1), np.stack([lo[:, 0], hi[:, 1]], 1),
                          np.stack([hi[:, 0], hi[:, 1]], 1), np.stack([hi[:, 0], lo[:, 1]], 1)], axis=1)
        kwargs = {**common, "shape_type": "rectangle", "face_color": "transparent", "edge_color": color,
                  "edge_width": 1.0, "features": {"box": texts}}
        if text and any(texts):
            kwargs["text"] = {"string": "{box}", "anchor": "upper_left", "size": 10, "color": color}
        return [(rects, kwargs, "shapes")]

    bounds = np.stack([np.round(lo), np.round(hi)], axis=-1).astype(np.int64)
    if grid_shape is None:
        grid_shape = np.maximum(bounds[..., 1].max(axis=0) + 1, 1) if n else (1,) * ndim
    features = {"index": list(range(1, n + 1)), "box": texts,
                "label": [str(v) for v in labels] if labels is not None else [""] * n,
                "score": [float(v) if v is not None else float("nan") for v in scores] if scores is not None
                else [float("nan")] * n}
    from napari.utils.colormaps import DirectLabelColormap

    color_dict = {None: (0.0, 0.0, 0.0, 0.0), 0: (0.0, 0.0, 0.0, 0.0)}
    color_dict.update({i: tuple(c) for i, c in enumerate(color, start=1)})
    kwargs = {**common, "features": features, "colormap": DirectLabelColormap(color_dict=color_dict), "opacity": 1.0}
    return [(BoxShells(grid_shape, bounds), kwargs, "labels")]


def add_bboxes(viewer, mins, maxs, **kwargs) -> list:
    """Add boxes to ``viewer`` (see ``bbox_layers`` for the arguments); returns the new layers."""
    layers = []
    for data, layer_kwargs, layer_type in bbox_layers(mins, maxs, **kwargs):
        layers.append(getattr(viewer, f"add_{layer_type}")(data, **layer_kwargs))
    return layers
