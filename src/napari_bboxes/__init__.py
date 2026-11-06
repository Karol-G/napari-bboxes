"""Axis-aligned 2-D and 3-D bounding boxes as napari layers, visible in every plane and in 3-D."""

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

from ._layers import PALETTE, BoxShells, add_bboxes, bbox_layers, box_colors, box_texts

__all__ = ["PALETTE", "BoxShells", "add_bboxes", "bbox_layers", "box_colors", "box_texts", "__version__"]
