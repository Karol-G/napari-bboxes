import numpy as np
import pytest
from napari.components import ViewerModel

from napari_bboxes import BoxShells, add_bboxes, bbox_layers, box_colors, box_texts


def _add(layer_data):
    viewer = ViewerModel()
    for data, kwargs, layer_type in layer_data:
        viewer._add_layer_from_data(data, kwargs, layer_type)
    return viewer


def test_shells_are_hollow_box_outlines():
    shells = BoxShells((8, 9, 10), np.array([[[1, 4], [2, 6], [3, 8]]]))
    full = np.asarray(shells)
    on = np.argwhere(full)
    assert on.min(0).tolist() == [1, 2, 3] and on.max(0).tolist() == [4, 6, 8]
    assert not full[2:4, 3:6, 4:8].any()                 # interior empty
    assert full[1, 2:7, 3:9].all() and full[4, 2:7, 3:9].all()   # end faces of dim 0 filled
    assert full[2, 2, 3:9].all() and not full[2, 3, 4]   # middle slice: only the rectangle outline


@pytest.mark.parametrize("key", [
    (slice(None),), (2,), (Ellipsis, 5), (slice(1, 5), 3), (-1,), (slice(None, None, 2), slice(None), 4),
    (slice(None, None, -1), 2), (3, slice(1, 7, 3), slice(None)),
])
def test_shell_slices_match_the_full_array(key):
    shells = BoxShells((8, 9, 10), np.array([[[1, 4], [2, 6], [3, 8]], [[3, 7], [0, 8], [5, 9]]]))
    full = np.asarray(shells)
    np.testing.assert_array_equal(shells[key], full[key])


def test_overlapping_boxes_later_wins_and_numbers_are_one_based():
    shells = BoxShells((5, 5, 5), np.array([[[0, 4]] * 3, [[1, 3]] * 3]))
    full = np.asarray(shells)
    assert set(np.unique(full)) == {0, 1, 2}
    assert full[1, 1, 1] == 2


def test_3d_boxes_become_a_labels_layer_aligned_through_the_affine():
    affine = np.array([[-2.0, 0, 0, 10], [0, 0.5, 0, -3], [0, 0, 1.5, 7], [0, 0, 0, 1]])
    viewer = _add(bbox_layers([[1, 2, 3]], [[4, 6, 8]], labels=[1], scores=[0.875], names={1: "tumor"},
                              grid_shape=(10, 10, 10), affine=affine, axis_labels=("I", "P", "L"), name="det"))
    layer = viewer.layers[0]
    assert layer._type_string == "labels" and layer.name == "det"
    assert tuple(layer.axis_labels) == ("I", "P", "L")
    on = np.argwhere(np.asarray(layer.data))
    world = np.array([layer.data_to_world(i) for i in (on.min(0), on.max(0))])
    expected = np.array([affine[:3, :3] @ c + affine[:3, 3] for c in ([1, 2, 3], [4, 6, 8])])
    np.testing.assert_allclose(np.sort(world, 0), np.sort(expected, 0))
    assert layer.features["box"][0] == "tumor (1) 0.875"
    assert layer.features["score"][0] == 0.875


def test_2d_boxes_become_rectangles_with_text():
    viewer = _add(bbox_layers([[1, 2], [5, 5]], [[3, 6], [8, 9]], labels=["a", "b"]))
    layer = viewer.layers[0]
    assert layer._type_string == "shapes" and list(layer.shape_type) == ["rectangle", "rectangle"]
    np.testing.assert_allclose(layer.data[0], [[1, 2], [1, 6], [3, 6], [3, 2]])
    assert list(layer.features["box"]) == ["a", "b"]


def test_colours_per_class():
    colors = box_colors([1, 2, 1], {2: [0, 255, 0]})
    np.testing.assert_allclose(colors[1], [0, 1, 0, 1])
    np.testing.assert_allclose(colors[0], colors[2])
    assert not np.allclose(colors[0], colors[1])
    assert box_texts([1], [None], {"1": "liver"}, 1) == ["liver (1)"]


def test_default_grid_and_validation():
    data, _, _ = bbox_layers([[0, 0, 0]], [[2, 3, 4]])[0]
    assert data.shape == (3, 4, 5)
    with pytest.raises(ValueError):
        bbox_layers([[0, 0, 0, 0]], [[1, 1, 1, 1]])
    with pytest.raises(ValueError):
        bbox_layers([[0, 0]], [[1, 1]], labels=[1, 2])


def test_add_bboxes_returns_layers():
    viewer = ViewerModel()
    layers = add_bboxes(viewer, [[0, 0, 0]], [[1, 1, 1]], grid_shape=(3, 3, 3))
    assert len(layers) == 1 and layers[0] in viewer.layers
