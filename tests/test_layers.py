import numpy as np
import pytest
from napari.components import ViewerModel

from napari_bboxes import BoxShells, add_bboxes, bbox_layers, box_colors, box_texts


def _add(layer_data):
    viewer = ViewerModel()
    for data, kwargs, layer_type in layer_data:
        viewer._add_layer_from_data(data, kwargs, layer_type)
    return viewer


def test_faces_are_hollow_box_outlines():
    shells = BoxShells((8, 9, 10), np.array([[[1, 4], [2, 6], [3, 8]]]), mode="faces")
    full = np.asarray(shells)
    on = np.argwhere(full)
    assert on.min(0).tolist() == [1, 2, 3] and on.max(0).tolist() == [4, 6, 8]
    assert not full[2:4, 3:6, 4:8].any()                 # interior empty
    assert full[1, 2:7, 3:9].all() and full[4, 2:7, 3:9].all()   # end faces of dim 0 filled
    assert full[2, 2, 3:9].all() and not full[2, 3, 4]   # middle slice: only the rectangle outline


def test_edges_are_a_wireframe():
    shells = BoxShells((8, 9, 10), np.array([[[1, 4], [2, 6], [3, 8]]]), mode="edges")
    full = np.asarray(shells)
    assert full[1, 2, 3:9].all() and full[1:5, 6, 8].all() and full[4, 2:7, 3].all()   # edges
    assert not full[1, 4, 5] and not full[2, 2, 5]       # face centres empty
    assert int((full > 0).sum()) == 4 * (4 + 5 + 6) - 2 * 8   # 12 edges, corners counted once


def test_auto_draws_rectangle_outlines_in_slices_and_edges_for_volumes():
    lo, hi = np.array([1, 2, 3]), np.array([4, 6, 8])
    auto = BoxShells((8, 9, 10), np.stack([lo, hi], -1)[None])
    np.testing.assert_array_equal(np.asarray(auto), np.asarray(BoxShells((8, 9, 10), np.stack([lo, hi], -1)[None],
                                                                         mode="edges")))   # napari's 3-D view
    for d in range(3):                                  # napari's 2-D slices in every plane, incl. the face planes
        others = [e for e in range(3) if e != d]
        for k in range(auto.shape[d]):
            expected = np.zeros([auto.shape[e] for e in others], np.uint16)
            if lo[d] <= k <= hi[d]:
                (a0, a1), (b0, b1) = [(lo[e], hi[e]) for e in others]
                expected[a0:a1 + 1, b0] = expected[a0:a1 + 1, b1] = 1
                expected[a0, b0:b1 + 1] = expected[a1, b0:b1 + 1] = 1
            np.testing.assert_array_equal(auto[(slice(None),) * d + (k,)], expected)


@pytest.mark.parametrize("key", [
    (slice(None),), (2,), (Ellipsis, 5), (slice(1, 5), 3), (-1,), (slice(None, None, 2), slice(None), 4),
    (slice(None, None, -1), 2), (3, slice(1, 7, 3), slice(None)), (slice(2, 6), slice(1, 8), slice(4, 9)),
])
@pytest.mark.parametrize("mode", ["faces", "edges"])
def test_shell_slices_match_the_full_array(key, mode):
    if mode == "faces" and any(isinstance(k, int) for k in key):
        pytest.skip("faces: a single-position slice shows the cut outline, not the face (tested above)")
    shells = BoxShells((8, 9, 10), np.array([[[1, 4], [2, 6], [3, 8]], [[3, 7], [0, 8], [5, 9]]]), mode=mode)
    full = np.asarray(shells)
    np.testing.assert_array_equal(shells[key], full[key])


def test_overlapping_boxes_later_wins_and_numbers_are_one_based():
    shells = BoxShells((5, 5, 5), np.array([[[0, 4]] * 3, [[1, 3]] * 3]), mode="faces")
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
    on = np.argwhere(np.asarray(layer.data))           # the edges span the whole box too
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
    assert len(layers) == 1 and layers[0] in viewer.layers   # no labels or scores: no text layer


def test_texts_follow_the_slice_and_show_in_3d():
    viewer = ViewerModel()
    shells, texts = add_bboxes(viewer, [[2, 3, 4], [6, 1, 1]], [[4, 6, 8], [8, 2, 2]], labels=[1, 2],
                               scores=[0.9, 0.25], names={1: "tumor", 2: "node"}, grid_shape=(10, 10, 10))
    assert texts._type_string == "points" and list(texts.features["box"]) == ["tumor (1) 0.900", "node (2) 0.250"]
    viewer.dims.order = (0, 1, 2)                       # axial: dim 0 hidden
    viewer.dims.set_point(0, 3)
    assert list(texts.shown) == [True, False]
    pad = 0.015 * 10                                        # TEXT_PAD x largest world extent (10 voxels of 1)
    np.testing.assert_allclose(texts.data[0], [3, 3 - pad, 4])   # slice position, above the top-left of the cut
    viewer.dims.set_point(0, 7)
    assert list(texts.shown) == [False, True]
    viewer.dims.order = (1, 0, 2)                       # coronal: dim 1 hidden
    viewer.dims.set_point(1, 5)
    assert list(texts.shown) == [True, False]
    np.testing.assert_allclose(texts.data[0], [2 - pad, 5, 4])
    viewer.dims.ndisplay = 3
    assert all(texts.shown)
    np.testing.assert_allclose(texts.data, np.array([[2, 3, 4], [6, 1, 1]]) - pad)


def test_no_text_layer_without_labels_or_scores():
    assert len(bbox_layers([[0, 0, 0]], [[1, 1, 1]])) == 1
    assert len(bbox_layers([[0, 0, 0]], [[1, 1, 1]], labels=[1], text=False)) == 1


def test_texts_sit_at_the_top_left_on_screen_when_the_affine_flips():
    viewer = ViewerModel()
    affine = np.diag([-1.0, 1.0, -2.0, 1.0])
    _, texts = add_bboxes(viewer, [[2, 3, 4]], [[4, 6, 8]], labels=[1], grid_shape=(10, 10, 10), affine=affine)
    viewer.dims.order = (0, 1, 2)
    viewer.dims.set_point(0, -3)                       # world dim 0 = -index
    assert list(texts.shown) == [True]
    world = texts.data_to_world(texts.data[0])
    lo_world = np.array([texts.data_to_world(c) for c in ([2, 3, 4], [4, 6, 8])]).min(0)
    pad = 0.015 * 20                                       # largest world extent: 10 voxels x 2
    np.testing.assert_allclose(world[1:], lo_world[1:] - [pad, 0])   # above the top-left corner on screen
