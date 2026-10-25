import numpy as np

from mining.dataset import CLASSES, is_rare, make_batch, make_frame, to_chw


def test_deterministic_for_same_seed():
    img_a, cls_a = make_frame(42)
    img_b, cls_b = make_frame(42)
    assert cls_a == cls_b
    assert np.array_equal(img_a, img_b)


def test_different_seeds_can_differ():
    classes = {make_frame(s)[1] for s in range(200)}
    assert len(classes) > 1


def test_class_is_always_valid():
    for s in range(100):
        _, cls = make_frame(s)
        assert cls in CLASSES


def test_image_shape_and_range():
    img, _ = make_frame(1)
    assert img.shape == (32, 32, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_normal_class_dominates_a_large_sample():
    _, labels = make_batch(range(5000))
    counts = {c: labels.count(c) for c in CLASSES}
    assert counts["normal"] > sum(counts[c] for c in CLASSES if c != "normal")


def test_to_chw_transposes_correctly():
    images_hwc, _ = make_batch(range(4))
    images_chw = to_chw(images_hwc)
    assert images_chw.shape == (4, 3, 32, 32)
    assert np.allclose(images_chw[0, :, 0, 0], images_hwc[0, 0, 0, :])


def test_is_rare_flags_non_normal_classes():
    labels = ["normal", "low_light", "wet_surface", "construction_zone", "normal"]
    assert list(is_rare(labels)) == [0, 1, 1, 1, 0]
