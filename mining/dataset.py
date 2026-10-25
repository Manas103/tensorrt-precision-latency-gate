"""Synthetic 32x32 driving-frame generator: 4 scenario classes (normal, low_light,
wet_surface, construction_zone), deliberately imbalanced so "normal" dominates and
the other three are genuinely rare, which is the entire reason a mining step (rather
than uniform sampling) matters.

Every frame carries its true scenario label and an `is_rare` flag from construction,
not from any model's output. These are used two ways, both disclosed in the README:
as the "hand-checked" ground truth for the 1,000-frame label audit (a scripted proxy
for a human checker, the same disclosed precedent as ground-truth-labeling-autolabeler-eval's
scripted annotator), and as the test-time ground truth the trained detector's low-light
recall is measured against. Neither use is hidden from the other; a VLM label-audit
score and a downstream recall score both ultimately trace back to this same generator.
"""
from __future__ import annotations

import numpy as np

IMAGE_SIZE = 32
CLASSES = ("normal", "low_light", "wet_surface", "construction_zone")
CLASS_PROBS = (0.85, 0.06, 0.05, 0.04)
RARE_CLASSES = ("low_light", "wet_surface", "construction_zone")

# Disjoint seed ranges so the corpus, the calibration set and the held-out test set never
# overlap. CORPUS_SEED_RANGE is the 200,000-frame full-scale corpus every frame of which is
# scored by the VLM; the 1,000-frame label audit (see label_audit.py) samples from within it,
# after scoring. CALIB and TEST are drawn from ranges strictly beyond it.
CORPUS_SEED_RANGE = (0, 200_000)
CALIB_SEED_RANGE = (200_000, 200_512)
TEST_SEED_RANGE = (210_000, 220_000)


def _road_base(rng: np.random.Generator) -> np.ndarray:
    """A flat asphalt-grey background with two vertical lane-line stripes."""
    base_gray = rng.uniform(0.35, 0.5)
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), base_gray, dtype=np.float32)
    img += rng.normal(0, 0.02, size=img.shape).astype(np.float32)
    for lane_x in (IMAGE_SIZE // 3, 2 * IMAGE_SIZE // 3):
        img[:, lane_x - 1:lane_x + 1, :] = [0.9, 0.9, 0.85]
    return np.clip(img, 0, 1)


def make_frame(seed: int) -> tuple[np.ndarray, str]:
    """Deterministic in seed. Returns (HWC float32 image in [0, 1], class name)."""
    rng = np.random.default_rng(seed)
    cls = rng.choice(CLASSES, p=CLASS_PROBS)
    img = _road_base(rng)

    if cls == "normal":
        pass
    elif cls == "low_light":
        img = img * rng.uniform(0.08, 0.22)
        img += rng.normal(0, 0.01, size=img.shape).astype(np.float32)
    elif cls == "wet_surface":
        # The entire frame gets a strong blue-grey sheen plus several bright, near-white
        # horizontal reflection bands spanning its full width: a mirror-like glossy look,
        # pushed to cover the whole frame rather than a strip of it, on the same
        # "the signal must survive heavy downsampling" reasoning as construction_zone below
        # (see Findings for the two earlier, too-subtle versions of this that a VLM could not
        # tell apart from "normal").
        img *= 0.6
        img[:, :, 2] = np.clip(img[:, :, 2] + 0.45, 0, 1)
        for _ in range(7):
            y = rng.integers(0, IMAGE_SIZE)
            width = rng.integers(1, 3)
            img[max(0, y - width):y + width, :, :] = np.clip(
                img[max(0, y - width):y + width, :, :] + rng.uniform(0.5, 0.7), 0, 1
            )
    elif cls == "construction_zone":
        # Several large, saturated orange/white cone-striped blocks covering a sizeable
        # fraction of the frame, rather than a few small patches, for the same
        # "the signal must survive aggressive resizing" reason as wet_surface above.
        for _ in range(6):
            cx = rng.integers(2, IMAGE_SIZE - 2)
            cy = rng.integers(2, IMAGE_SIZE - 2)
            half = rng.integers(3, 6)
            y0, y1 = max(0, cy - half), min(IMAGE_SIZE, cy + half)
            x0, x1 = max(0, cx - half), min(IMAGE_SIZE, cx + half)
            patch = img[y0:y1, x0:x1, :]
            stripe = (np.arange(patch.shape[0]) // 2) % 2 == 0
            patch[stripe, :, :] = [0.98, 0.55, 0.02]
            patch[~stripe, :, :] = [0.95, 0.95, 0.9]

    return np.clip(img, 0, 1).astype(np.float32), str(cls)


def make_batch(seeds: np.ndarray | range) -> tuple[np.ndarray, list[str]]:
    images = []
    labels = []
    for s in seeds:
        img, cls = make_frame(int(s))
        images.append(img)
        labels.append(cls)
    return np.stack(images), labels


def to_chw(images_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(images_hwc, (0, 3, 1, 2)).copy()


SENSOR_NOISE_STD = 0.10


def add_sensor_noise(images_chw: np.ndarray, seed: int) -> np.ndarray:
    """Adds fixed-seed Gaussian observation noise, used only by the detector's train/eval
    pipeline (mining/train.py, mining/evaluate.py), never by the VLM scoring pipeline
    (mining/corpus_mine.py sees the clean images). This is what makes the low-light-recall
    comparison between the mined and uniform training sets actually sensitive to how many
    low-light examples each strategy's training set contains: on the clean images, a CNN
    separates low_light from normal by brightness alone so cleanly that a handful of examples
    is already enough regardless of sampling strategy (see Findings for the first, too-easy
    version of this experiment), which would make "how the training set was sampled" not
    matter and defeat the entire point of comparing mined against uniform sampling.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, SENSOR_NOISE_STD, size=images_chw.shape).astype(np.float32)
    return np.clip(images_chw + noise, 0, 1).astype(np.float32)


def is_rare(labels: list[str]) -> np.ndarray:
    return np.array([1 if c in RARE_CLASSES else 0 for c in labels], dtype=np.int64)
