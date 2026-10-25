import numpy as np

from mining.label_audit import audit
from mining.sampling import mined_training_seeds, uniform_training_seeds


def _fake_scores(tmp_path, n=2000, rare_frac=0.2, true_positive_rate=0.9):
    rng = np.random.default_rng(0)
    seeds = np.arange(n)
    true_is_rare = rng.random(n) < rare_frac
    true_labels = np.where(true_is_rare, "low_light", "normal")

    pred_is_rare = true_is_rare.copy()
    flip_fp = rng.random(n) < 0.05
    flip_fn = rng.random(n) < (1 - true_positive_rate)
    pred_is_rare = np.where(true_is_rare, ~(flip_fn & true_is_rare) & true_is_rare, flip_fp)
    pred_labels = np.where(pred_is_rare, "low_light", "normal")

    path = tmp_path / "scores.npz"
    np.savez(path, seeds=seeds, true_labels=true_labels, pred_labels=pred_labels,
             rare_probs=np.zeros(n, dtype=np.float32))
    return str(path)


def test_mined_training_seeds_are_half_rare_when_enough_available(tmp_path):
    path = _fake_scores(tmp_path, n=2000, rare_frac=0.3)
    seeds = mined_training_seeds(path, n_total=200)
    data = np.load(path, allow_pickle=True)
    pred_labels = data["pred_labels"]
    seed_to_idx = {int(s): i for i, s in enumerate(data["seeds"])}
    chosen_pred = [pred_labels[seed_to_idx[int(s)]] for s in seeds]
    rare_count = sum(1 for p in chosen_pred if p != "normal")
    assert rare_count == 100
    assert len(seeds) == 200
    assert len(set(seeds.tolist())) == 200


def test_uniform_training_seeds_right_size_no_duplicates(tmp_path):
    path = _fake_scores(tmp_path)
    seeds = uniform_training_seeds(path, n_total=150)
    assert len(seeds) == 150
    assert len(set(seeds.tolist())) == 150


def test_mined_seeds_falls_back_to_all_available_when_corpus_is_small(tmp_path):
    path = _fake_scores(tmp_path, n=50, rare_frac=0.3)
    seeds = mined_training_seeds(path, n_total=10_000)
    assert len(seeds) <= 50
    assert len(set(seeds.tolist())) == len(seeds)


def test_audit_precision_is_well_defined(tmp_path):
    path = _fake_scores(tmp_path, n=5000, rare_frac=0.3)
    result = audit(path, n=500)
    assert result["num_audited"] == 500
    assert 0.0 <= result["precision"] <= 1.0
    assert result["num_true_positive"] <= 500
