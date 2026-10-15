import torch

from gate import per_class_logit_diff, ACCURACY_BUDGET


def test_per_class_logit_diff_zero_for_identical_tensors():
    t = torch.randn(20, 10)
    diff = per_class_logit_diff(t, t)
    assert diff["max_abs_diff_overall"] == 0.0
    assert diff["mean_abs_diff_overall"] == 0.0
    assert all(v == 0.0 for v in diff["per_class_max_abs_diff"])


def test_per_class_logit_diff_isolates_the_corrupted_class():
    framework = torch.zeros(5, 3)
    engine = framework.clone()
    engine[:, 1] += 2.0  # only class 1's column is corrupted

    diff = per_class_logit_diff(engine, framework, num_classes=3)
    assert diff["per_class_max_abs_diff"][0] == 0.0
    assert diff["per_class_max_abs_diff"][1] == 2.0
    assert diff["per_class_max_abs_diff"][2] == 0.0


def test_accuracy_budget_is_one_percent():
    # Not a tautology: this pins the constant gate.py actually enforces,
    # so an accidental edit to ACCURACY_BUDGET shows up as a failing test
    # rather than silently changing what "blocked" means.
    assert ACCURACY_BUDGET == 0.01


def test_blocked_decision_matches_the_documented_rule():
    framework_top1 = 0.80

    def blocked(engine_top1):
        return (framework_top1 - engine_top1) > ACCURACY_BUDGET

    assert not blocked(0.795)   # 0.5pt drop, within budget
    assert not blocked(0.80)    # no drop
    assert blocked(0.78)        # 2pt drop, breaches budget
    assert blocked(0.60)        # a catastrophic drop must also block
