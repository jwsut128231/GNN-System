import torch
from app.models._lr import DEFAULT_LR_GAMMA, build_scheduler


def test_exponential_lr_decay():
    model = torch.nn.Linear(2, 1)
    base_lr = 0.01
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    sched = build_scheduler(opt)
    for _ in range(10):
        sched.step()
    expected = base_lr * (DEFAULT_LR_GAMMA ** 10)
    actual = opt.param_groups[0]["lr"]
    assert abs(actual - expected) < 1e-6, f"expected {expected}, got {actual}"
