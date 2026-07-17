from types import SimpleNamespace

import torch

from utils.training.memory import estimate_training_memory


class TinyTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.frozen = torch.nn.Parameter(
            torch.zeros(4, dtype=torch.float32), requires_grad=False
        )
        self.register_buffer("statistics", torch.zeros(3, dtype=torch.float32))
        self.ema = SimpleNamespace(shadow_params=[self.trainable.detach().clone()])


def test_adamw_memory_estimate_separates_fixed_and_possible_ddp_bytes():
    module = TinyTrainingModule()
    optimizer = torch.optim.AdamW([module.trainable])
    estimate = estimate_training_memory(module, optimizer, world_size=8)

    assert estimate.resident_bytes == (6 + 4 + 3) * 4
    assert estimate.gradient_bytes == 6 * 4
    assert estimate.optimizer_bytes == 2 * 6 * 4
    assert estimate.ema_bytes == 6 * 4
    assert estimate.ddp_bucket_bytes == 6 * 4
    assert estimate.fixed_lower_bound_bytes == (13 + 6 + 12 + 6) * 4
    assert estimate.fixed_with_ddp_bucket_bytes == (13 + 6 + 12 + 6 + 6) * 4


def test_single_device_memory_estimate_has_no_ddp_bucket_allowance():
    module = TinyTrainingModule()
    optimizer = torch.optim.AdamW([module.trainable])
    estimate = estimate_training_memory(module, optimizer, world_size=1)

    assert estimate.ddp_bucket_bytes == 0
