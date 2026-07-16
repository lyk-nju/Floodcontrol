from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from utils.training.lightning_module import (
    BasicLightningModule,
    EMARestoreOnException,
)


class _Module(BasicLightningModule):
    def __init__(self):
        cfg = SimpleNamespace(model=SimpleNamespace(ema_decay=0.9))
        super().__init__(cfg, model=nn.Linear(2, 2, bias=False))

    def _step(self, batch, is_training=True):
        return {"total": self.model(batch).square().mean()}


def _set_training_and_ema(module):
    with torch.no_grad():
        module.model.weight.fill_(1.0)
        module.ema.shadow_params[0].fill_(2.0)


def test_ema_scope_is_nested_and_restores_training_parameters():
    module = _Module()
    _set_training_and_ema(module)
    training = module.model.weight.detach().clone()

    with module.use_ema_parameters():
        assert module._ema_parameters_active
        assert torch.equal(module.model.weight, torch.full_like(training, 2.0))
        collected = module.ema.collected_params
        with module.use_ema_parameters():
            assert module.ema.collected_params is collected
            assert torch.equal(module.model.weight, torch.full_like(training, 2.0))
        assert module.ema.collected_params is collected

    assert torch.equal(module.model.weight, training)
    assert module.ema.collected_params is None
    assert not module._ema_parameters_active


def test_checkpoint_never_serializes_temporary_ema_parameters():
    module = _Module()
    _set_training_and_ema(module)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["ema_state"]["collected_params"] is None
    assert torch.equal(checkpoint["state_dict"]["weight"], module.model.weight)

    module._activate_ema_parameters()
    with pytest.raises(RuntimeError, match="EMA parameters are active"):
        module.on_save_checkpoint({})
    module._restore_training_parameters()


def test_loading_old_ema_state_releases_stale_collected_parameters():
    source = _Module()
    _set_training_and_ema(source)
    source._activate_ema_parameters()
    old_ema_state = dict(source.ema.state_dict())
    source._restore_training_parameters()

    target = _Module()
    target.on_load_checkpoint(
        {"state_dict": source.model.state_dict(), "ema_state": old_ema_state}
    )
    assert target.ema.collected_params is None
    assert not target._ema_parameters_active


def test_exception_callback_restores_training_parameters():
    module = _Module()
    _set_training_and_ema(module)
    training = module.model.weight.detach().clone()
    module._activate_ema_parameters()

    EMARestoreOnException().on_exception(None, module, RuntimeError("boom"))

    assert torch.equal(module.model.weight, training)
    assert module.ema.collected_params is None
    assert not module._ema_parameters_active
