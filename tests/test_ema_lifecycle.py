from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from utils.training.lightning_module import (
    BasicLightningModule,
    EMARestoreOnException,
)


class _Module(BasicLightningModule):
    def __init__(self, *, debug=False):
        cfg = SimpleNamespace(
            model=SimpleNamespace(ema_decay=0.9),
            debug=debug,
            trainer=SimpleNamespace(log_every_n_steps=50),
        )
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


def test_prefit_inference_validation_keeps_ema_trainable():
    module = _Module()
    _set_training_and_ema(module)

    # Reproduce the persistent storage type created when an EMA transfer or
    # checkpoint load happened under Lightning's inference-mode validation.
    with torch.inference_mode():
        module.ema.shadow_params = [
            value.to(dtype=torch.float64).to(dtype=value.dtype)
            for value in module.ema.shadow_params
        ]
    assert all(torch.is_inference(value) for value in module.ema.shadow_params)

    with torch.inference_mode():
        module._activate_ema_parameters()
        module._restore_training_parameters()
    module.on_fit_start()

    assert all(
        not torch.is_inference(value) for value in module.ema.shadow_params
    )
    module.ema.update()


def test_checkpoint_loading_inside_inference_mode_keeps_ema_trainable():
    source = _Module()
    _set_training_and_ema(source)
    checkpoint = {}
    source.on_save_checkpoint(checkpoint)

    target = _Module()
    with torch.inference_mode():
        target.on_load_checkpoint(checkpoint)

    assert all(
        not torch.is_inference(value) for value in target.ema.shadow_params
    )
    target.ema.update()


def test_detailed_numerical_checks_are_periodic_and_debug_forces_them():
    module = _Module()
    assert module._detailed_numerical_check_due(step=0, is_training=True)
    assert not module._detailed_numerical_check_due(step=1, is_training=True)
    assert module._detailed_numerical_check_due(step=500, is_training=True)
    assert module._detailed_numerical_check_due(step=1, is_training=False)

    debug_module = _Module(debug=True)
    assert debug_module._detailed_numerical_check_due(
        step=1, is_training=True
    )


def test_numerical_scope_temporarily_changes_opted_in_modules():
    module = _Module()
    module.model.numerical_validation_enabled = True
    module._detailed_numerical_check_due = lambda **_: False

    with module._numerical_validation_scope(is_training=True):
        assert not module.model.numerical_validation_enabled
        assert not module._detailed_numerical_validation_active

    assert module.model.numerical_validation_enabled
    assert module._detailed_numerical_validation_active


def test_final_loss_guard_remains_active_each_step():
    finite = torch.tensor(1.0)
    assert BasicLightningModule._validate_final_loss({"total": finite}) is finite
    with pytest.raises(FloatingPointError, match="non-finite"):
        BasicLightningModule._validate_final_loss(
            {"total": torch.tensor(float("nan"))}
        )


def test_cpu_step_timer_reports_forward_and_full_step_time():
    module = _Module()
    module._start_step_timing()
    module.model(torch.ones(1, 2))
    module._mark_forward_timing()
    timings = module._finish_step_timing()

    assert set(timings) == {"net_time", "step_time", "step_wall_time"}
    assert 0.0 <= timings["net_time"] <= timings["step_time"]
    assert timings["step_time"] <= timings["step_wall_time"]
